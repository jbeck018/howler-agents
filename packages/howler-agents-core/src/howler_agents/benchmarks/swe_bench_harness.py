"""SWE-Bench harness for howler-agents evaluation.

Loads SWE-bench instances from HuggingFace, manages Docker containers for
repo checkouts, and produces predictions in the standard SWE-bench format.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SWEBenchInstance:
    """A single SWE-bench task instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""  # gold patch (for reference, not shown to agent)
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    version: str = ""


@dataclass
class SWEBenchPrediction:
    """Agent's prediction for a single instance."""

    instance_id: str
    model_name_or_path: str
    model_patch: str


@dataclass
class SWEBenchResult:
    """Evaluation result for a single instance."""

    instance_id: str
    resolved: bool
    error: str | None = None
    fail_to_pass_results: dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: dict[str, bool] = field(default_factory=dict)


class SWEBenchHarness:
    """Manages SWE-bench dataset loading, repo preparation, and evaluation.

    Usage:
        harness = SWEBenchHarness(dataset="princeton-nlp/SWE-bench_Lite")
        instances = harness.load_instances(limit=10)
        # ... run agents to produce predictions ...
        harness.write_predictions(predictions, "predictions.json")
        results = harness.evaluate("predictions.json", run_id="eval-001")
    """

    def __init__(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        workspace: str | Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._split = split
        self._workspace = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="howler-swe-"))
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._instances: list[SWEBenchInstance] = []

    @property
    def workspace(self) -> Path:
        return self._workspace

    def load_instances(
        self,
        limit: int | None = None,
        instance_ids: list[str] | None = None,
    ) -> list[SWEBenchInstance]:
        """Load SWE-bench instances from HuggingFace.

        Args:
            limit: Max number of instances to load.
            instance_ids: Only load these specific instances.

        Returns:
            List of SWEBenchInstance objects.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the datasets package: pip install datasets\n"
                "Required for loading SWE-bench instances from HuggingFace."
            ) from exc

        logger.info("loading_swebench", dataset=self._dataset, split=self._split)
        ds = load_dataset(self._dataset, split=self._split)

        instances = []
        for item in ds:
            iid = item["instance_id"]
            if instance_ids and iid not in instance_ids:
                continue

            fail_to_pass = item.get("FAIL_TO_PASS", "[]")
            pass_to_pass = item.get("PASS_TO_PASS", "[]")

            instance = SWEBenchInstance(
                instance_id=iid,
                repo=item["repo"],
                base_commit=item["base_commit"],
                problem_statement=item["problem_statement"],
                patch=item.get("patch", ""),
                test_patch=item.get("test_patch", ""),
                fail_to_pass=json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass,
                pass_to_pass=json.loads(pass_to_pass) if isinstance(pass_to_pass, str) else pass_to_pass,
                version=item.get("version", ""),
            )
            instances.append(instance)

            if limit and len(instances) >= limit:
                break

        self._instances = instances
        logger.info("swebench_loaded", count=len(instances))
        return instances

    def checkout_repo(self, instance: SWEBenchInstance) -> Path:
        """Clone and checkout a repo at the base_commit for an instance.

        Returns the path to the checked-out repository.
        """
        repo_dir = self._workspace / "repos" / instance.instance_id.replace("/", "__")

        if repo_dir.exists():
            # Reset to base commit if already cloned
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=repo_dir,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=repo_dir,
                capture_output=True,
                check=True,
            )
            return repo_dir

        repo_url = f"https://github.com/{instance.repo}.git"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        logger.info("cloning_repo", repo=instance.repo, commit=instance.base_commit[:8])
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(repo_dir)],
            capture_output=True,
            check=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "checkout", instance.base_commit],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        return repo_dir

    def get_repo_context(
        self,
        repo_dir: Path,
        instance: SWEBenchInstance,
        max_files: int = 50,
    ) -> str:
        """Build context string from the repo for the agent.

        Provides the directory structure and relevant file contents
        to help the agent understand the codebase.
        """
        # Get directory structure
        result = subprocess.run(
            ["find", ".", "-name", "*.py", "-not", "-path", "./.git/*"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        py_files = sorted(result.stdout.strip().splitlines()[:max_files])

        context_parts = [
            f"Repository: {instance.repo}",
            f"Commit: {instance.base_commit[:12]}",
            f"\nPython files ({len(py_files)} shown):",
            "\n".join(py_files),
        ]

        return "\n".join(context_parts)

    def write_predictions(
        self,
        predictions: list[SWEBenchPrediction],
        output_path: str | Path,
    ) -> Path:
        """Write predictions in SWE-bench JSONL format."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "instance_id": p.instance_id,
                "model_name_or_path": p.model_name_or_path,
                "model_patch": p.model_patch,
            }
            for p in predictions
        ]

        with output.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info("predictions_written", path=str(output), count=len(predictions))
        return output

    def evaluate(
        self,
        predictions_path: str | Path,
        run_id: str = "howler-eval",
        max_workers: int = 4,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Run SWE-bench evaluation on predictions.

        Requires the swebench package and Docker to be installed.

        Returns dict with:
            - total: total instances
            - submitted: instances with predictions
            - resolved: instances that passed
            - resolved_rate: percentage resolved
            - per_instance: list of per-instance results
        """
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self._dataset,
            "--predictions_path", str(predictions_path),
            "--max_workers", str(max_workers),
            "--run_id", run_id,
        ]

        if namespace is not None:
            cmd.extend(["--namespace", namespace])

        # On Apple Silicon, use empty namespace for local builds
        import platform
        if platform.machine() == "arm64" and namespace is None:
            cmd.extend(["--namespace", ""])

        logger.info("starting_evaluation", run_id=run_id, cmd=" ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode != 0:
            logger.error("evaluation_failed", stderr=result.stderr[:500])
            return {
                "total": len(self._instances),
                "submitted": 0,
                "resolved": 0,
                "resolved_rate": 0.0,
                "error": result.stderr[:1000],
                "per_instance": [],
            }

        # Parse results
        results_dir = Path(f"evaluation_results/{run_id}")
        return self._parse_results(results_dir)

    def _parse_results(self, results_dir: Path) -> dict[str, Any]:
        """Parse evaluation results from the SWE-bench output directory."""
        results_file = results_dir / "results.json"

        if not results_file.exists():
            # Try to find results in the output
            logger.warning("results_file_not_found", path=str(results_file))
            return {
                "total": len(self._instances),
                "submitted": 0,
                "resolved": 0,
                "resolved_rate": 0.0,
                "per_instance": [],
            }

        with results_file.open() as f:
            raw = json.load(f)

        # Parse per-instance results
        per_instance = []
        resolved_ids = set(raw.get("resolved", []))

        for inst in self._instances:
            per_instance.append({
                "instance_id": inst.instance_id,
                "resolved": inst.instance_id in resolved_ids,
                "repo": inst.repo,
            })

        total = raw.get("total", len(self._instances))
        submitted = raw.get("submitted", 0)
        resolved = len(resolved_ids)

        return {
            "total": total,
            "submitted": submitted,
            "resolved": resolved,
            "resolved_rate": (resolved / submitted * 100) if submitted > 0 else 0.0,
            "per_instance": per_instance,
        }

    def validate_setup(self) -> dict[str, Any]:
        """Check that all prerequisites are met for SWE-bench evaluation.

        Returns a dict with status for each requirement.
        """
        checks: dict[str, Any] = {}

        # 1. Check datasets package
        try:
            import datasets  # noqa: F401
            checks["datasets_package"] = {"ok": True}
        except ImportError:
            checks["datasets_package"] = {"ok": False, "fix": "pip install datasets"}

        # 2. Check swebench package
        try:
            import swebench  # noqa: F401
            checks["swebench_package"] = {"ok": True}
        except ImportError:
            checks["swebench_package"] = {"ok": False, "fix": "pip install swebench"}

        # 3. Check Docker
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            checks["docker"] = {"ok": result.returncode == 0}
            if result.returncode != 0:
                checks["docker"]["fix"] = "Start Docker Desktop or install Docker"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            checks["docker"] = {"ok": False, "fix": "Install Docker"}

        # 4. Check git
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=5)
            checks["git"] = {"ok": True}
        except (FileNotFoundError, subprocess.CalledProcessError):
            checks["git"] = {"ok": False, "fix": "Install git"}

        # 5. Check disk space
        import shutil
        _total, _used, free = shutil.disk_usage(self._workspace)
        free_gb = free / (1024**3)
        checks["disk_space"] = {
            "ok": free_gb >= 20,
            "free_gb": round(free_gb, 1),
            "minimum_gb": 20,
        }

        all_ok = all(c["ok"] for c in checks.values())
        checks["ready"] = all_ok

        return checks
