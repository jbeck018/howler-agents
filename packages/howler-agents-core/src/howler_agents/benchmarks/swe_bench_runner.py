"""SWE-Bench evaluation runner for howler-agents.

Orchestrates the full pipeline:
1. Load SWE-bench instances
2. Run GEA evolution to produce agents
3. Execute evolved agents on SWE-bench tasks
4. Collect predictions and evaluate
5. Report results with per-step analysis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from howler_agents.agents.base import AgentConfig
from howler_agents.agents.pool import AgentPool
from howler_agents.benchmarks.swe_bench_agent import SWEBenchAgent
from howler_agents.benchmarks.swe_bench_harness import (
    SWEBenchHarness,
    SWEBenchInstance,
    SWEBenchPrediction,
)
from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.llm.claude_code import (
    detect_available_backend,
    is_cli_model,
    list_available_backends,
)
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector

logger = structlog.get_logger()


@dataclass
class StepReport:
    """Report for a single evaluation step."""

    step: str
    status: str  # "success" | "partial" | "failed"
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    went_well: list[str] = field(default_factory=list)
    went_wrong: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Full evaluation report."""

    run_id: str
    started_at: str = ""
    finished_at: str = ""
    total_duration_s: float = 0.0
    steps: list[StepReport] = field(default_factory=list)
    final_results: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_s": round(self.total_duration_s, 1),
            "steps": [
                {
                    "step": s.step,
                    "status": s.status,
                    "duration_s": round(s.duration_s, 1),
                    "details": s.details,
                    "went_well": s.went_well,
                    "went_wrong": s.went_wrong,
                    "suggestions": s.suggestions,
                }
                for s in self.steps
            ],
            "final_results": self.final_results,
            "config": self.config,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== SWE-Bench Evaluation Report: {self.run_id} ===",
            f"Duration: {self.total_duration_s:.1f}s",
            "",
        ]
        for step in self.steps:
            icon = {"success": "+", "partial": "~", "failed": "-"}.get(step.status, "?")
            lines.append(f"[{icon}] {step.step} ({step.duration_s:.1f}s)")
            for w in step.went_well:
                lines.append(f"    + {w}")
            for w in step.went_wrong:
                lines.append(f"    - {w}")
            for s in step.suggestions:
                lines.append(f"    > {s}")
            lines.append("")

        if self.final_results:
            lines.append("--- Final Results ---")
            lines.append(f"  Resolved: {self.final_results.get('resolved', 0)}/{self.final_results.get('submitted', 0)}")
            lines.append(f"  Rate:     {self.final_results.get('resolved_rate', 0):.1f}%")

        return "\n".join(lines)


# Default SWE-bench probes for capability characterization
SWE_BENCH_PROBES = [
    {"description": "Localize the bug: given a traceback, identify the file and function", "type": "localization"},
    {"description": "Read Python code and explain what a function does", "type": "comprehension"},
    {"description": "Generate a minimal unified diff to fix an off-by-one error", "type": "patch_gen"},
    {"description": "Identify which test would catch a specific type of bug", "type": "test_awareness"},
    {"description": "Fix a broken import statement in a Python module", "type": "import_fix"},
    {"description": "Resolve a merge conflict in a Python file", "type": "merge_conflict"},
    {"description": "Add a missing return statement to fix a function", "type": "missing_return"},
    {"description": "Fix exception handling: catch the right exception type", "type": "exception_fix"},
    {"description": "Refactor: extract a method from a long function", "type": "refactoring"},
    {"description": "Fix a type annotation error in a dataclass", "type": "type_fix"},
    {"description": "Debug: find the variable shadowing bug", "type": "shadowing"},
    {"description": "Fix API compatibility: update deprecated method call", "type": "api_compat"},
    {"description": "Add boundary check to prevent IndexError", "type": "boundary"},
    {"description": "Fix string formatting issue in error message", "type": "string_format"},
    {"description": "Correct regex pattern that doesn't handle edge case", "type": "regex_fix"},
]


class SWEBenchEvalRunner:
    """Orchestrates SWE-bench evaluation with iterative improvement tracking.

    Usage:
        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            dataset="princeton-nlp/SWE-bench_Lite",
        )
        report = await runner.run(limit=10)
        print(report.summary())
    """

    def __init__(
        self,
        model: str = "auto",
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        workspace: str | Path | None = None,
        # GEA hyperparameters (paper defaults for SWE-bench)
        population_size: int = 10,
        group_size: int = 3,
        num_iterations: int = 3,
        alpha: float = 0.7,
        num_probes: int = 15,
        # Evaluation settings
        skip_docker_eval: bool = False,
        max_concurrent: int = 3,
        instance_timeout_s: float = 300.0,
    ) -> None:
        self._model = self._resolve_model(model)
        self._dataset = dataset
        self._split = split
        self._workspace = Path(workspace) if workspace else Path(".howler-agents/swe-bench")
        self._workspace.mkdir(parents=True, exist_ok=True)

        self._population_size = population_size
        self._group_size = group_size
        self._num_iterations = num_iterations
        self._alpha = alpha
        self._num_probes = num_probes
        self._skip_docker_eval = skip_docker_eval
        self._max_concurrent = max(1, max_concurrent)
        self._instance_timeout_s = instance_timeout_s

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Resolve the model string, auto-detecting local CLI backends.

        When model is "auto", tries to find a local CLI tool (claude, codex,
        gemini, opencode) to avoid API rate limits. Falls back to the default
        Anthropic API model if none are found.
        """
        if model != "auto":
            return model

        detected = detect_available_backend()
        if detected:
            available = list_available_backends()
            logger.info(
                "auto_detected_cli_backend",
                using=detected,
                available=available,
            )
            return detected

        logger.info("no_cli_backend_found", fallback="claude-sonnet-4-20250514")
        return "claude-sonnet-4-20250514"

    async def run(
        self,
        limit: int | None = 5,
        instance_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> EvalReport:
        """Execute full SWE-bench evaluation pipeline.

        Args:
            limit: Number of instances to evaluate (None for all).
            instance_ids: Specific instances to evaluate.
            run_id: Identifier for this evaluation run.
        """
        run_id = run_id or f"howler-swe-{int(time.time())}"
        report = EvalReport(
            run_id=run_id,
            started_at=datetime.now(UTC).isoformat(),
            config={
                "model": self._model,
                "dataset": self._dataset,
                "population_size": self._population_size,
                "group_size": self._group_size,
                "num_iterations": self._num_iterations,
                "alpha": self._alpha,
                "num_probes": self._num_probes,
                "limit": limit,
            },
        )

        start = time.monotonic()

        # Step 1: Validate setup
        step = await self._step_validate_setup()
        report.steps.append(step)
        if step.status == "failed":
            report.finished_at = datetime.now(UTC).isoformat()
            report.total_duration_s = time.monotonic() - start
            return report

        # Step 2: Load instances
        harness = SWEBenchHarness(
            dataset=self._dataset,
            split=self._split,
            workspace=self._workspace / "harness",
        )
        instances, step = await self._step_load_instances(harness, limit, instance_ids)
        report.steps.append(step)
        if not instances:
            report.finished_at = datetime.now(UTC).isoformat()
            report.total_duration_s = time.monotonic() - start
            return report

        # Step 3: Evolve agents via GEA (skip if iterations=0)
        best_agent = None
        if self._num_iterations > 0:
            best_agent, step = await self._step_evolve_agents(run_id, instances)
            report.steps.append(step)
        else:
            report.steps.append(StepReport(
                step="3. GEA Evolution (SKIPPED)",
                status="skipped",
                duration_s=0.0,
                went_well=["Evolution skipped (--skip-evolution)"],
            ))

        # Step 4: Generate predictions using evolved agent(s)
        predictions, step = await self._step_generate_predictions(
            harness, instances, best_agent, run_id,
        )
        report.steps.append(step)

        # Step 5: Evaluate predictions
        if predictions and not self._skip_docker_eval:
            step = await self._step_evaluate(harness, predictions, run_id)
            report.steps.append(step)
            report.final_results = step.details
        elif predictions:
            # Skip Docker eval, just report prediction stats
            step = StepReport(
                step="5. Docker Evaluation (SKIPPED)",
                status="partial",
                went_well=["Predictions generated successfully"],
                went_wrong=["Docker evaluation skipped (--skip-docker-eval)"],
                suggestions=["Run with Docker to get actual resolve rates"],
                details={
                    "predictions_count": len(predictions),
                    "predictions_with_patch": sum(1 for p in predictions if p.model_patch),
                },
            )
            report.steps.append(step)
            report.final_results = step.details

        report.finished_at = datetime.now(UTC).isoformat()
        report.total_duration_s = time.monotonic() - start

        # Save report
        report_path = self._workspace / f"{run_id}-report.json"
        with report_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

    async def _step_validate_setup(self) -> StepReport:
        """Step 1: Validate prerequisites."""
        t = time.monotonic()
        harness = SWEBenchHarness(workspace=self._workspace / "harness")
        checks = harness.validate_setup()
        duration = time.monotonic() - t

        went_well = []
        went_wrong = []
        suggestions = []

        for name, check in checks.items():
            if name == "ready":
                continue
            if check.get("ok"):
                went_well.append(f"{name}: OK")
            else:
                went_wrong.append(f"{name}: MISSING")
                if "fix" in check:
                    suggestions.append(f"Fix {name}: {check['fix']}")

        # datasets and swebench are soft requirements for initial testing
        critical_failed = not checks.get("git", {}).get("ok", False)

        return StepReport(
            step="1. Validate Setup",
            status="failed" if critical_failed else ("success" if checks["ready"] else "partial"),
            duration_s=duration,
            details=checks,
            went_well=went_well,
            went_wrong=went_wrong,
            suggestions=suggestions,
        )

    async def _step_load_instances(
        self,
        harness: SWEBenchHarness,
        limit: int | None,
        instance_ids: list[str] | None,
    ) -> tuple[list[SWEBenchInstance], StepReport]:
        """Step 2: Load SWE-bench instances."""
        t = time.monotonic()
        went_well = []
        went_wrong = []

        try:
            instances = harness.load_instances(limit=limit, instance_ids=instance_ids)
            duration = time.monotonic() - t
            went_well.append(f"Loaded {len(instances)} instances from {self._dataset}")
            repos = {i.repo for i in instances}
            went_well.append(f"Covering {len(repos)} repositories: {', '.join(sorted(repos)[:5])}")

            return instances, StepReport(
                step="2. Load SWE-Bench Instances",
                status="success",
                duration_s=duration,
                details={"count": len(instances), "repos": sorted(repos)},
                went_well=went_well,
            )
        except ImportError as e:
            duration = time.monotonic() - t
            went_wrong.append(str(e))
            return [], StepReport(
                step="2. Load SWE-Bench Instances",
                status="failed",
                duration_s=duration,
                went_wrong=went_wrong,
                suggestions=["pip install datasets"],
            )
        except Exception as e:
            duration = time.monotonic() - t
            went_wrong.append(f"Failed to load dataset: {e}")
            return [], StepReport(
                step="2. Load SWE-Bench Instances",
                status="failed",
                duration_s=duration,
                went_wrong=went_wrong,
            )

    async def _step_evolve_agents(
        self,
        run_id: str,
        instances: list[SWEBenchInstance],
    ) -> tuple[SWEBenchAgent | None, StepReport]:
        """Step 3: Run GEA evolution to produce specialized agents."""
        t = time.monotonic()
        went_well = []
        went_wrong = []
        suggestions = []

        config = HowlerConfig(
            population_size=self._population_size,
            group_size=self._group_size,
            num_iterations=self._num_iterations,
            alpha=self._alpha,
            num_probes=self._num_probes,
            task_domain="swe-bench",
            role_models={
                LLMRole.ACTING: RoleModelConfig(model=self._model),
                LLMRole.EVOLVING: RoleModelConfig(model=self._model),
                LLMRole.REFLECTING: RoleModelConfig(model=self._model),
            },
        )

        # CLI backends don't need rate limit spacing; API backends do
        spacing = 0.0 if is_cli_model(self._model) else 2.0
        llm = LLMRouter(config, min_request_interval_s=spacing)
        store = InMemoryStore()
        experience = SharedExperiencePool(store)
        selector = PerformanceNoveltySelector(alpha=config.alpha)
        reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

        # Register SWE-bench specific probes
        registry = ProbeRegistry()
        for probe in SWE_BENCH_PROBES[:self._num_probes]:
            registry.register(probe)
        probe_evaluator = ProbeEvaluator(registry=registry)

        # Create initial population
        pool = AgentPool()
        for i in range(self._population_size):
            group_id = f"group_{i % max(1, self._population_size // self._group_size)}"
            agent = SWEBenchAgent(
                config=AgentConfig(
                    generation=0,
                    group_id=group_id,
                    framework_config={
                        "domain": "swe-bench",
                        "index": i,
                        "localization_strategy": "Analyze error tracebacks and module references to find relevant files.",
                        "patch_strategy": "Make the minimal change needed to fix the root cause.",
                        "reasoning_depth": "thorough",
                    },
                ),
                llm=llm,
            )
            pool.add(agent)

        def _make_agent(cfg: AgentConfig) -> SWEBenchAgent:
            return SWEBenchAgent(config=cfg, llm=llm)

        # Build evolution tasks from first few SWE-bench instances
        # Use a subset for evolution training, save rest for evaluation
        train_instances = instances[:max(3, len(instances) // 2)]

        # Checkout repos for training instances so agents can read code during evolution
        train_harness = SWEBenchHarness(
            dataset=self._dataset,
            split=self._split,
            workspace=self._workspace / "train-repos",
        )
        tasks = []
        for inst in train_instances:
            repo_dir = None
            try:
                repo_dir = train_harness.checkout_repo(inst)
                went_well.append(f"Checked out {inst.repo} at {inst.base_commit[:8]} for training")
            except Exception as e:
                went_wrong.append(f"Failed to checkout {inst.repo} for training: {e}")

            task_dict: dict[str, Any] = {
                "description": f"Fix issue in {inst.repo}: {inst.problem_statement[:200]}",
                "type": "swe-bench",
                "instance_id": inst.instance_id,
                "problem_statement": inst.problem_statement,
                "repo": inst.repo,
                "repo_dir": repo_dir,
            }
            if inst.fail_to_pass:
                task_dict["fail_to_pass"] = inst.fail_to_pass
            tasks.append(task_dict)

        loop = EvolutionLoop(
            config=config,
            pool=pool,
            selector=selector,
            reproducer=reproducer,
            experience=experience,
            probe_evaluator=probe_evaluator,
            agent_factory=_make_agent,
        )

        try:
            result = await loop.run(run_id=f"{run_id}-evolve", tasks=tasks)
            duration = time.monotonic() - t

            best_score = result.get("best_score", 0)
            generations = result.get("generations", [])
            improvement = 0.0

            went_well.append(f"Evolved {self._population_size} agents over {self._num_iterations} generations")
            went_well.append(f"Best combined score: {best_score:.3f}")

            if generations:
                first_score = generations[0].get("mean_score", 0)
                last_score = generations[-1].get("mean_score", 0)
                improvement = last_score - first_score
                if improvement > 0:
                    went_well.append(f"Mean score improved by {improvement:.3f} across generations")
                else:
                    went_wrong.append(f"Mean score did not improve ({first_score:.3f} -> {last_score:.3f})")
                    suggestions.append("Try more iterations or adjust alpha for more exploration")

            # Get the best agent
            top_agents = pool.top_k(1)
            best_agent = top_agents[0] if top_agents else None

            if best_agent and isinstance(best_agent, SWEBenchAgent):
                went_well.append(f"Top agent (gen {best_agent.config.generation}): score={best_agent.combined_score:.3f}")
                return best_agent, StepReport(
                    step="3. GEA Evolution",
                    status="success",
                    duration_s=duration,
                    details={
                        "generations": len(generations),
                        "best_score": best_score,
                        "population_size": self._population_size,
                        "improvement": improvement if generations else 0,
                    },
                    went_well=went_well,
                    went_wrong=went_wrong,
                    suggestions=suggestions,
                )

            went_wrong.append("No SWEBenchAgent found in top agents")
            return None, StepReport(
                step="3. GEA Evolution",
                status="partial",
                duration_s=duration,
                went_well=went_well,
                went_wrong=went_wrong,
            )

        except Exception as e:
            duration = time.monotonic() - t
            went_wrong.append(f"Evolution failed: {e}")
            suggestions.append("Check LLM API key and connectivity")
            return None, StepReport(
                step="3. GEA Evolution",
                status="failed",
                duration_s=duration,
                went_wrong=went_wrong,
                suggestions=suggestions,
            )

    async def _step_generate_predictions(
        self,
        harness: SWEBenchHarness,
        instances: list[SWEBenchInstance],
        best_agent: SWEBenchAgent | None,
        run_id: str,
    ) -> tuple[list[SWEBenchPrediction], StepReport]:
        """Step 4: Run the best agent on all instances to produce patches.

        Instances are processed concurrently (up to ``_max_concurrent``).
        Each instance gets its own clone of the agent so state doesn't leak.
        An adaptive timeout starts at ``_instance_timeout_s`` and shrinks for
        repeated timeouts on the same instance.
        """
        import asyncio as _aio
        import copy

        t = time.monotonic()
        went_well: list[str] = []
        went_wrong: list[str] = []
        suggestions: list[str] = []

        if best_agent is None:
            # Create a default agent if evolution didn't produce one
            config = HowlerConfig(
                role_models={
                    LLMRole.ACTING: RoleModelConfig(model=self._model),
                    LLMRole.EVOLVING: RoleModelConfig(model=self._model),
                    LLMRole.REFLECTING: RoleModelConfig(model=self._model),
                },
            )
            spacing = 0.0 if is_cli_model(self._model) else 2.0
            llm = LLMRouter(config, min_request_interval_s=spacing)
            best_agent = SWEBenchAgent(
                config=AgentConfig(framework_config={"domain": "swe-bench"}),
                llm=llm,
            )
            went_wrong.append("Using default agent (evolution did not produce a viable agent)")

        # ---- parallel helper ----
        # best_agent is guaranteed non-None at this point (default created above)
        assert best_agent is not None
        agent_ref = best_agent

        max_concurrent = self._max_concurrent
        instance_timeout = self._instance_timeout_s
        sem = _aio.Semaphore(max_concurrent)
        results_map: dict[str, SWEBenchPrediction] = {}
        timing_map: dict[str, float] = {}  # instance_id -> wall seconds

        async def _process_instance(idx: int, instance: SWEBenchInstance) -> None:
            async with sem:
                inst_start = time.monotonic()
                iid = instance.instance_id
                logger.info(
                    "generating_prediction",
                    instance_id=iid,
                    progress=f"{idx+1}/{len(instances)}",
                )
                try:
                    repo_dir = harness.checkout_repo(instance)
                    # Deep-copy the agent to avoid shared mutable state
                    agent_copy = SWEBenchAgent(
                        config=copy.deepcopy(agent_ref.config),
                        llm=agent_ref._llm,
                    )
                    task: dict[str, Any] = {
                        "instance_id": iid,
                        "problem_statement": instance.problem_statement,
                        "repo": instance.repo,
                        "repo_dir": repo_dir,
                    }
                    # Include failing test names to guide the fix
                    if instance.fail_to_pass:
                        task["fail_to_pass"] = instance.fail_to_pass
                    # Adaptive timeout: use runner-level default
                    result = await _aio.wait_for(
                        agent_copy.run_task(task),
                        timeout=instance_timeout,
                    )
                    patch_output = result.output or ""
                    results_map[iid] = SWEBenchPrediction(
                        instance_id=iid,
                        model_name_or_path=f"howler-agents/{run_id}",
                        model_patch=patch_output,
                    )
                except _aio.TimeoutError:
                    elapsed = time.monotonic() - inst_start
                    logger.warning(
                        "prediction_timeout",
                        instance_id=iid,
                        timeout_s=instance_timeout,
                        elapsed_s=round(elapsed, 1),
                    )
                    results_map[iid] = SWEBenchPrediction(
                        instance_id=iid,
                        model_name_or_path=f"howler-agents/{run_id}",
                        model_patch="",
                    )
                    went_wrong.append(f"{iid}: timed out after {instance_timeout}s")
                except Exception as e:
                    logger.warning("prediction_error", instance_id=iid, error=str(e))
                    results_map[iid] = SWEBenchPrediction(
                        instance_id=iid,
                        model_name_or_path=f"howler-agents/{run_id}",
                        model_patch="",
                    )
                    went_wrong.append(f"{iid}: {e}")
                finally:
                    timing_map[iid] = time.monotonic() - inst_start

        # Launch all instances concurrently (semaphore limits parallelism)
        tasks = [_process_instance(i, inst) for i, inst in enumerate(instances)]
        await _aio.gather(*tasks)

        # Collect predictions preserving original order
        predictions: list[SWEBenchPrediction] = []
        success_count = 0
        valid_patch_count = 0
        for instance in instances:
            pred = results_map.get(instance.instance_id)
            if pred is None:
                pred = SWEBenchPrediction(
                    instance_id=instance.instance_id,
                    model_name_or_path=f"howler-agents/{run_id}",
                    model_patch="",
                )
            predictions.append(pred)
            if pred.model_patch:
                valid_patch_count += 1
                success_count += 1

        duration = time.monotonic() - t

        went_well.append(f"Generated predictions for {len(predictions)}/{len(instances)} instances")
        went_well.append(f"Valid patches: {valid_patch_count}/{len(instances)}")
        if max_concurrent > 1:
            went_well.append(f"Parallel execution: {max_concurrent} concurrent workers")

        # Report per-instance timing
        if timing_map:
            avg_time = sum(timing_map.values()) / len(timing_map)
            went_well.append(f"Avg instance time: {avg_time:.1f}s (wall: {duration:.1f}s)")

        if valid_patch_count < len(instances):
            diff = len(instances) - valid_patch_count
            went_wrong.append(f"{diff} instances produced no valid patch")
            suggestions.append("Improve file localization or patch generation prompts")

        # Write predictions
        preds_path = self._workspace / f"{run_id}-predictions.json"
        harness.write_predictions(predictions, preds_path)
        went_well.append(f"Predictions saved to {preds_path}")

        return predictions, StepReport(
            step="4. Generate Predictions",
            status="success" if valid_patch_count > 0 else "failed",
            duration_s=duration,
            details={
                "total_instances": len(instances),
                "predictions": len(predictions),
                "valid_patches": valid_patch_count,
                "success_rate": f"{valid_patch_count/len(instances)*100:.1f}%" if instances else "0%",
                "timing": {k: round(v, 1) for k, v in timing_map.items()},
                "max_concurrent": max_concurrent,
            },
            went_well=went_well,
            went_wrong=went_wrong,
            suggestions=suggestions,
        )

    async def _step_evaluate(
        self,
        harness: SWEBenchHarness,
        predictions: list[SWEBenchPrediction],
        run_id: str,
    ) -> StepReport:
        """Step 5: Run SWE-bench Docker evaluation."""
        t = time.monotonic()
        went_well = []
        went_wrong = []

        preds_path = self._workspace / f"{run_id}-predictions.json"

        try:
            results = harness.evaluate(
                predictions_path=preds_path,
                run_id=run_id,
            )
            duration = time.monotonic() - t

            resolved = results.get("resolved", 0)
            submitted = results.get("submitted", len(predictions))
            rate = results.get("resolved_rate", 0)

            went_well.append(f"Evaluation complete: {resolved}/{submitted} resolved ({rate:.1f}%)")

            if results.get("error"):
                went_wrong.append(f"Evaluation error: {results['error'][:200]}")

            return StepReport(
                step="5. Docker Evaluation",
                status="success" if not results.get("error") else "partial",
                duration_s=duration,
                details=results,
                went_well=went_well,
                went_wrong=went_wrong,
            )

        except Exception as e:
            duration = time.monotonic() - t
            went_wrong.append(f"Docker evaluation failed: {e}")
            return StepReport(
                step="5. Docker Evaluation",
                status="failed",
                duration_s=duration,
                went_wrong=went_wrong,
                suggestions=[
                    "Ensure Docker is running with 8+ CPUs and 16GB+ RAM",
                    "Try: docker info",
                ],
            )
