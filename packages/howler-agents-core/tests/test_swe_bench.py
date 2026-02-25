"""Tests for SWE-bench harness, agent, and evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from howler_agents.agents.base import AgentConfig, FrameworkPatch
from howler_agents.benchmarks.swe_bench_agent import SWEBenchAgent
from howler_agents.benchmarks.swe_bench_harness import (
    SWEBenchHarness,
    SWEBenchInstance,
    SWEBenchPrediction,
)
from howler_agents.benchmarks.swe_bench_runner import (
    SWE_BENCH_PROBES,
    EvalReport,
    StepReport,
    SWEBenchEvalRunner,
)

# --------------------------------------------------------------------------- #
# SWEBenchInstance fixtures                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_instance() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="test__repo-123",
        repo="test/repo",
        base_commit="abc123",
        problem_statement="The function foo() returns None instead of 42 when called with bar=True.",
        patch="diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,3 +1,3 @@\n def foo(bar=False):\n-    return None\n+    return 42 if bar else None\n",
        test_patch="",
        fail_to_pass=["test_foo_bar_true"],
        pass_to_pass=["test_foo_default"],
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="test response")
    return llm


# --------------------------------------------------------------------------- #
# SWEBenchHarness tests                                                        #
# --------------------------------------------------------------------------- #


class TestSWEBenchHarness:
    def test_harness_init(self, tmp_path: Path) -> None:
        harness = SWEBenchHarness(workspace=tmp_path / "harness")
        assert harness.workspace.exists()

    def test_write_predictions(self, tmp_path: Path) -> None:
        harness = SWEBenchHarness(workspace=tmp_path)
        predictions = [
            SWEBenchPrediction(
                instance_id="test__repo-1",
                model_name_or_path="howler-agents/test",
                model_patch="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n",
            ),
            SWEBenchPrediction(
                instance_id="test__repo-2",
                model_name_or_path="howler-agents/test",
                model_patch="",
            ),
        ]

        out = harness.write_predictions(predictions, tmp_path / "preds.json")
        assert out.exists()

        with out.open() as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["instance_id"] == "test__repo-1"
        assert data[0]["model_patch"].startswith("diff --git")
        assert data[1]["model_patch"] == ""

    def test_validate_setup_checks_git(self, tmp_path: Path) -> None:
        harness = SWEBenchHarness(workspace=tmp_path)
        checks = harness.validate_setup()

        # git should always be available in CI/dev
        assert "git" in checks
        assert checks["git"]["ok"] is True

        # disk_space should be present
        assert "disk_space" in checks
        assert "free_gb" in checks["disk_space"]


# --------------------------------------------------------------------------- #
# SWEBenchAgent tests                                                          #
# --------------------------------------------------------------------------- #


class TestSWEBenchAgent:
    @pytest.mark.asyncio
    async def test_run_task_no_problem(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        result = await agent.run_task({"instance_id": "x"})
        assert result.success is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_run_task_generates_patch(
        self,
        mock_llm: MagicMock,
        sample_instance: SWEBenchInstance,
        tmp_path: Path,
    ) -> None:
        # Mock the localization response
        mock_llm.complete = AsyncMock(
            side_effect=[
                # Step 1: localize files
                "./src/foo.py\n./src/bar.py",
                # Step 2: generate patch
                "diff --git a/src/foo.py b/src/foo.py\n"
                "--- a/src/foo.py\n"
                "+++ b/src/foo.py\n"
                "@@ -1,3 +1,3 @@\n"
                " def foo(bar=False):\n"
                "-    return None\n"
                "+    return 42 if bar else None\n",
            ]
        )

        # Create fake repo directory with a file
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "foo.py").write_text("def foo(bar=False):\n    return None\n")
        (repo_dir / "src" / "bar.py").write_text("# bar module\n")

        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        result = await agent.run_task(
            {
                "instance_id": sample_instance.instance_id,
                "problem_statement": sample_instance.problem_statement,
                "repo": sample_instance.repo,
                "repo_dir": repo_dir,
            }
        )

        assert result.output != ""
        assert "diff --git" in result.output
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_apply_patch(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        fw_patch = FrameworkPatch(
            intent="Improve localization strategy",
            config_updates={"localization_strategy": "Use AST analysis"},
        )
        await agent.apply_patch(fw_patch)

        assert len(agent.patches) == 1
        assert agent.config.framework_config["localization_strategy"] == "Use AST analysis"

    @pytest.mark.asyncio
    async def test_patch_retry_on_empty(self, mock_llm: MagicMock) -> None:
        """Validation retry should re-prompt when the first response has no diff."""
        mock_llm.complete = AsyncMock(
            side_effect=[
                # Localization
                "./src/foo.py",
                # Attempt 1: garbage response (no diff)
                "I think the fix is to change line 5.",
                # Attempt 2: valid diff
                "diff --git a/src/foo.py b/src/foo.py\n"
                "--- a/src/foo.py\n"
                "+++ b/src/foo.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def foo(bar=False):\n"
                "-    return None\n"
                "+    return 42 if bar else None\n",
            ]
        )

        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        result = await agent.run_task(
            {
                "instance_id": "test__repo-retry",
                "problem_statement": "foo returns None instead of 42",
                "repo": "test/repo",
                "repo_dir": None,
            }
        )

        assert result.output != ""
        assert "diff --git" in result.output
        # Should have called complete 3 times: 1 localize + 2 patch attempts
        assert mock_llm.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_extract_diff_from_fences(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)

        response = (
            "Here is the fix:\n"
            "```diff\n"
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```\n"
        )
        diff = agent._extract_diff(response)
        assert "diff --git" in diff
        assert "+new" in diff

    @pytest.mark.asyncio
    async def test_extract_diff_empty(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        diff = agent._extract_diff("I'm not sure how to fix this.")
        assert diff == ""

    def test_extract_diff_json_wrapped(self, mock_llm: MagicMock) -> None:
        """Test extraction from JSON-wrapped output (Claude Code --output-format json)."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        response = json.dumps(
            {
                "result": (
                    "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
                )
            }
        )
        diff = agent._extract_diff(response)
        assert "diff --git" in diff
        assert "+new" in diff

    def test_extract_diff_partial_without_git_header(self, mock_llm: MagicMock) -> None:
        """Test extraction of diff without diff --git header (just --- a/ +++ b/)."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        response = (
            "Here's the minimal fix:\n"
            "--- a/astropy/io/ascii/qdp.py\n"
            "+++ b/astropy/io/ascii/qdp.py\n"
            "@@ -55,7 +55,7 @@\n"
            "     _line_type_re = re.compile(_type_re)\n"
            "-    _line_type_re = re.compile(_type_re)\n"
            "+    _line_type_re = re.compile(_type_re, re.IGNORECASE)\n"
        )
        diff = agent._extract_diff(response)
        assert "diff --git" in diff
        assert "re.IGNORECASE" in diff

    def test_fix_hunk_headers_truncated_patch(self, mock_llm: MagicMock) -> None:
        """Test that _fix_hunk_headers corrects wrong line counts.

        LLMs often emit @@ headers with wrong counts (e.g. claiming 7 lines
        but only producing 4). Docker's patch(1) rejects these while git-apply
        is lenient. The fix recalculates counts from actual content.
        """
        SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        # This is the exact truncated patch that failed in cc-012 Docker eval
        bad_patch = (
            "diff --git a/astropy/io/fits/fitsrec.py b/astropy/io/fits/fitsrec.py\n"
            "--- a/astropy/io/fits/fitsrec.py\n"
            "+++ b/astropy/io/fits/fitsrec.py\n"
            "@@ -1243,7 +1243,7 @@ class FITS_rec(np.recarray):\n"
            "         # Replace exponent separator in floating point numbers\n"
            "         if 'D' in format:\n"
            "-            output_field.replace(encode_ascii('E'), encode_ascii('D'))\n"
            "+            output_field[:] = output_field.replace(encode_ascii('E'), encode_ascii('D'))\n"
        )
        fixed = SWEBenchAgent._fix_hunk_headers(bad_patch)
        # Should have corrected counts: 3 old lines (2 context + 1 removed),
        # 3 new lines (2 context + 1 added)
        assert "@@ -1243,3 +1243,3 @@" in fixed
        # Content should be preserved
        assert "output_field[:] =" in fixed
        assert "diff --git" in fixed

    def test_fix_hunk_headers_correct_patch(self, mock_llm: MagicMock) -> None:
        """Patches always get recalculated counts from actual content."""
        good_patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,5 +10,5 @@\n"
            " before\n"
            " context\n"
            "-old\n"
            "+new\n"
            " after\n"
        )
        fixed = SWEBenchAgent._fix_hunk_headers(good_patch)
        # 3 context + 1 removed = 4 old, 3 context + 1 added = 4 new
        assert "@@ -10,4 +10,4 @@" in fixed

    def test_fix_corrupt_lines_missing_space(self, mock_llm: MagicMock) -> None:
        """Context lines without leading space get one prepended."""
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,3 +10,3 @@\n"
            "context_without_space\n"  # missing leading space
            "-old\n"
            "+new\n"
        )
        fixed = SWEBenchAgent._fix_corrupt_lines(patch)
        assert " context_without_space\n" in fixed
        assert "-old\n" in fixed
        assert "+new\n" in fixed

    def test_fix_corrupt_lines_empty_line_in_hunk(self, mock_llm: MagicMock) -> None:
        """Empty lines in hunk body become single-space context lines."""
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,4 +10,4 @@\n"
            " before\n"
            "\n"  # empty line — should become " "
            "-old\n"
            "+new\n"
        )
        fixed = SWEBenchAgent._fix_corrupt_lines(patch)
        lines = fixed.splitlines()
        # The empty line should now be a single space
        hunk_start = next(i for i, line in enumerate(lines) if line.startswith("@@"))
        assert lines[hunk_start + 2] == " "

    def test_fix_corrupt_lines_preserves_valid(self, mock_llm: MagicMock) -> None:
        """Valid diff lines are not modified."""
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,3 +10,3 @@\n"
            " context\n"
            "-old\n"
            "+new\n"
        )
        fixed = SWEBenchAgent._fix_corrupt_lines(patch)
        # Should be unchanged
        assert fixed == patch + "\n" or fixed.strip() == patch.strip()

    def test_read_test_code(self, mock_llm: MagicMock, tmp_path: Path) -> None:
        """Test reading test function source from a file."""
        test_file = tmp_path / "tests" / "test_foo.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "import pytest\n\n"
            "class TestFoo:\n"
            "    def test_bar(self):\n"
            "        assert 1 + 1 == 2\n\n"
            "    def test_baz(self):\n"
            "        assert 2 + 2 == 4\n"
        )
        result = SWEBenchAgent._read_test_code(tmp_path, ["tests/test_foo.py::TestFoo::test_bar"])
        assert "test_bar" in result
        assert "assert 1 + 1 == 2" in result
        # Should NOT include test_baz
        assert "test_baz" not in result

    def test_read_test_code_no_repo(self, mock_llm: MagicMock) -> None:
        """Returns empty string when no repo_dir."""
        result = SWEBenchAgent._read_test_code(None, ["test::foo"])
        assert result == ""

    def test_keyword_filter_files_small_list(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        files = ["./src/foo.py", "./src/bar.py"]
        result = agent._keyword_filter_files(files, "something about foo", max_files=80)
        # Small list returned as-is
        assert result == files

    def test_keyword_filter_files_large_list(self, mock_llm: MagicMock) -> None:
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        # Create 200 files, only a few matching "modeling"
        files = [f"./pkg/module{i}.py" for i in range(200)]
        files.append("./pkg/modeling/core.py")
        files.append("./pkg/modeling/utils.py")
        files.sort()

        result = agent._keyword_filter_files(
            files,
            "The CompoundModel in modeling returns wrong results",
            max_files=20,
        )
        # Matching files should be included
        assert "./pkg/modeling/core.py" in result
        assert "./pkg/modeling/utils.py" in result
        # Limited to max_files
        assert len(result) <= 20

    def test_focus_extract_matches_keyword(self, mock_llm: MagicMock) -> None:
        """Focused extraction keeps matching class/function blocks."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        # Simulate a large Python file with multiple classes
        lines = [
            "import os",
            "",
            "class Unrelated:",
            "    def method(self):",
            "        pass",
            "",
            "class RstWriter:",
            "    def write_header(self):",
            "        return '==='",
            "    def write_row(self, row):",
            "        return row",
            "",
            "class AnotherUnrelated:",
            "    pass",
        ]
        text = "\n".join(lines)
        keywords = {"RstWriter", "write_header"}
        result = agent._focus_extract(text, keywords, "./test.py")
        assert "RstWriter" in result
        assert "write_header" in result

    def test_focus_extract_per_file_cap(self, mock_llm: MagicMock) -> None:
        """Per-file cap limits extraction even with many matching blocks."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        # Create a very large file with many matching class blocks
        lines = ["import os", ""]
        for i in range(20):
            lines.append(f"class Model{i}:")
            # Each class has many lines to exceed per-file cap
            lines.extend([f"    attr_{j} = 'value_{j}_{'x' * 200}'" for j in range(50)])
            lines.append("")
        text = "\n".join(lines)
        assert len(text) > 50_000  # Much larger than per-file cap
        keywords = {"Model"}  # Matches all 20 classes
        result = agent._focus_extract(text, keywords, "./huge.py")
        # Result should be capped at _MAX_PER_FILE (10K)
        from howler_agents.benchmarks.swe_bench_agent import _MAX_PER_FILE

        assert len(result) <= _MAX_PER_FILE + 500  # small tolerance for markers
        # Should have cap marker
        assert "per-file cap reached" in result

    def test_focus_extract_fallback_head_tail(self, mock_llm: MagicMock) -> None:
        """When no keywords match, include head and tail of file."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        lines = [f"line_{i} = {i}" for i in range(200)]
        text = "\n".join(lines)
        result = agent._focus_extract(text, {"NoMatch"}, "./big.py")
        # Should include first 80 lines
        assert "line_0" in result
        assert "line_79" in result
        # Should include tail
        assert "line_199" in result
        # Should have omission marker
        assert "omitted" in result

    def test_read_files_focused_extraction(self, mock_llm: MagicMock, tmp_path: Path) -> None:
        """_read_files uses focused extraction for large files."""
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)
        # Set the problem text for keyword extraction
        agent.config.framework_config["_problem_text"] = "RstWriter header_rows"

        # Create a large file (>5K chars) with relevant and irrelevant content
        big_lines = ["import ast", ""]
        big_lines.append("class Unrelated:")
        big_lines.extend(
            [
                f"    attr_{i} = 'value_{i}_padding_extra_text_to_exceed_threshold'"
                for i in range(300)
            ]
        )
        big_lines.append("")
        big_lines.append("class RstWriter:")
        big_lines.append("    def write(self, data):")
        big_lines.append("        return data")
        big_content = "\n".join(big_lines)
        assert len(big_content) > 5000  # Must exceed _FOCUS_THRESHOLD
        big_file = tmp_path / "big.py"
        big_file.write_text(big_content)

        # Create a small file
        small_file = tmp_path / "small.py"
        small_file.write_text("def helper():\n    return 1\n")

        contents = agent._read_files(tmp_path, ["./big.py", "./small.py"])
        assert "./big.py" in contents
        assert "./small.py" in contents
        # Small file should be included in full
        assert "helper" in contents["./small.py"]
        # Big file should have focused extraction (much smaller than original)
        assert "RstWriter" in contents["./big.py"]
        assert len(contents["./big.py"]) < len(big_content)


# --------------------------------------------------------------------------- #
# SWE-bench probes                                                             #
# --------------------------------------------------------------------------- #


def test_swe_bench_probes_count() -> None:
    assert len(SWE_BENCH_PROBES) >= 15


def test_swe_bench_probes_have_required_fields() -> None:
    for probe in SWE_BENCH_PROBES:
        assert "description" in probe
        assert "type" in probe
        assert len(probe["description"]) > 10


# --------------------------------------------------------------------------- #
# EvalReport                                                                   #
# --------------------------------------------------------------------------- #


class TestEvalReport:
    def test_to_dict(self) -> None:
        report = EvalReport(
            run_id="test-001",
            started_at="2025-01-01T00:00:00",
            steps=[
                StepReport(
                    step="1. Test",
                    status="success",
                    duration_s=1.5,
                    went_well=["Everything worked"],
                ),
            ],
            final_results={"resolved": 3, "submitted": 5, "resolved_rate": 60.0},
        )
        d = report.to_dict()
        assert d["run_id"] == "test-001"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["status"] == "success"
        assert d["final_results"]["resolved_rate"] == 60.0

    def test_summary(self) -> None:
        report = EvalReport(
            run_id="test-002",
            steps=[
                StepReport(
                    step="1. Setup",
                    status="success",
                    duration_s=0.5,
                    went_well=["OK"],
                ),
                StepReport(
                    step="2. Load",
                    status="failed",
                    duration_s=1.0,
                    went_wrong=["Missing dataset"],
                    suggestions=["pip install datasets"],
                ),
            ],
            final_results={"resolved": 0, "submitted": 0, "resolved_rate": 0.0},
        )
        summary = report.summary()
        assert "test-002" in summary
        assert "[+] 1. Setup" in summary
        assert "[-] 2. Load" in summary
        assert "Missing dataset" in summary
        assert "pip install datasets" in summary


# --------------------------------------------------------------------------- #
# SWEBenchEvalRunner constructor                                               #
# --------------------------------------------------------------------------- #


class TestEvalRunnerInit:
    def test_default_params(self, tmp_path: Path) -> None:
        runner = SWEBenchEvalRunner(model="claude-sonnet-4-20250514", workspace=tmp_path)
        assert runner._max_concurrent == 3
        assert runner._instance_timeout_s == 300.0

    def test_custom_concurrency(self, tmp_path: Path) -> None:
        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            workspace=tmp_path,
            max_concurrent=5,
            instance_timeout_s=120.0,
        )
        assert runner._max_concurrent == 5
        assert runner._instance_timeout_s == 120.0

    def test_min_concurrent_is_one(self, tmp_path: Path) -> None:
        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            workspace=tmp_path,
            max_concurrent=0,
        )
        assert runner._max_concurrent == 1


class TestParallelPredictions:
    """Test that _step_generate_predictions runs instances concurrently."""

    @pytest.fixture
    def instances(self) -> list[SWEBenchInstance]:
        return [
            SWEBenchInstance(
                instance_id=f"test__repo-{i}",
                repo="test/repo",
                base_commit="abc123",
                problem_statement=f"Bug {i}",
                patch="",
                test_patch="",
                fail_to_pass=[],
                pass_to_pass=[],
            )
            for i in range(4)
        ]

    @pytest.mark.asyncio
    async def test_predictions_preserve_order(
        self, instances: list[SWEBenchInstance], tmp_path: Path
    ) -> None:
        """Results should be in the same order as input instances."""
        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            workspace=tmp_path,
            max_concurrent=2,
            instance_timeout_s=5.0,
        )

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        )
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)

        harness = MagicMock()
        harness.checkout_repo = MagicMock(return_value=tmp_path)
        harness.write_predictions = MagicMock()

        predictions, step = await runner._step_generate_predictions(
            harness, instances, agent, "test-run"
        )

        assert len(predictions) == 4
        for i, pred in enumerate(predictions):
            assert pred.instance_id == f"test__repo-{i}"
        assert step.details["max_concurrent"] == 2

    @pytest.mark.asyncio
    async def test_timeout_produces_empty_patch(
        self, instances: list[SWEBenchInstance], tmp_path: Path
    ) -> None:
        """Instances that exceed the timeout should produce empty patches."""
        import asyncio

        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            workspace=tmp_path,
            max_concurrent=4,
            instance_timeout_s=0.1,  # Very short timeout
        )

        mock_llm = MagicMock()

        async def slow_complete(*_a: object, **_kw: object) -> str:
            await asyncio.sleep(10)  # Will exceed timeout
            return "diff --git a/f.py b/f.py\n"

        mock_llm.complete = slow_complete
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)

        harness = MagicMock()
        harness.checkout_repo = MagicMock(return_value=tmp_path)
        harness.write_predictions = MagicMock()

        predictions, step = await runner._step_generate_predictions(
            harness, instances[:1], agent, "test-run"
        )

        assert len(predictions) == 1
        assert predictions[0].model_patch == ""
        assert any("timed out" in w for w in step.went_wrong)

    @pytest.mark.asyncio
    async def test_concurrent_timing(
        self, instances: list[SWEBenchInstance], tmp_path: Path
    ) -> None:
        """With max_concurrent=4, 4 instances should finish faster than sequential."""
        import asyncio
        import time

        runner = SWEBenchEvalRunner(
            model="claude-sonnet-4-20250514",
            workspace=tmp_path,
            max_concurrent=4,
            instance_timeout_s=5.0,
        )

        mock_llm = MagicMock()

        async def delayed_complete(*_a: object, **_kw: object) -> str:
            await asyncio.sleep(0.1)
            return "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"

        mock_llm.complete = delayed_complete
        agent = SWEBenchAgent(config=AgentConfig(), llm=mock_llm)

        harness = MagicMock()
        harness.checkout_repo = MagicMock(return_value=tmp_path)
        harness.write_predictions = MagicMock()

        t0 = time.monotonic()
        predictions, _step = await runner._step_generate_predictions(
            harness, instances, agent, "test-run"
        )
        elapsed = time.monotonic() - t0

        assert len(predictions) == 4
        # 4 instances each sleeping 0.1s — sequential would take ~0.4s+,
        # parallel should take ~0.1s. Allow generous margin for CI.
        assert elapsed < 1.0, f"Parallel execution took too long: {elapsed:.2f}s"
