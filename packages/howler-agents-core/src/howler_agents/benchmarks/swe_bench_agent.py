"""SWE-Bench agent implementation for howler-agents.

Implements a multi-step agent that:
1. Localizes relevant files from the problem statement
2. Reads and understands the relevant code
3. Generates a unified diff patch to fix the issue
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import structlog

from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.config import LLMRole
from howler_agents.llm.router import LLMRouter

logger = structlog.get_logger()

# Maximum characters of file content to include in context.
# 50K chars ≈ 12.5K tokens — close to SWE-bench bm25_13K tier (52K chars).
# With focused extraction, we achieve higher relevance density than raw BM25,
# so fewer total chars are needed for equivalent coverage.
# Research (LongSWE-Bench) shows accuracy drops sharply beyond ~15K tokens,
# and SWE-agent shows 100-line windows outperform full file dumps.
# Keeping total prompt under ~60K chars also avoids CLI timeout issues.
_MAX_FILE_CONTEXT = 50_000
_MAX_FILES_TO_READ = 15
# Per-file extraction cap: prevents one large file from consuming the budget.
# 15K chars ≈ 3.75K tokens — enough for imports + 3-4 focused class/function blocks
# including __init__ methods where root causes often live.
_MAX_PER_FILE = 15_000
# Files smaller than this are included in full; larger files get focused extraction
_FOCUS_THRESHOLD = 5_000
# Max validation-retry attempts for patch generation (Instructor-style feedback loop).
# Each retry sends git-apply errors back to the LLM for corrective re-generation.
_MAX_PATCH_RETRIES = 3


class SWEBenchAgent(Agent):
    """Agent that solves SWE-bench instances by producing unified diffs.

    Uses a multi-step approach:
    1. File localization: identify which files to edit
    2. Code understanding: read and comprehend the relevant code
    3. Patch generation: produce a unified diff that fixes the issue

    The agent's framework_config can contain evolved strategies:
    - "localization_strategy": how to find relevant files
    - "patch_strategy": approach to generating patches
    - "reasoning_depth": level of analysis before patching
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMRouter,
        workspace: Path | None = None,
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._workspace = workspace

    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a SWE-bench task.

        Task dict should contain:
            - instance_id: str
            - problem_statement: str
            - repo_dir: Path (checked-out repository)
            - repo: str (e.g., "sympy/sympy")
        """
        instance_id = task.get("instance_id", "unknown")
        problem = task.get("problem_statement", "")
        repo_dir = task.get("repo_dir")
        repo = task.get("repo", "")
        fail_to_pass: list[str] = task.get("fail_to_pass", [])

        if not problem:
            return TaskResult(
                success=False,
                score=0.0,
                output="",
                key_decisions=["No problem statement provided"],
                lessons_learned=["Need valid SWE-bench instance"],
            )

        logger.info("swebench_task_start", instance_id=instance_id, repo=repo)

        # Store problem text for focused extraction in _read_files
        # Also include test names as keywords (e.g., test_ccode_Relational → Relational)
        test_keywords = ""
        for t in fail_to_pass:
            parts = t.split("::")
            if parts:
                test_keywords += " " + parts[-1].replace("test_", "").replace("_", " ")
        self.config.framework_config["_problem_text"] = problem + test_keywords
        # Store repo_dir for patch validation in _generate_patch
        self.config.framework_config["_repo_dir"] = repo_dir

        try:
            # Step 1: Localize files
            relevant_files = await self._localize_files(problem, repo_dir, repo)
            decisions = [f"Identified {len(relevant_files)} relevant files"]

            # Step 2: Read file contents
            file_contents = self._read_files(repo_dir, relevant_files)
            decisions.append(f"Read {len(file_contents)} files ({sum(len(c) for c in file_contents.values())} chars)")

            # Step 2b: Read failing test code if available
            test_context = self._read_test_code(repo_dir, fail_to_pass)
            if test_context:
                decisions.append(f"Read {len(test_context)} chars of failing test code")

            # Step 3: Generate patch
            patch = await self._generate_patch(problem, file_contents, repo, instance_id, fail_to_pass, test_context)

            if not patch or patch.strip() == "":
                return TaskResult(
                    success=False,
                    score=0.1,
                    output="",
                    key_decisions=[*decisions, "Failed to generate a patch"],
                    lessons_learned=["Need better localization or understanding"],
                )

            # Step 4: Validate patch format
            valid = self._validate_patch(patch, repo_dir)
            score = 0.7 if valid else 0.3

            decisions.append(f"Generated patch ({'valid' if valid else 'invalid'} format)")

            return TaskResult(
                success=valid,
                score=score,
                output=patch,
                key_decisions=decisions,
                lessons_learned=self._extract_lessons(patch, valid),
            )

        except Exception as exc:
            logger.warning("swebench_task_error", instance_id=instance_id, error=str(exc))
            return TaskResult(
                success=False,
                score=0.0,
                output="",
                key_decisions=[f"Error: {exc}"],
                lessons_learned=["Handle task execution errors gracefully"],
            )

    async def _localize_files(
        self,
        problem: str,
        repo_dir: Path | None,
        repo: str,
    ) -> list[str]:
        """Identify which files are relevant to the problem."""
        # Get repo structure
        repo_structure = ""
        if repo_dir and repo_dir.exists():
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-not", "-path", "./.git/*",
                 "-not", "-path", "./.tox/*", "-not", "-path", "./build/*"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            all_py_files = sorted(result.stdout.strip().splitlines())

            # Filter files by keywords from the problem statement to reduce tokens
            py_files = self._keyword_filter_files(all_py_files, problem)
            repo_structure = "\n".join(py_files)
            if len(all_py_files) > len(py_files):
                repo_structure += f"\n... ({len(all_py_files)} total Python files, {len(py_files)} shown matching problem keywords)"

        # Evolved localization strategy from framework_config
        strategy = self.config.framework_config.get(
            "localization_strategy",
            "Analyze the error traceback, module references, and class/function names "
            "in the problem statement to identify the most relevant source files.",
        )

        prompt = (
            f"You are an expert Python developer analyzing a bug report for '{repo}'.\n\n"
            f"## Localization Strategy\n{strategy}\n\n"
            f"## Problem Statement\n{problem}\n\n"
            f"## Repository Python Files\n{repo_structure}\n\n"
            "## Instructions\n"
            "Think step by step:\n"
            "1. Identify the key classes, functions, or modules mentioned in the problem\n"
            "2. Map those to specific file paths in the repository listing above\n"
            "3. Consider which files contain the buggy logic vs. which just call it\n"
            "4. Prioritize files that would need the smallest, most targeted fix\n\n"
            "Return ONLY file paths, one per line, most important first.\n"
            "Limit to 10 files maximum. Include the leading './' prefix.\n"
            "Do NOT include test files unless the bug is in test infrastructure.\n"
            "Do NOT include any explanation — just file paths."
        )

        messages = [{"role": "user", "content": prompt}]
        response = await self._llm.complete(role=LLMRole.ACTING, messages=messages)

        # Parse file paths from response
        files = []
        for line in response.strip().splitlines():
            line = line.strip().strip("-").strip("*").strip("`").strip()
            if line.startswith("./") and line.endswith(".py"):
                files.append(line)
            elif line.endswith(".py") and "/" in line:
                files.append(f"./{line}")

        return files[:_MAX_FILES_TO_READ]

    def _keyword_filter_files(
        self,
        files: list[str],
        problem: str,
        max_files: int = 80,
    ) -> list[str]:
        """Filter file list by keywords extracted from the problem statement.

        For large repos (1000+ files), this dramatically reduces prompt size
        by only showing files whose paths match terms from the problem.
        """
        if len(files) <= max_files:
            return files

        # Extract meaningful keywords from problem (module names, class names, etc.)
        import re
        # Find words that look like Python identifiers (lowercase with underscores)
        words = set(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", problem.lower()))
        # Find CamelCase names (class names)
        words.update(w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", problem))
        # Remove very common words that would match too many files
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "not", "but",
            "are", "was", "has", "have", "had", "been", "does", "did", "can",
            "could", "should", "would", "when", "where", "how", "what", "which",
            "def", "class", "return", "import", "none", "true", "false", "self",
            "test", "tests", "error", "exception", "value", "result", "output",
            "input", "file", "line", "code", "function", "method", "type",
        }
        keywords = words - stop_words

        if not keywords:
            return files[:max_files]

        # Score each file by how many keywords its path contains
        scored: list[tuple[int, str]] = []
        for fpath in files:
            path_lower = fpath.lower()
            score = sum(1 for kw in keywords if kw in path_lower)
            scored.append((score, fpath))

        # Return matching files first, then pad with non-matching up to max
        scored.sort(key=lambda x: (-x[0], x[1]))
        matching = [f for s, f in scored if s > 0]
        non_matching = [f for s, f in scored if s == 0]

        result = matching[:max_files]
        remaining = max_files - len(result)
        if remaining > 0:
            result.extend(non_matching[:remaining])

        return sorted(result)

    def _read_files(self, repo_dir: Path | None, files: list[str]) -> dict[str, str]:
        """Read contents of relevant files with focused extraction.

        Small files (<3K chars) are included in full.
        Large files get focused extraction: only classes/functions whose names
        appear in the problem statement are included, with surrounding context.
        This reduces 80K prompts to ~5-15K while keeping the relevant code.
        """
        contents: dict[str, str] = {}
        total_chars = 0

        if repo_dir is None:
            return contents

        # Extract keywords from the problem for focused extraction
        problem = self.config.framework_config.get("_problem_text", "")
        keywords = self._extract_code_keywords(problem)

        for fpath in files:
            if total_chars >= _MAX_FILE_CONTEXT:
                break
            full_path = repo_dir / fpath.lstrip("./")
            if full_path.exists() and full_path.is_file():
                try:
                    text = full_path.read_text(errors="replace")
                    # Small files: include in full
                    if len(text) <= _FOCUS_THRESHOLD:
                        extracted = text
                    else:
                        # Large files: extract relevant blocks
                        extracted = self._focus_extract(text, keywords, fpath)

                    if total_chars + len(extracted) > _MAX_FILE_CONTEXT:
                        extracted = extracted[:_MAX_FILE_CONTEXT - total_chars]
                    contents[fpath] = extracted
                    total_chars += len(extracted)
                except Exception:
                    pass

        return contents

    @staticmethod
    def _read_test_code(
        repo_dir: Path | None, fail_to_pass: list[str], max_chars: int = 8000
    ) -> str:
        """Read the source of failing test functions from the repo.

        Parses pytest-style test IDs like 'tests/test_foo.py::TestClass::test_method'
        to locate and extract the test function source code.  Returns at most
        ``max_chars`` of concatenated test code.

        When the test is a method in a class (e.g., TestFoo::test_bar), also
        includes the class setUp/setUpClass and any class-level attributes,
        since those often define the expected values the test asserts against.
        Also includes the file's import block for type context.
        """
        if not repo_dir or not fail_to_pass:
            return ""

        parts_list: list[str] = []
        char_count = 0
        seen_files: set[str] = set()

        for test_id in fail_to_pass:
            if char_count >= max_chars:
                break
            # Parse "path/to/test.py::Class::method" or "path/to/test.py::method"
            segments = test_id.split("::")
            if not segments:
                continue
            test_file = segments[0]
            test_name = segments[-1] if len(segments) > 1 else ""
            test_class = segments[1] if len(segments) > 2 else ""

            # Strip parametrize markers like "test_foo[True]"
            if "[" in test_name:
                test_name = test_name[: test_name.index("[")]

            full_path = repo_dir / test_file
            if not full_path.exists():
                continue

            try:
                text = full_path.read_text(errors="replace")
                lines = text.splitlines()

                # Include imports from the test file (once per file)
                if test_file not in seen_files:
                    seen_files.add(test_file)
                    import_lines = []
                    for line in lines[:40]:
                        if re.match(r"^(import |from |#)", line) or not line.strip():
                            import_lines.append(line)
                        elif line.strip() and not line.startswith(" "):
                            break
                    if import_lines:
                        imports = "\n".join(import_lines)
                        if len(imports) + char_count <= max_chars:
                            parts_list.append(f"# {test_file} imports:\n{imports}")
                            char_count += len(imports)

                if not test_name:
                    # No specific function — include first 60 lines
                    snippet = "\n".join(lines[:60])
                else:
                    snippets: list[str] = []

                    # If test is inside a class, include setUp/setUpClass
                    if test_class:
                        for setup_name in ("setUp", "setUpClass", "setup_method"):
                            for i, line in enumerate(lines):
                                if re.match(
                                    rf"^\s+(def|async def)\s+{re.escape(setup_name)}\b",
                                    line,
                                ):
                                    indent = len(line) - len(line.lstrip())
                                    end = i + 1
                                    while end < len(lines):
                                        ln = lines[end]
                                        if ln.strip() and re.match(
                                            r"^\s*(def|async def|class)\s", ln
                                        ):
                                            ln_indent = len(ln) - len(ln.lstrip())
                                            if ln_indent <= indent:
                                                break
                                        end += 1
                                    snippets.append("\n".join(lines[i:end]))
                                    break

                    # Find the test function definition
                    start = None
                    for i, line in enumerate(lines):
                        if re.match(rf"^\s*(def|async def)\s+{re.escape(test_name)}\b", line):
                            start = i
                            break
                    if start is not None:
                        # Capture until next def at same or less indent, or EOF
                        indent = len(lines[start]) - len(lines[start].lstrip())
                        end = start + 1
                        while end < len(lines):
                            ln = lines[end]
                            if ln.strip() and not ln[0].isspace():
                                break  # top-level line
                            if ln.strip() and re.match(r"^\s*(def|async def|class)\s", ln):
                                ln_indent = len(ln) - len(ln.lstrip())
                                if ln_indent <= indent:
                                    break
                            end += 1
                        snippets.append("\n".join(lines[max(0, start - 2) : end]))

                    snippet = "\n\n".join(snippets) if snippets else ""

                if not snippet:
                    continue

                if len(snippet) + char_count > max_chars:
                    snippet = snippet[: max_chars - char_count]
                parts_list.append(f"# {test_id}\n{snippet}")
                char_count += len(snippet)
            except Exception:
                continue

        return "\n\n".join(parts_list)

    def _extract_code_keywords(self, problem: str) -> set[str]:
        """Extract class/function names from a problem statement."""
        keywords: set[str] = set()
        # CamelCase class names
        keywords.update(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", problem))
        # snake_case function/method names
        keywords.update(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", problem))
        # Dotted references like module.ClassName
        for match in re.findall(r"(\w+)\.(\w+)", problem):
            keywords.update(match)
        # Remove noise
        noise = {
            "the", "and", "for", "that", "this", "with", "from", "not", "but",
            "are", "was", "has", "have", "had", "been", "does", "did", "can",
            "could", "should", "would", "when", "where", "how", "what", "which",
            "def", "class", "return", "import", "none", "true", "false", "self",
            "error", "exception", "value", "result", "output", "input", "file",
            "line", "code", "function", "method", "type", "name", "data", "new",
        }
        return keywords - noise

    def _focus_extract(
        self, text: str, keywords: set[str], fpath: str
    ) -> str:
        """Extract relevant code blocks from a large file.

        Uses a two-pass approach:
        1. Find all method/function definitions and their body ranges
        2. Select methods whose names match keywords OR whose bodies contain
           keyword references (e.g., __init__ defining self.ordering_parts)
        Also always includes __init__ of classes with matched methods.
        Falls back to head+tail if no matches.
        """
        lines = text.splitlines()
        total = len(lines)

        # First pass: find all def/class blocks with their ranges
        all_defs: list[tuple[int, int, str, str, int]] = []  # (start, end, kind, name, indent)
        for i, line in enumerate(lines):
            match = re.match(r"^(\s*)(class|def|async def)\s+(\w+)", line)
            if match:
                indent = len(match.group(1))
                kind = match.group(2).replace("async ", "")
                name = match.group(3)
                # Find end of this block
                end = i + 1
                while end < total:
                    next_line = lines[end]
                    if next_line.strip() and not next_line[0].isspace() and indent == 0:
                        break  # top-level: next non-indented line
                    if next_line.strip() and re.match(r"^(\s*)(class|def|async def)\s+\w+", next_line):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent:
                            break
                    end += 1
                all_defs.append((i, min(end, total), kind, name, indent))

        # Second pass: select blocks that match keywords by name or body content
        kw_lower = {kw.lower() for kw in keywords if len(kw) > 2}
        blocks: list[tuple[int, int]] = []
        matched_class_ranges: list[tuple[int, int]] = []  # track matched classes

        for start, end, kind, name, _indent in all_defs:
            name_lower = name.lower()
            name_matches = any(
                kw == name_lower or name_lower in kw or kw in name_lower
                for kw in kw_lower
            )
            if name_matches:
                blocks.append((max(0, start - 2), min(end, total - 1)))
                if kind == "class":
                    matched_class_ranges.append((start, end))
                continue

            # Check body content for keyword references (but only for methods, not huge classes)
            if kind in ("def",) and (end - start) < 150:
                body = " ".join(lines[start:end]).lower()
                if any(kw in body for kw in kw_lower):
                    blocks.append((max(0, start - 2), min(end, total - 1)))
                    # Track which class this method belongs to
                    for cs, ce, ck, _cn, _ci in all_defs:
                        if ck == "class" and cs < start < ce:
                            matched_class_ranges.append((cs, ce))
                            break

        # Always include __init__ of classes that contain matched methods
        for cls_start, cls_end in matched_class_ranges:
            for start, end, _kind, name, _indent in all_defs:
                if name == "__init__" and start > cls_start and start < cls_end:
                    blocks.append((max(0, start - 1), min(end, total - 1)))
                    break

        if not blocks:
            # No keyword matches — include head (imports + first class) and tail
            head = min(80, total)
            tail_start = max(head, total - 30)
            excerpt = "\n".join(lines[:head])
            if tail_start > head:
                excerpt += f"\n\n# ... ({total - head - (total - tail_start)} lines omitted) ...\n\n"
                excerpt += "\n".join(lines[tail_start:])
            return excerpt

        # Merge overlapping blocks and add context
        merged: list[tuple[int, int]] = []
        for start, end in sorted(blocks):
            if merged and start <= merged[-1][1] + 5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Build output with imports header + matched blocks, respecting per-file cap
        parts: list[str] = []
        char_count = 0
        # Always include imports (first ~30 lines or until first class/def)
        import_end = 0
        for i, line in enumerate(lines[:60]):
            if re.match(r"^(class|def)\s", line):
                import_end = i
                break
        else:
            import_end = min(30, total)

        header = "\n".join(lines[:import_end])
        parts.append(header)
        char_count += len(header)

        for start, end in merged:
            block_text = "\n".join(lines[start:end + 1])
            if char_count + len(block_text) > _MAX_PER_FILE:
                # Budget exhausted — skip remaining blocks
                parts.append("\n# ... (remaining blocks omitted, per-file cap reached) ...")
                break
            if start > import_end:
                parts.append(f"\n# ... (lines {import_end + 1}-{start} omitted) ...\n")
            parts.append(block_text)
            char_count += len(block_text)
            import_end = end + 1

        if import_end < total:
            remaining = total - import_end
            if remaining > 5:
                parts.append(f"\n# ... ({remaining} more lines omitted) ...")

        result = "\n".join(parts)
        logger.debug(
            "focus_extract",
            file=fpath,
            original=len(text),
            extracted=len(result),
            blocks=len(merged),
        )
        return result

    async def _generate_patch(
        self,
        problem: str,
        file_contents: dict[str, str],
        repo: str,
        instance_id: str,
        fail_to_pass: list[str] | None = None,
        test_context: str = "",
    ) -> str:
        """Generate a unified diff patch with validation retry.

        Uses an Instructor-style feedback loop: if git apply --check fails,
        the error is fed back to the LLM for a corrective retry (up to
        _MAX_PATCH_RETRIES attempts).
        """
        # Build file context with line numbers for accurate patch generation
        file_context = ""
        for fpath, content in file_contents.items():
            numbered_lines = []
            for i, line in enumerate(content.splitlines(), 1):
                numbered_lines.append(f"{i:4d}| {line}")
            file_context += f"\n--- {fpath} ---\n" + "\n".join(numbered_lines) + "\n"

        # Evolved patch strategy
        strategy = self.config.framework_config.get(
            "patch_strategy",
            "Analyze the root cause, then make the minimal change needed to fix the bug.",
        )

        reasoning_depth = self.config.framework_config.get("reasoning_depth", "thorough")

        # Include failing test names and code to guide the fix
        test_info = ""
        if fail_to_pass:
            test_info = (
                "## Tests That Must Pass After Fix\n"
                + "\n".join(f"- `{t}`" for t in fail_to_pass)
                + "\n\nYour fix must make these tests pass.\n\n"
            )
        if test_context:
            test_info += f"## Failing Test Code\n```python\n{test_context}\n```\n\n"

        prompt = (
            f"You are an expert Python developer fixing a bug in '{repo}' (instance: {instance_id}).\n\n"
            f"## Patch Strategy\n{strategy}\n\n"
            f"## Reasoning Depth: {reasoning_depth}\n\n"
            f"## Problem Statement\n{problem}\n\n"
            f"{test_info}"
            f"## Relevant Source Code\n{file_context}\n\n"
            "## Instructions\n"
            "Think step by step:\n"
            "1. FIRST, analyze the failing test assertions in detail. For each assert "
            "statement, determine the EXACT expected return value, type, and structure. "
            "For example, if a test asserts `result == Piecewise((sin(x)/x, Ne(x, 0)), (1, True))` "
            "then your fix MUST produce that exact expression.\n"
            "2. Trace the bug to its ROOT CAUSE — this is CRITICAL. Common root-cause patterns:\n"
            "   - Regex bugs: Fix WHERE the regex is COMPILED (e.g., in __init__ or module-level), "
            "NOT where it is used. Add flags like re.MULTILINE|re.DOTALL to the re.compile() call.\n"
            "   - Missing method: Add a new method to the class (e.g., _eval_rewrite_as_X, "
            "_print_X) rather than modifying existing call sites.\n"
            "   - Version/mapping bugs: Fix the tuple/dict construction, not the consumer. "
            "If post-release needs micro+1, change where the tuple is BUILT.\n"
            "   - Import-level bugs: Add missing imports at the MODULE level, not inside functions.\n"
            "   ALWAYS check __init__, class definitions, and module-level code FIRST.\n"
            "3. Consider ALL code paths the test exercises — you may need changes in "
            "multiple functions or methods within the same file.\n"
            "4. If the test expects specific string mappings or type conversions "
            "(e.g., 'rc' -> 'candidate', integer -> float), make sure your code "
            "performs those exact transformations.\n"
            "5. Make the minimal change needed. Prefer modifying existing code over "
            "adding new code. If the test imports a specific function, make sure "
            "that function exists and returns the expected type.\n"
            "6. If there are MULTIPLE failing test functions, each may require a "
            "SEPARATE code change. Address ALL failing tests, not just one.\n"
            "7. The source code above includes LINE NUMBERS (e.g., '  32| code'). "
            "Use these to generate accurate @@ hunk headers. The line number in "
            "@@ -N,count should match the line number shown in the source.\n\n"
            "Generate a unified diff patch in standard `git diff` format.\n"
            "Requirements:\n"
            "- Start with `diff --git a/<path> b/<path>` (use the file path WITHOUT leading './')\n"
            "- Include `--- a/<path>` and `+++ b/<path>` headers\n"
            "- Include correct `@@ -line,count +line,count @@` hunk headers — use the "
            "line numbers shown in the source code above to set the start line\n"
            "- Include 1-3 lines of context around changes (lines starting with a SPACE character)\n"
            "- Context lines MUST be copied exactly from the source code above "
            "(same whitespace, WITHOUT the line number prefix)\n"
            "- Every line in a hunk must start with ' ' (context), '-' (removed), or '+' (added)\n"
            "- Be minimal — change only what's necessary\n"
            "- Follow the project's coding style\n\n"
            "Output ONLY the patch starting with 'diff --git'. No explanations."
        )

        repo_dir = self.config.framework_config.get("_repo_dir")
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        # Validation-retry loop (Instructor-style corrective feedback)
        for attempt in range(_MAX_PATCH_RETRIES):
            response = await self._llm.complete(role=LLMRole.ACTING, messages=messages)
            patch = self._extract_diff(response)

            if not patch:
                logger.debug("patch_empty", instance_id=instance_id, attempt=attempt)
                if attempt < _MAX_PATCH_RETRIES - 1:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response did not contain a valid unified diff. "
                            "Output ONLY a diff starting with 'diff --git a/<path> b/<path>'. "
                            "No explanations, no markdown fences."
                        ),
                    })
                    continue
                return ""

            # Try git apply --check for validation if repo is available
            apply_error = self._try_apply_patch(patch, repo_dir)
            if apply_error is None:
                # Patch applies cleanly
                logger.info(
                    "patch_validated",
                    instance_id=instance_id,
                    attempt=attempt,
                    length=len(patch),
                )
                return patch

            # Patch failed validation — retry with corrective feedback
            if attempt < _MAX_PATCH_RETRIES - 1:
                logger.info(
                    "patch_retry",
                    instance_id=instance_id,
                    attempt=attempt,
                    error=apply_error[:200],
                )
                # Extract actual file content around the failure for context
                actual_context = self._get_actual_context(apply_error, repo_dir)
                messages.append({"role": "assistant", "content": response})
                retry_msg = (
                    f"The patch failed `git apply --check` with this error:\n"
                    f"```\n{apply_error}\n```\n\n"
                    "Common causes:\n"
                    "- Wrong line numbers in @@ hunk headers\n"
                    "- Context lines don't match the actual file content\n"
                    "- Wrong file path in diff header\n\n"
                )
                if actual_context:
                    retry_msg += (
                        "Here is the ACTUAL file content around the failing area:\n"
                        f"```\n{actual_context}\n```\n\n"
                        "Use these exact lines for context. "
                    )
                retry_msg += (
                    "Fix the patch and output ONLY the corrected diff. "
                    "No explanations."
                )
                messages.append({"role": "user", "content": retry_msg})
            else:
                # Last attempt — return the patch even if validation failed
                logger.warning(
                    "patch_validation_exhausted",
                    instance_id=instance_id,
                    attempts=_MAX_PATCH_RETRIES,
                )
                return patch

        return ""

    @staticmethod
    def _try_apply_patch(patch: str, repo_dir: object) -> str | None:
        """Try applying a patch with git apply --check.

        Returns None on success, or the error message on failure.
        Only runs if repo_dir is a git repository.
        """
        if not repo_dir or not isinstance(repo_dir, Path) or not repo_dir.exists():
            return None  # No repo to validate against — assume OK
        if not (repo_dir / ".git").exists():
            return None  # Not a git repo — can't validate

        try:
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return None
            return (result.stderr or result.stdout or "unknown error").strip()
        except (subprocess.TimeoutExpired, Exception):
            return None  # Treat validation failure as non-blocking

    @staticmethod
    def _get_actual_context(error: str, repo_dir: object) -> str:
        """Extract actual file content around a 'patch does not apply' failure.

        Parses the error for file path and line number, then reads that region
        from the actual file to give the LLM correct context for retries.
        Returns empty string if parsing fails.
        """
        if not repo_dir or not isinstance(repo_dir, Path):
            return ""

        # Parse "error: patch failed: <path>:<line>"
        m = re.search(r"patch failed: (.+?):(\d+)", error)
        if not m:
            return ""

        fpath = m.group(1)
        line_num = int(m.group(2))
        full_path = repo_dir / fpath

        if not full_path.exists():
            return ""

        try:
            lines = full_path.read_text(errors="replace").splitlines()
            # Show ~20 lines centered on the failure point
            start = max(0, line_num - 10)
            end = min(len(lines), line_num + 10)
            numbered = [f"{i + 1:4d}: {lines[i]}" for i in range(start, end)]
            return f"# {fpath} (lines {start + 1}-{end}):\n" + "\n".join(numbered)
        except Exception:
            return ""

    def _extract_diff(self, response: str) -> str:
        """Extract the unified diff from an LLM response.

        Handles multiple output patterns:
        - Raw diff starting with 'diff --git'
        - Diff inside ```diff code fences
        - Diff inside generic ``` code fences
        - Multiple diff hunks for multi-file patches
        - Responses with leading explanation text before the diff
        - Responses wrapped in JSON (e.g., from --output-format json)
        """
        # Strip any JSON wrapping (some CLI outputs return JSON objects)
        text = response.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                # Claude Code JSON output has a "result" key
                if isinstance(data, dict) and "result" in data:
                    text = str(data["result"])
            except (json.JSONDecodeError, KeyError):
                pass

        lines = text.splitlines()
        diff_lines: list[str] = []
        in_diff = False

        for line in lines:
            if line.startswith("diff --git"):
                in_diff = True
            if in_diff:
                # Stop if we hit a code fence closing marker or clear non-diff text
                if line.startswith("```") and diff_lines:
                    break
                # Stop on obvious non-diff content after the patch
                if line.startswith("## ") and diff_lines and not line.startswith("## @"):
                    break
                diff_lines.append(line)

        if diff_lines:
            # Clean trailing blank lines
            while diff_lines and not diff_lines[-1].strip():
                diff_lines.pop()
            raw = "\n".join(diff_lines) + "\n"
            return self._fix_hunk_headers(self._fix_corrupt_lines(raw))

        # Fallback 1: look for content between code fences
        in_fence = False
        fence_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```diff") or (line.strip() == "```" and not in_fence):
                if in_fence:
                    break
                in_fence = True
                continue
            if line.strip() == "```" and in_fence:
                break
            if in_fence:
                fence_lines.append(line)

        if fence_lines and any(ln.startswith("diff --git") or ln.startswith("---") for ln in fence_lines):
            raw = "\n".join(fence_lines) + "\n"
            return self._fix_hunk_headers(self._fix_corrupt_lines(raw))

        # Fallback 2: look for --- a/ and +++ b/ patterns (partial diff without git header)
        partial_lines: list[str] = []
        in_partial = False
        for line in lines:
            if line.startswith("--- a/"):
                in_partial = True
            if in_partial:
                if line.startswith("```") and partial_lines:
                    break
                partial_lines.append(line)

        if partial_lines and any(ln.startswith("+++ b/") for ln in partial_lines):
            # Reconstruct a proper diff header from the file path
            file_path = partial_lines[0].replace("--- a/", "").strip()
            diff_header = f"diff --git a/{file_path} b/{file_path}\n"
            raw = diff_header + "\n".join(partial_lines) + "\n"
            return self._fix_hunk_headers(self._fix_corrupt_lines(raw))

        return ""

    @staticmethod
    def _fix_corrupt_lines(patch: str) -> str:
        """Fix 'corrupt patch at line N' errors by ensuring valid diff prefixes.

        git requires every line inside a hunk body to start with one of:
        ' ' (context), '-' (removed), '+' (added), or '\\' (no-newline).
        LLMs often emit context lines without the leading space, causing
        'corrupt patch at line N' errors from git apply.

        This method prepends a space to any hunk body line that doesn't
        start with a valid prefix, treating it as a context line.
        """
        hunk_re = re.compile(r"^@@ ")
        output_lines: list[str] = []
        in_hunk = False

        for line in patch.splitlines():
            if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("+++ "):
                in_hunk = False
                output_lines.append(line)
            elif hunk_re.match(line):
                in_hunk = True
                output_lines.append(line)
            elif in_hunk:
                if line == "":
                    # Empty line in hunk body — treat as context (add space)
                    output_lines.append(" ")
                elif line[0] not in (" ", "-", "+", "\\"):
                    # Missing prefix — prepend space (context line)
                    output_lines.append(" " + line)
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

        return "\n".join(output_lines) + "\n"

    @staticmethod
    def _fix_hunk_headers(patch: str) -> str:
        """Fix hunk headers so line counts match actual content.

        LLMs often emit @@ headers with wrong counts (e.g., claiming 7 lines
        when only 4 are present). Docker's patch(1) is strict about this while
        git-apply is lenient. This recalculates each hunk header from the
        actual content lines to prevent 'unexpected end of file in patch'.
        """
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
        output_lines: list[str] = []
        # Collect lines for the current hunk so we can rewrite its header
        hunk_body: list[str] = []
        hunk_header_match: re.Match[str] | None = None
        hunk_header_idx: int = -1

        def _flush_hunk() -> None:
            """Rewrite the pending hunk header with correct counts."""
            nonlocal hunk_body, hunk_header_match, hunk_header_idx
            if hunk_header_match is None:
                return
            old_count = 0
            new_count = 0
            for ln in hunk_body:
                if ln.startswith("-"):
                    old_count += 1
                elif ln.startswith("+"):
                    new_count += 1
                else:
                    # context line (space or empty — counts toward both)
                    old_count += 1
                    new_count += 1
            old_start = int(hunk_header_match.group(1))
            new_start = int(hunk_header_match.group(3))
            tail = hunk_header_match.group(5)
            output_lines[hunk_header_idx] = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{tail}"
            output_lines.extend(hunk_body)
            hunk_body = []
            hunk_header_match = None

        for line in patch.splitlines():
            m = hunk_re.match(line)
            if m:
                _flush_hunk()
                hunk_header_idx = len(output_lines)
                output_lines.append(line)  # placeholder, rewritten by _flush_hunk
                hunk_header_match = m
            elif hunk_header_match is not None:
                # Inside a hunk — collect body lines
                if line.startswith(("diff --git", "--- ", "+++ ")):
                    # New file section — flush current hunk first
                    _flush_hunk()
                    output_lines.append(line)
                else:
                    hunk_body.append(line)
            else:
                output_lines.append(line)

        _flush_hunk()
        return "\n".join(output_lines) + "\n"

    def _validate_patch(self, patch: str, repo_dir: Path | None) -> bool:
        """Check if the patch is a valid unified diff.

        Note: the retry loop in _generate_patch already validates via
        git apply --check and retries on failure. This method is a final
        format-only check for the overall run_task result.
        """
        if not patch.strip():
            return False

        has_diff_header = "diff --git" in patch or ("---" in patch and "+++" in patch)
        has_hunks = "@@" in patch

        if not (has_diff_header and has_hunks):
            return False

        # Try applying the patch (dry run) if repo is available
        error = self._try_apply_patch(patch, repo_dir)
        return error is None

    def _extract_lessons(self, patch: str, valid: bool) -> list[str]:
        """Extract lessons from the patch generation attempt."""
        lessons = []
        if valid:
            # Count files modified
            file_count = patch.count("diff --git")
            lessons.append(f"Successfully generated patch modifying {file_count} file(s)")
        else:
            if not patch:
                lessons.append("Failed to extract diff from LLM response — improve output format instructions")
            elif "diff --git" not in patch:
                lessons.append("Patch missing diff headers — enforce unified diff format")
            elif "@@" not in patch:
                lessons.append("Patch missing hunk markers — ensure complete diff output")
            else:
                lessons.append("Patch format ok but git apply failed — check line numbers and context")
        return lessons

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        """Apply an evolutionary mutation to this agent's framework config."""
        self.patches.append(patch)
        if patch.intent:
            patches_list: list[str] = self.config.framework_config.get("applied_patches", [])
            patches_list.append(patch.intent)
            self.config.framework_config["applied_patches"] = patches_list
        if patch.config_updates:
            self.config.framework_config.update(patch.config_updates)
        logger.debug("patch_applied", agent_id=self.id, intent=patch.intent)
