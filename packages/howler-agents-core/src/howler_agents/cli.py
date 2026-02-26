"""CLI entry point for howler-agents."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Howler Agents - Group-Evolving AI Agents."""
    pass


# --------------------------------------------------------------------------- #
# serve                                                                        #
# --------------------------------------------------------------------------- #


@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    show_default=True,
    help="Transport protocol for the MCP server.",
)
@click.option(
    "--port",
    type=int,
    default=8765,
    show_default=True,
    help="Port to listen on when using the SSE transport.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind when using the SSE transport.",
)
@click.option(
    "--api-url",
    envvar="HOWLER_API_URL",
    default=None,
    help="Remote howler-agents API URL. When set, the server proxies calls to that URL. "
    "Defaults to local in-memory mode.",
)
def serve(transport: str, port: int, host: str, api_url: str | None) -> None:
    """Start the MCP server for use with AI coding tools.

    \b
    Stdio transport (for Claude Code, Cursor, OpenCode, Codex):
        howler-agents serve --transport stdio

    \b
    SSE transport (for remote/network access):
        howler-agents serve --transport sse --port 8765
    """
    # Set env var so mcp_server picks it up
    if api_url:
        os.environ["HOWLER_API_URL"] = api_url

    try:
        from howler_agents.mcp_server import run_sse, run_stdio
    except ImportError as exc:
        click.echo(
            f"Error: MCP dependencies missing. Install with: pip install 'howler-agents-core[mcp]'\n{exc}",
            err=True,
        )
        sys.exit(1)

    if transport == "stdio":
        click.echo("Starting howler-agents MCP server (stdio transport)…", err=True)
        asyncio.run(run_stdio())
    else:
        click.echo(f"Starting howler-agents MCP server (SSE transport) on {host}:{port}…", err=True)
        asyncio.run(run_sse(host=host, port=port))


# --------------------------------------------------------------------------- #
# evolve                                                                       #
# --------------------------------------------------------------------------- #


@cli.command()
@click.option("--population", "-p", default=10, show_default=True, help="Population size (K).")
@click.option("--groups", "-g", default=3, show_default=True, help="Agents per group (M).")
@click.option("--iterations", "-n", default=5, show_default=True, help="Number of generations.")
@click.option(
    "--model",
    "-m",
    default="claude-sonnet-4-20250514",
    show_default=True,
    help="LiteLLM model string for all roles.",
)
@click.option(
    "--domain",
    "-d",
    default="general",
    show_default=True,
    type=click.Choice(["general", "coding", "math", "writing", "swe-bench"]),
    help="Task domain for the evolution run.",
)
@click.option(
    "--alpha",
    "-a",
    default=0.5,
    show_default=True,
    help="Performance vs novelty balance (0.0=pure novelty, 1.0=pure performance).",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="API key for the LLM provider (also read from env).",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Output the final run result as JSON.",
)
def evolve(
    population: int,
    groups: int,
    iterations: int,
    model: str,
    domain: str,
    alpha: float,
    api_key: str | None,
    json_output: bool,
) -> None:
    """Run a local evolution loop and print results.

    \b
    Example:
        howler-agents evolve --population 8 --iterations 3 --domain coding --model gpt-4o
    """
    try:
        from howler_agents.local_runner import LocalRunner
    except ImportError as exc:
        click.echo(f"Error loading howler_agents: {exc}", err=True)
        sys.exit(1)

    runner = LocalRunner()
    if api_key:
        runner.configure(api_key=api_key)

    click.echo(
        f"Starting evolution: {population} agents, {iterations} generations, domain={domain}, model={model}",
        err=True,
    )

    run_id = runner.start_run(
        population_size=population,
        group_size=groups,
        num_iterations=iterations,
        alpha=alpha,
        task_domain=domain,
        model=model,
    )

    async def _run() -> dict:
        return await runner.run_async(run_id)

    try:
        result = asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Evolution failed: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return

    # Human-readable summary
    best_score = result.get("best_score", 0.0)
    generations = result.get("generations", [])

    click.echo("\n--- Evolution Complete ---")
    click.echo(f"Run ID   : {run_id}")
    click.echo(f"Domain   : {domain}")
    click.echo(f"Generations: {len(generations)}")
    click.echo(f"Best score : {best_score:.4f}")

    if generations:
        click.echo("\nGeneration summaries:")
        for gen in generations:
            click.echo(
                f"  Gen {gen['generation']:>2}: best={gen['best_score']:.4f}  "
                f"mean={gen['mean_score']:.4f}  pop={gen['population_size']}"
            )

    top_agents = runner.list_agents(run_id, top_k=3)
    if top_agents:
        click.echo("\nTop agents:")
        for rank, agent in enumerate(top_agents, 1):
            click.echo(
                f"  #{rank} {agent['agent_id'][:8]}…  "
                f"score={agent['combined_score']:.4f}  "
                f"gen={agent['generation']}  "
                f"group={agent['group_id']}"
            )


# --------------------------------------------------------------------------- #
# status                                                                       #
# --------------------------------------------------------------------------- #


@cli.command()
@click.option(
    "--run-id",
    default=None,
    help="Specific run ID to show. If omitted, shows all runs in the current process.",
)
@click.option("--json-output", is_flag=True, default=False, help="Output as JSON.")
def status(run_id: str | None, json_output: bool) -> None:
    """Show status of local runs.

    NOTE: Because howler-agents stores run state in memory, this command only
    shows runs that were started in the same process invocation (e.g. when
    called programmatically or from a long-running serve session).
    For persistent status across invocations, use a remote API with HOWLER_API_URL.
    """
    try:
        from howler_agents.local_runner import LocalRunner
    except ImportError as exc:
        click.echo(f"Error loading howler_agents: {exc}", err=True)
        sys.exit(1)

    # A fresh LocalRunner will have no runs. This command is most useful when
    # the runner is shared via the MCP server process.
    runner = LocalRunner()
    runs = runner.list_runs()

    if run_id:
        try:
            run_status = runner.get_status(run_id)
            runs = [run_status]
        except KeyError:
            click.echo(f"No run found with id: {run_id}", err=True)
            sys.exit(1)

    if json_output:
        click.echo(json.dumps(runs, indent=2))
        return

    if not runs:
        click.echo("No runs found in this process. Start a run with: howler-agents evolve")
        return

    for run in runs:
        click.echo(f"\nRun: {run['run_id']}")
        click.echo(f"  Status     : {run['status']}")
        click.echo(f"  Domain     : {run['task_domain']}")
        click.echo(f"  Generation : {run['current_generation']} / {run['total_generations']}")
        click.echo(f"  Population : {run['population_size']}")
        click.echo(f"  Best score : {run['best_score']:.4f}")
        click.echo(f"  Mean score : {run['mean_score']:.4f}")
        click.echo(f"  Started at : {run['started_at']}")
        if run["finished_at"]:
            click.echo(f"  Finished at: {run['finished_at']}")
        if run["error"]:
            click.echo(f"  Error      : {run['error']}")


# --------------------------------------------------------------------------- #
# configure                                                                    #
# --------------------------------------------------------------------------- #


@cli.command()
@click.option("--acting-model", default=None, help="Model for agents performing tasks.")
@click.option("--evolving-model", default=None, help="Model for generating mutations.")
@click.option("--reflecting-model", default=None, help="Model for reflective analysis.")
@click.option("--show", is_flag=True, default=False, help="Show current configuration.")
def configure(
    acting_model: str | None,
    evolving_model: str | None,
    reflecting_model: str | None,
    show: bool,
) -> None:
    """Configure LLM model assignments for the evolutionary roles.

    \b
    Example:
        howler-agents configure --acting-model gpt-4o --evolving-model claude-opus-4-6
    """
    try:
        from howler_agents.local_runner import LocalRunner
    except ImportError as exc:
        click.echo(f"Error loading howler_agents: {exc}", err=True)
        sys.exit(1)

    runner = LocalRunner()

    if show:
        cfg = runner.get_session_config()
        if not cfg:
            click.echo("No session configuration set. Using defaults.")
        else:
            for key, value in cfg.items():
                if key == "api_key":
                    click.echo("  api_key: ***")
                else:
                    click.echo(f"  {key}: {value}")
        return

    if not any([acting_model, evolving_model, reflecting_model]):
        click.echo(
            "No configuration provided. Use --acting-model, --evolving-model, or --reflecting-model."
        )
        return

    runner.configure(
        acting_model=acting_model,
        evolving_model=evolving_model,
        reflecting_model=reflecting_model,
    )
    click.echo("Configuration updated.")
    if acting_model:
        click.echo(f"  acting_model    = {acting_model}")
    if evolving_model:
        click.echo(f"  evolving_model  = {evolving_model}")
    if reflecting_model:
        click.echo(f"  reflecting_model = {reflecting_model}")


# --------------------------------------------------------------------------- #
# install                                                                      #
# --------------------------------------------------------------------------- #

_HOST_CHOICES = [
    "auto",
    "claude-code",
    "codex",
    "gemini",
    "cursor",
    "windsurf",
    "vscode",
    "opencode",
    "continue",
    "zed",
    "amazon-q",
    "jetbrains",
    "cline",
    "all",
]

_MANUAL_HOSTS = {"continue", "zed", "cline"}


def _detect_hosts() -> list[str]:
    """Return list of host names that appear to be present in the environment."""
    cwd = Path.cwd()
    home = Path.home()
    detected: list[str] = []

    checks: list[tuple[str, bool]] = [
        ("claude-code", (cwd / ".mcp.json").exists() or bool(os.environ.get("CLAUDE_CODE"))),
        ("codex", (home / ".codex").is_dir()),
        ("gemini", (home / ".gemini").is_dir()),
        ("cursor", (cwd / ".cursor").is_dir() or (home / ".cursor").is_dir()),
        ("windsurf", (home / ".codeium" / "windsurf").is_dir()),
        ("vscode", (cwd / ".vscode").is_dir()),
        ("opencode", (cwd / "opencode.json").exists() or (cwd / "opencode.jsonc").exists()),
        ("continue", (cwd / ".continue").is_dir()),
        ("zed", (home / ".config" / "zed").is_dir()),
        ("amazon-q", (cwd / ".amazonq").is_dir() or (home / ".aws" / "amazonq").is_dir()),
        ("jetbrains", (cwd / ".idea").is_dir()),
        (
            "cline",
            any(
                p.is_dir()
                for p in (home / ".vscode" / "extensions").glob("saoudrizwan.claude-dev*")
                if (home / ".vscode" / "extensions").is_dir()
            ),
        ),
    ]

    for name, present in checks:
        if present:
            detected.append(name)

    return detected


def _build_config(host: str, command: str) -> tuple[str, object] | None:
    """Return (config_path_template, config_data) or None for manual-only hosts."""
    # Support npx-style invocation: command="npx" wraps args
    if command == "npx":
        cmd = "npx"
        args = ["howler-agents@latest", "serve", "--transport", "stdio"]
    elif command == "uvx":
        cmd = "uvx"
        args = [
            "--from",
            "howler-agents-core[mcp]",
            "howler-agents",
            "serve",
            "--transport",
            "stdio",
        ]
    else:
        cmd = command
        args = ["serve", "--transport", "stdio"]

    if host == "claude-code":
        return ".mcp.json", {"mcpServers": {"howler-agents": {"command": cmd, "args": args}}}

    if host == "codex":
        return "~/.codex/config.toml", None  # TOML handled separately

    if host == "gemini":
        return "~/.gemini/settings.json", {
            "mcpServers": {"howler-agents": {"command": cmd, "args": args}}
        }

    if host == "cursor":
        return ".cursor/mcp.json", {"mcpServers": {"howler-agents": {"command": cmd, "args": args}}}

    if host == "windsurf":
        return "~/.codeium/windsurf/mcp_config.json", {
            "mcpServers": {"howler-agents": {"command": cmd, "args": args}}
        }

    if host == "vscode":
        return ".vscode/mcp.json", {
            "servers": {"howler-agents": {"type": "stdio", "command": cmd, "args": args}}
        }

    if host == "opencode":
        return "opencode.json", {
            "mcp": {"howler-agents": {"type": "local", "command": [cmd, *args], "enabled": True}}
        }

    if host == "amazon-q":
        return ".amazonq/mcp.json", {
            "mcpServers": {"howler-agents": {"command": cmd, "args": args}}
        }

    if host == "jetbrains":
        return ".idea/mcp.json", {"mcpServers": {"howler-agents": {"command": cmd, "args": args}}}

    return None


def _write_json_config(path: Path, patch: dict) -> None:
    """Merge patch into an existing JSON file or create it."""
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    def _deep_merge(base: dict, update: dict) -> dict:
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged = _deep_merge(existing, patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n")


def _write_codex_toml(path: Path, command: str) -> None:
    """Append or create the Codex TOML config snippet."""
    if command == "npx":
        args_str = '["howler-agents@latest", "serve", "--transport", "stdio"]'
        cmd = "npx"
    elif command == "uvx":
        args_str = '["--from", "howler-agents-core[mcp]", "howler-agents", "serve", "--transport", "stdio"]'
        cmd = "uvx"
    else:
        args_str = '["serve", "--transport", "stdio"]'
        cmd = command
    snippet = f'\n[mcp_servers.howler-agents]\ncommand = "{cmd}"\nargs = {args_str}\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    if "[mcp_servers.howler-agents]" in existing:
        return  # already present, skip
    with path.open("a") as fh:
        fh.write(snippet)


def _resolve_path(template: str, global_scope: bool) -> Path:
    """Resolve a path template, honouring --global by forcing home-relative paths."""
    if template.startswith("~"):
        return Path(template).expanduser()

    # project-local paths
    if global_scope:
        # For global scope, fall back to home-relative equivalents where sensible.
        mapping = {
            ".mcp.json": Path.home() / ".mcp.json",
            ".cursor/mcp.json": Path.home() / ".cursor" / "mcp.json",
            ".vscode/mcp.json": Path.home() / ".vscode" / "mcp.json",
            ".amazonq/mcp.json": Path.home() / ".aws" / "amazonq" / "mcp.json",
            ".idea/mcp.json": Path.home() / ".config" / "jetbrains" / "mcp.json",
            "opencode.json": Path.home() / "opencode.json",
        }
        if template in mapping:
            return mapping[template]

    return Path.cwd() / template


@cli.command("install")
@click.option(
    "--host",
    type=click.Choice(_HOST_CHOICES),
    default="auto",
    show_default=True,
    help="MCP host to configure. Use 'auto' to detect, 'all' for every host.",
)
@click.option(
    "--command",
    "command",
    default="howler-agents",
    show_default=True,
    help="Server command: 'howler-agents' (direct), 'npx' (auto-install via npm), or 'uvx' (auto-install via uv).",
)
@click.option(
    "--global",
    "global_scope",
    is_flag=True,
    default=False,
    help="Write to the global/user config instead of the project config.",
)
def install(host: str, command: str, global_scope: bool) -> None:
    """Install howler-agents as an MCP server in one or more AI coding tools.

    \b
    Auto-detect installed hosts and configure them:
        howler-agents install

    \b
    Target a specific host:
        howler-agents install --host cursor

    \b
    Use npx for zero-install setup (recommended for teams):
        howler-agents install --command npx

    \b
    Use uvx for Python-native auto-install:
        howler-agents install --command uvx

    \b
    Configure all supported hosts at once:
        howler-agents install --host all

    \b
    Write to global config instead of project config:
        howler-agents install --host claude-code --global
    """
    all_configurable = [h for h in _HOST_CHOICES if h not in ("auto", "all")]

    if host == "auto":
        targets = _detect_hosts()
        if not targets:
            click.echo("No supported MCP hosts detected in this environment.")
            return
        click.echo(f"Detected hosts: {', '.join(targets)}")
    elif host == "all":
        targets = all_configurable
    else:
        targets = [host]

    for target in targets:
        if target in _MANUAL_HOSTS:
            click.echo(f"  [{target}] Configuration for {target} requires manual setup.")
            continue

        result = _build_config(target, command)
        if result is None:
            click.echo(f"  [{target}] No config template available — skipping.")
            continue

        path_template, config_data = result
        dest: Path = _resolve_path(path_template, global_scope)

        try:
            if target == "codex":
                _write_codex_toml(dest, command)
            else:
                assert isinstance(config_data, dict)
                _write_json_config(dest, config_data)
            click.echo(f"  [{target}] Written to {dest}")
        except OSError as exc:
            click.echo(f"  [{target}] Failed to write {dest}: {exc}", err=True)


# --------------------------------------------------------------------------- #
# swe-bench                                                                    #
# --------------------------------------------------------------------------- #


@cli.command("swe-bench")
@click.option(
    "--dataset",
    default="princeton-nlp/SWE-bench_Lite",
    show_default=True,
    help="HuggingFace dataset path.",
)
@click.option(
    "--limit", "-n", default=5, show_default=True, help="Number of instances to evaluate."
)
@click.option(
    "--model",
    "-m",
    default="auto",
    show_default=True,
    help="LLM backend: 'auto' detects local CLI (claude-code, codex, gemini-cli, opencode), or specify e.g. 'claude-code/sonnet', 'codex/default', or a LiteLLM model string.",
)
@click.option(
    "--population", "-p", default=6, show_default=True, help="Population size (K) for evolution."
)
@click.option("--groups", "-g", default=2, show_default=True, help="Agents per group (M).")
@click.option(
    "--iterations", default=2, show_default=True, help="Evolution generations before eval."
)
@click.option(
    "--alpha", "-a", default=0.7, show_default=True, help="Performance vs novelty balance."
)
@click.option(
    "--skip-docker-eval",
    is_flag=True,
    default=False,
    help="Skip Docker-based evaluation (just generate predictions).",
)
@click.option(
    "--workspace",
    default=None,
    help="Directory for repos/predictions. Default: .howler-agents/swe-bench",
)
@click.option("--run-id", default=None, help="Custom run ID.")
@click.option("--instance-ids", default=None, help="Comma-separated instance IDs to evaluate.")
@click.option("--json-output", is_flag=True, default=False, help="Output report as JSON.")
@click.option("--check", is_flag=True, default=False, help="Only validate setup prerequisites.")
@click.option(
    "--max-concurrent", default=3, show_default=True, help="Max parallel prediction workers."
)
@click.option(
    "--instance-timeout", default=300.0, show_default=True, help="Per-instance timeout in seconds."
)
@click.option(
    "--skip-evolution",
    is_flag=True,
    default=False,
    help="Skip GEA evolution, use default agent directly.",
)
def swe_bench(
    dataset: str,
    limit: int,
    model: str,
    population: int,
    groups: int,
    iterations: int,
    alpha: float,
    skip_docker_eval: bool,
    skip_evolution: bool,
    workspace: str | None,
    run_id: str | None,
    instance_ids: str | None,
    json_output: bool,
    check: bool,
    max_concurrent: int,
    instance_timeout: float,
) -> None:
    """Run SWE-Bench evaluation with GEA-evolved agents.

    \b
    Quick check that prerequisites are installed:
        howler-agents swe-bench --check

    \b
    Evaluate on 5 instances with default settings:
        howler-agents swe-bench --limit 5

    \b
    Full SWE-bench Lite run (300 instances):
        howler-agents swe-bench --limit 0 --population 50 --iterations 60 --alpha 0.7

    \b
    Specific instances only:
        howler-agents swe-bench --instance-ids sympy__sympy-20590,django__django-16379

    \b
    Generate predictions without Docker eval:
        howler-agents swe-bench --skip-docker-eval --limit 10
    """
    if check:
        from howler_agents.benchmarks.swe_bench_harness import SWEBenchHarness

        harness = SWEBenchHarness()
        checks = harness.validate_setup()

        if json_output:
            click.echo(json.dumps(checks, indent=2))
            return

        click.echo("SWE-Bench Setup Check:")
        for name, c in checks.items():
            if name == "ready":
                continue
            ok = c.get("ok", False)
            icon = "+" if ok else "-"
            click.echo(f"  [{icon}] {name}")
            if not ok and "fix" in c:
                click.echo(f"      Fix: {c['fix']}")

        if checks.get("ready"):
            click.echo("\nAll checks passed. Ready to run SWE-bench evaluation.")
        else:
            click.echo("\nSome checks failed. Fix the issues above before running evaluation.")
            sys.exit(1)
        return

    from howler_agents.benchmarks.swe_bench_runner import SWEBenchEvalRunner

    ids = [s.strip() for s in instance_ids.split(",")] if instance_ids else None
    actual_limit = None if limit == 0 else limit

    runner = SWEBenchEvalRunner(
        model=model,
        dataset=dataset,
        workspace=workspace,
        population_size=population,
        group_size=groups,
        num_iterations=0 if skip_evolution else iterations,
        alpha=alpha,
        skip_docker_eval=skip_docker_eval,
        max_concurrent=max_concurrent,
        instance_timeout_s=instance_timeout,
    )

    click.echo(f"SWE-Bench Evaluation: {dataset}", err=True)
    click.echo(f"  Model:      {model}", err=True)
    click.echo(f"  Population: {population}, Groups: {groups}", err=True)
    click.echo(f"  Iterations: {iterations}, Alpha: {alpha}", err=True)
    click.echo(f"  Parallel:   {max_concurrent}, Timeout: {instance_timeout}s", err=True)
    click.echo(f"  Instances:  {actual_limit or 'all'}", err=True)
    click.echo("", err=True)

    async def _run() -> tuple[dict, str]:
        report = await runner.run(
            limit=actual_limit,
            instance_ids=ids,
            run_id=run_id,
        )
        return report.to_dict(), report.summary()

    try:
        result_dict, summary = asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Evaluation failed: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(result_dict, indent=2))
    else:
        click.echo(summary)


# --------------------------------------------------------------------------- #
# init                                                                         #
# --------------------------------------------------------------------------- #

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _copy_tree(src: Path, dst: Path, overwrite: bool) -> list[str]:
    """Recursively copy src into dst. Returns list of written paths."""
    written: list[str] = []
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.read_bytes())
        # Preserve executable bit
        if os.access(item, os.X_OK):
            target.chmod(target.stat().st_mode | 0o111)
        written.append(str(rel))
    return written


def _ensure_gitignore_entry(root: Path, entry: str) -> bool:
    """Add entry to .gitignore if not already present. Returns True if added."""
    gitignore = root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        # Check for exact line match
        for line in content.splitlines():
            if line.strip() == entry:
                return False
        # Append
        if not content.endswith("\n"):
            content += "\n"
        content += f"{entry}\n"
        gitignore.write_text(content)
    else:
        gitignore.write_text(f"{entry}\n")
    return True


def _ensure_mcp_json(root: Path, command: str) -> bool:
    """Add howler-agents to .mcp.json. Returns True if modified."""
    if command == "npx":
        cmd = "npx"
        args = ["howler-agents@latest", "serve", "--transport", "stdio"]
    elif command == "uvx":
        cmd = "uvx"
        args = [
            "--from",
            "howler-agents-core[mcp]",
            "howler-agents",
            "serve",
            "--transport",
            "stdio",
        ]
    else:
        cmd = "howler-agents"
        args = ["serve", "--transport", "stdio"]

    mcp_json = root / ".mcp.json"
    existing: dict = {}
    if mcp_json.exists():
        try:
            existing = json.loads(mcp_json.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    servers = existing.setdefault("mcpServers", {})
    if "howler-agents" in servers:
        return False

    servers["howler-agents"] = {"command": cmd, "args": args}
    mcp_json.write_text(json.dumps(existing, indent=2) + "\n")
    return True


@cli.command("init")
@click.option(
    "--command",
    "mcp_command",
    default="npx",
    show_default=True,
    type=click.Choice(["npx", "uvx", "howler-agents"]),
    help="MCP server command: 'npx' (zero-install), 'uvx' (Python), or 'howler-agents' (direct).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing skill/agent files.",
)
@click.option(
    "--skip-skills",
    is_flag=True,
    default=False,
    help="Skip installing Claude Code skills.",
)
@click.option(
    "--skip-agents",
    is_flag=True,
    default=False,
    help="Skip installing Claude Code agent definitions.",
)
@click.option(
    "--skip-mcp",
    is_flag=True,
    default=False,
    help="Skip registering the MCP server in .mcp.json.",
)
def init(
    mcp_command: str,
    overwrite: bool,
    skip_skills: bool,
    skip_agents: bool,
    skip_mcp: bool,
) -> None:
    """Initialize howler-agents in the current repository.

    Sets up Claude Code skills, agent definitions, MCP server registration,
    and the local .howler-agents/ directory. Run this once in any repo to
    enable all /howler-* slash commands.

    \b
    Quick setup (recommended):
        npx howler-agents init

    \b
    Using uvx (Python-native):
        npx howler-agents init --command uvx

    \b
    Overwrite existing files:
        npx howler-agents init --overwrite

    \b
    MCP registration only (no skills):
        npx howler-agents init --skip-skills --skip-agents
    """
    root = Path.cwd()
    click.echo("howler-agents init")
    click.echo(f"  Directory: {root}\n")

    # Verify templates exist
    if not _TEMPLATES_DIR.is_dir():
        click.echo(
            "Error: Template files not found. "
            "Reinstall with: pip install 'howler-agents-core[mcp]'",
            err=True,
        )
        sys.exit(1)

    # 1. Create .howler-agents/ local directory
    howler_dir = root / ".howler-agents"
    if howler_dir.exists():
        click.echo("  [ok] .howler-agents/ already exists")
    else:
        howler_dir.mkdir(parents=True)
        click.echo("  [+] Created .howler-agents/")

    # 2. Add .howler-agents/ to .gitignore
    if _ensure_gitignore_entry(root, ".howler-agents/"):
        click.echo("  [+] Added .howler-agents/ to .gitignore")
    else:
        click.echo("  [ok] .howler-agents/ already in .gitignore")

    # Also add the claude local memory DB
    if _ensure_gitignore_entry(root, ".claude/memory.db"):
        click.echo("  [+] Added .claude/memory.db to .gitignore")

    # 3. Install skills
    if not skip_skills:
        skills_src = _TEMPLATES_DIR / "skills"
        skills_dst = root / ".claude" / "skills"
        if skills_src.is_dir():
            written = _copy_tree(skills_src, skills_dst, overwrite)
            if written:
                # Count unique skill directories
                skill_names = {Path(w).parts[0] for w in written}
                click.echo(
                    f"  [+] Installed {len(skill_names)} skills "
                    f"({len(written)} files) to .claude/skills/"
                )
                for name in sorted(skill_names):
                    click.echo(f"      /{''.join(name)}")
            else:
                click.echo("  [ok] All skills already exist (use --overwrite to replace)")
        else:
            click.echo("  [!] No skill templates found in package", err=True)

    # 4. Install agent definitions
    if not skip_agents:
        agents_src = _TEMPLATES_DIR / "agents"
        agents_dst = root / ".claude" / "agents"
        if agents_src.is_dir():
            written = _copy_tree(agents_src, agents_dst, overwrite)
            if written:
                click.echo(f"  [+] Installed {len(written)} agent definitions to .claude/agents/")
            else:
                click.echo(
                    "  [ok] All agent definitions already exist (use --overwrite to replace)"
                )
        else:
            click.echo("  [!] No agent templates found in package", err=True)

    # 5. Register MCP server
    if not skip_mcp:
        if _ensure_mcp_json(root, mcp_command):
            click.echo(f"  [+] Registered MCP server in .mcp.json (command: {mcp_command})")
        else:
            click.echo("  [ok] MCP server already registered in .mcp.json")

    # Summary
    click.echo("\nDone! Available slash commands:")
    click.echo("  /howler-agents          Best-first-pass solution (hive-mind + GEA)")
    click.echo("  /howler-agents-wiggam   Iterative loop with hive-mind + GEA per iteration")
    click.echo("  /howler-init            Seed hive-mind with repo intelligence")
    click.echo("  /howler-evolve          Start a full GEA evolution run")
    click.echo("  /howler-auto-evolve     One-shot evolution with auto-deployment")
    click.echo("  /howler-setup           Re-run local environment setup")
    click.echo("  /howler-status          Check run progress")
    click.echo("  /howler-memory          Browse collective memory")
    click.echo("  /howler-sync            Sync with team database")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Run /howler-init to seed the hive-mind with repo intelligence")
    click.echo("  2. Run /howler-agents <task> to solve tasks with collective intelligence")


if __name__ == "__main__":
    cli()
