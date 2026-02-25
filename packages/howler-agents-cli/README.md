# howler-agents

Group-Evolving AI Agents (GEA) — groups of AI agents evolve together by sharing experience across lineages.

Implements the GEA system from [arXiv:2602.04837](https://arxiv.org/abs/2602.04837), achieving **73.3% on SWE-bench Lite**.

## Quick Start

Initialize any repository with howler-agents skills, MCP server, and agent definitions:

```bash
npx howler-agents init
```

This sets up:
- **9 Claude Code skills** (`/howler-agents`, `/howler-agents-wiggam`, `/howler-init`, etc.)
- **4 agent definitions** (coordinator, evaluator, reproducer, actor)
- **MCP server** registration in `.mcp.json`
- **Local `.howler-agents/`** directory

Then use the slash commands in Claude Code:

```bash
/howler-init                           # Seed hive-mind with repo knowledge
/howler-agents Fix the auth bug        # Solve a task with collective intelligence
/howler-agents-wiggam Fix all tests \
  --completion-promise "ALL TESTS PASSING"  # Iterate until done
```

## Init Options

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | `npx` | MCP server command: `npx`, `uvx`, or `howler-agents` |
| `--overwrite` | off | Replace existing skill/agent files |
| `--skip-skills` | off | Skip installing Claude Code skills |
| `--skip-agents` | off | Skip installing agent definitions |
| `--skip-mcp` | off | Skip MCP server registration |

## Other Commands

```bash
npx howler-agents serve          # Start MCP server (stdio transport)
npx howler-agents evolve         # Run evolution locally
npx howler-agents install        # Register with AI coding tools
```

## How It Works

This is a thin Node.js wrapper that auto-installs the Python `howler-agents-core` package (via `uv` or `pip`) on first use, then proxies all commands to the Python CLI.

## Documentation

- [Getting Started](https://jbeck018.github.io/howler-agents/)
- [Skills Reference](https://jbeck018.github.io/howler-agents/skills)
- [Architecture](https://jbeck018.github.io/howler-agents/architecture)
- [Paper Results](https://jbeck018.github.io/howler-agents/paper-results)

## License

MIT
