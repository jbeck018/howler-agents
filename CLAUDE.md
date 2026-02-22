# Claude Code Configuration — Howler Agents

## Project Overview

- GEA (Group-Evolving Agents) implementation from arXiv:2602.04837
- Python + TypeScript monorepo: `uv` for Python, `pnpm` for Node
- Core library + FastAPI service + React dashboard + TypeScript SDK

## Monorepo Structure

```
packages/howler-agents-core/    # Python core (pip: howler-agents-core)
packages/howler-agents-service/ # FastAPI + gRPC service
packages/howler-agents-ui/      # Vite + React dashboard (port 3000)
packages/howler-agents-ts/      # TypeScript SDK (@howler-agents/sdk)
packages/howler-agents-docs/    # Documentation site
```

## Build & Test

```bash
uv sync --all-extras --all-packages    # Install Python deps
pnpm install                            # Install Node deps
uv run pytest -v                        # 52 Python tests (run from root)
cd packages/howler-agents-ts && pnpm exec vitest run  # TS tests
cd packages/howler-agents-ui && npx vite build         # Build UI
```

- ALWAYS run tests after making code changes
- ALWAYS verify build succeeds before committing

## MCP Server

- The howler-agents MCP server exposes 11 tools for evolution, memory, and hive-mind management
- Runs locally via SQLite (zero-config), optionally syncs to remote Postgres
- Entry point: `howler-agents serve --transport stdio`
- Registered in `.mcp.json`

### MCP Tools

| Tool | Purpose |
|------|---------|
| `howler_evolve` | Start an evolution run |
| `howler_status` | Check run progress |
| `howler_list_agents` | List/rank agents in a run |
| `howler_submit_experience` | Submit task experience trace |
| `howler_get_experience` | Retrieve collective experience |
| `howler_configure` | Set model configuration |
| `howler_memory` | Access hive-mind memory |
| `howler_history` | Browse past evolution runs |
| `howler_hivemind` | Hive-mind consensus operations |
| `howler_sync_push` | Push local data to remote |
| `howler_sync_pull` | Pull remote data to local |

## Skills (Slash Commands)

| Skill | Purpose |
|-------|---------|
| `/howler-setup` | Initialize local environment (install, register MCP, create `.howler-agents/`) |
| `/howler-evolve` | Start GEA evolution with hive-mind team |
| `/howler-status` | Check run progress and agent rankings |
| `/howler-memory` | Browse collective memory and lessons learned |
| `/howler-sync` | Push/pull team sync between local SQLite and remote Postgres |

## Agent Definitions

| Agent | File | Role |
|-------|------|------|
| Coordinator | `.claude/agents/howler/coordinator.md` | Monitors evolution, coordinates team |
| Evaluator | `.claude/agents/howler/evaluator.md` | Post-run quantitative analysis |
| Reproducer | `.claude/agents/howler/reproducer.md` | Extracts lessons, seeds hive-mind |
| Actor | `.claude/agents/howler/actor.md` | External task execution for coding-domain runs |

## Behavioral Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they are absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files unless explicitly requested
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files
- ALWAYS run tests after making code changes

## File Organization

- `/packages/howler-agents-core/src/` for Python source code
- `/packages/howler-agents-core/tests/` for Python tests
- `/packages/howler-agents-service/` for FastAPI + gRPC service code
- `/packages/howler-agents-ts/src/` for TypeScript SDK source
- `/packages/howler-agents-ui/src/` for React dashboard source
- `/docs/` for standalone documentation
- `.claude/skills/` for skill definitions
- `.claude/agents/` for agent definitions

## Code Style

- Python: ruff for linting/formatting, mypy for type checking
- TypeScript: eslint + prettier, strict mode
- Follow Domain-Driven Design with bounded contexts
- Keep files under 500 lines
- Use typed interfaces for all public APIs
- Input validation at system boundaries

## Key Dependencies

- **LiteLLM**: BYOK LLM routing (replaces direct Anthropic/OpenAI clients)
- **aiosqlite**: Local persistent storage
- **pydantic v2**: Data validation and settings
- **structlog**: Structured logging
- **mcp**: Model Context Protocol server SDK
- **scikit-learn**: KNN novelty computation
- **numpy**: Capability vector operations

## Common Issues & Fixes

- **uv workspace deps**: Must add `[tool.uv.sources] howler-agents-core = { workspace = true }` in both root and service `pyproject.toml`
- **pydantic-settings extra env vars**: Use `"extra": "ignore"` in `model_config` to skip unknown env vars from `.env`
- **pytest conftest collision**: Two test dirs with `__init__.py` collide; remove `__init__.py`, put shared helpers in `_helpers.py` with `sys.path.insert` in conftest
- **TanStack Start/vinxi**: Incompatible with TanStack Router v1.161+; use plain Vite SPA with `createRoute()` (not `createFileRoute()`)
- **Badge variant types**: Must include `"secondary"` in Record type unions alongside `"default"`, `"success"`, `"warning"`, `"destructive"`

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries
- Always sanitize file paths to prevent directory traversal
