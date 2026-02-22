# Howler Agents

Open-source implementation of Group-Evolving Agents (GEA) for open-ended self-improvement via experience sharing. Runs locally with zero-config SQLite persistence, integrates with Claude Code and other AI coding tools via MCP.

## What is GEA?

Group-Evolving Agents (GEA), introduced in [arXiv:2602.04837](https://arxiv.org/abs/2602.04837), is an evolutionary framework where a *group* of AI agents improves collectively by sharing experience across lineages. Unlike prior tree-structured approaches (e.g., Darwin Godel Machine) where useful discoveries die in isolated branches, GEA agents contribute trajectories, tool discoveries, and patches to a shared experience pool. A meta-LLM analyzes the pooled experience and generates framework-level patches that propagate the best innovations to the entire next generation.

### Key Results

| Benchmark | GEA | Prior SOTA | Human-Designed SOTA |
|---|---|---|---|
| SWE-bench Verified | **71.0%** | 56.7% (DGM) | 71.8% (OpenHands + GPT-5) |
| Polyglot | **88.3%** | 68.3% (DGM) | 52.0% (Aider + GPT-5) |

GEA's best agent draws experience from 17 unique ancestors (28.3% of the population), nearly double the 9 achieved by tree-structured evolution.

### Five Core Mechanisms

1. **Performance-Novelty Parent Selection** -- Parents are chosen by jointly optimizing task success rate and KNN distance in capability space, weighted by a tunable alpha parameter. Neither signal alone is sufficient: performance-only collapses to local optima; novelty-only never converges.

2. **Shared Experience Pool** -- All agents in the parent group contribute their full evolutionary traces (code patches, tool discoveries, probe outcomes, failure modes) into a single aggregated pool. Nothing is siloed.

3. **Evolutionary Traces** -- Each agent maintains a complete history of patches applied, probe task outcomes, and an LLM-readable experience narrative. These traces are the raw material for the meta-LLM's analysis.

4. **Framework Patches** -- The meta-LLM generates concrete code, config, and workflow patches that target agent frameworks rather than model-specific prompting. This makes patches model-agnostic and transferable across the population.

5. **Novelty via Probing** -- Each agent is characterized by a binary capability vector across a fixed suite of probe tasks. This vector drives KNN novelty computation and tracks which capabilities propagate across generations.

## Quick Start

```bash
git clone https://github.com/jbeck018/howler-agents.git
cd howler-agents
pip install -e "packages/howler-agents-core[mcp]"
claude mcp add howler-agents -- howler-agents serve --transport stdio
# Restart Claude Code, then use: /howler-setup
```

No Docker, no Postgres, no Redis required. Everything runs locally with SQLite.

## Local Usage with Claude Code

This is the primary way to use howler-agents. The MCP server runs inside your coding tool and persists all state to a local `.howler-agents/evolution.db` SQLite database in your repo root.

### Install and register

```bash
# Install the core library with MCP dependencies
pip install -e "packages/howler-agents-core[mcp]"

# Register the MCP server with Claude Code
claude mcp add howler-agents -- howler-agents serve --transport stdio

# Restart Claude Code to pick up the new server
```

### Skills

Once the MCP server is registered, these slash commands are available in Claude Code:

| Skill | Description |
|---|---|
| `/howler-setup` | Initialize the local environment and verify the MCP connection |
| `/howler-evolve` | Start a GEA evolution run with configurable parameters |
| `/howler-status` | Check progress of a running or completed evolution run |
| `/howler-memory` | Browse and manage persistent collective memory |
| `/howler-sync` | Push/pull evolution data to a shared team database |

### Example session

```
You: /howler-setup
     => Initializes .howler-agents/ directory and SQLite DB

You: /howler-evolve
     => Starts evolution with 10 agents over 5 generations
     => Returns a run_id for tracking

You: /howler-status
     => Shows generation progress, best score, population stats

You: /howler-memory
     => Browse lessons and decisions accumulated across runs
```

## MCP Server

The MCP server supports three operating modes, selected automatically based on environment variables:

| Mode | Config | Description |
|---|---|---|
| **local** | No env vars (default) | Zero-config SQLite persistence. Everything runs locally. |
| **hybrid** | `HOWLER_API_URL` + `HOWLER_API_KEY` | SQLite locally + sync to a shared team database. |
| **remote** | `HOWLER_API_URL` only | Full proxy to a remote howler-agents service. |

### Transport options

```bash
# Stdio (for Claude Code, Cursor, OpenCode, Codex)
howler-agents serve --transport stdio

# SSE (for network/remote access)
howler-agents serve --transport sse --port 8765
```

### MCP Tools

The server exposes 11 tools:

| Tool | Description |
|---|---|
| `howler_evolve` | Start an evolution run with configurable population, generations, domain, and model |
| `howler_status` | Check run progress: generation, best score, population statistics |
| `howler_list_agents` | List or rank agents in the current population by combined score |
| `howler_configure` | Set LLM models for acting, evolving, and reflecting roles |
| `howler_submit_experience` | Add a task experience trace to the shared experience pool |
| `howler_get_experience` | Retrieve aggregated experience context for LLM consumption |
| `howler_memory` | Hive-mind memory CRUD: store, retrieve, search, list, delete |
| `howler_history` | Browse persistent evolution history: runs, agents, traces, stats |
| `howler_hivemind` | Manage collective intelligence: status, seed from runs, export/import |
| `howler_sync_push` | Push completed runs to the team database (hybrid mode) |
| `howler_sync_pull` | Pull shared hive-mind memory from the team database (hybrid mode) |

## Hive-Mind

The hive-mind is a persistent collective intelligence layer built on top of the evolution engine. It accumulates knowledge across sessions and runs, stored in the local SQLite database.

**Collective Memory** -- Namespace-organized key-value store with text search and access tracking. Memory entries persist across process restarts and Claude Code sessions.

**Consensus Engine** -- Proposals are created automatically when lessons appear in three or more evolution traces. Proposals with confidence >= 0.6 are promoted to permanent collective memory under the `consensus` namespace.

**Automatic Seeding** -- When an evolution run completes, the hive-mind coordinator extracts the top-scoring lessons and decisions from traces and stores them. High-frequency patterns are submitted as consensus proposals.

**Team Sync** -- In hybrid mode, push local discoveries to a shared database and pull lessons from other developers' agents. Knowledge compounds across the team.

## Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           AI Coding Tools                │
                          │   Claude Code  |  Cursor  |  OpenCode   │
                          └─────────────────────┬────────────────────┘
                                                │
                                     MCP (stdio or SSE)
                                                │
                          ┌─────────────────────▼────────────────────┐
                          │           MCP Server (mcp_server.py)     │
                          │   Tools + Resources + Lifespan mgmt      │
                          └──────┬──────────────────┬────────────────┘
                                 │                  │
                   ┌─────────────▼──────┐  ┌───────▼─────────────────┐
                   │   LocalRunner       │  │    Persistence Layer    │
                   │   Evolution loops   │  │                         │
                   │   Agent management  │  │  SQLite (.howler-agents │
                   │   Experience pool   │  │  /evolution.db)         │
                   └─────────┬──────────┘  │                         │
                             │              │  Runs, agents, traces,  │
                   ┌─────────▼──────────┐  │  memory, consensus      │
                   │   Core Modules      │  └───────┬─────────────────┘
                   │                     │          │
                   │  AgentPool          │  ┌───────▼─────────────────┐
                   │  PerformanceNovelty │  │    Hive-Mind Layer      │
                   │    Selector         │  │                         │
                   │  SharedExperience   │  │  CollectiveMemory       │
                   │    Pool             │  │  ConsensusEngine        │
                   │  GroupReproducer    │  │  HiveMindCoordinator    │
                   │  ProbeEvaluator     │  │                         │
                   │  LLMRouter          │  │  Optional: SyncClient   │
                   └─────────────────────┘  │  (team DB push/pull)    │
                                            └─────────────────────────┘
```

## Monorepo Structure

```
howler-agents/
├── packages/
│   ├── howler-agents-core/       Python core library + MCP server + CLI
│   ├── howler-agents-service/    FastAPI + gRPC service layer
│   ├── howler-agents-ts/         TypeScript SDK (@howler-agents/sdk)
│   ├── howler-agents-ui/         React + Vite dashboard
│   └── howler-agents-docs/       Documentation site
├── examples/
│   ├── basic-python/             Minimal evolution example
│   ├── basic-ts/                 TypeScript SDK example
│   ├── swe-bench/                SWE-bench benchmark reproduction
│   └── polyglot/                 Polyglot benchmark reproduction
├── proto/                        Protobuf definitions (source of truth)
├── .howler-agents/               Local SQLite persistence (auto-created, gitignored)
└── .claude/skills/               Claude Code skill definitions
```

### Packages

| Package | Language | Description |
|---|---|---|
| `howler-agents-core` | Python | Core GEA algorithm, MCP server, CLI, local runner, hive-mind, SQLite persistence, LLM routing via LiteLLM |
| `howler-agents-service` | Python | Stateless FastAPI + gRPC service layer; meta-LLM reflection runs server-side |
| `howler-agents-ts` | TypeScript | `@howler-agents/sdk` -- thin gRPC/Connect client for Node.js and browsers |
| `howler-agents-ui` | TypeScript | React + Vite dashboard for monitoring evolution runs in real time |
| `howler-agents-docs` | TypeScript | Documentation site |

## Build Commands

```bash
# Install Python dependencies (from repo root)
uv sync --all-extras --all-packages

# Install Node dependencies
pnpm install

# Run Python tests (52 tests)
uv run pytest -v

# Run TypeScript SDK tests
cd packages/howler-agents-ts && pnpm exec vitest run

# Build the UI
cd packages/howler-agents-ui && npx vite build

# CLI usage
howler-agents serve --transport stdio     # Start MCP server
howler-agents evolve --domain coding      # Run evolution from CLI
howler-agents status                      # Show run status
howler-agents configure --show            # Show LLM config
```

## Documentation

- [Architecture deep dive](./GEA_Architecture.md)
- [GitHub repository](https://github.com/jbeck018/howler-agents)

## License

MIT
