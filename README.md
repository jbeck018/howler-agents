# Howler Agents

Open-source implementation of Group-Evolving Agents (GEA) for open-ended self-improvement via experience sharing.

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

### Prerequisites

- Python 3.12+
- Node.js 22+
- Docker (for Postgres + Redis)

### Setup

```bash
git clone https://github.com/howler-agents/howler-agents.git
cd howler-agents

make setup && make docker-up && make dev
```

Copy `.env.example` to `.env` and fill in your LLM API keys before running.

## Monorepo Structure

```
howler-agents/
├── packages/
│   ├── howler-agents-core/       Python core library
│   ├── howler-agents-service/    gRPC + REST service layer
│   ├── howler-agents-ts/         TypeScript SDK
│   ├── howler-agents-ui/         React dashboard
│   └── howler-agents-docs/       Documentation site
├── examples/
│   ├── basic-python/             Minimal evolution example
│   ├── basic-ts/                 TypeScript SDK example
│   ├── swe-bench/                SWE-bench benchmark reproduction
│   └── polyglot/                 Polyglot benchmark reproduction
├── proto/                        Protobuf definitions (source of truth)
├── migrations/                   SQL migration files
├── deploy/docker/                Docker Compose configuration
├── scripts/                      Build and generation scripts
└── Makefile                      Top-level build commands
```

### Packages

| Package | Language | Description |
|---|---|---|
| `howler-agents-core` | Python | Core GEA algorithm: agent pool, selection, experience sharing, evolution loop, LLM routing via LiteLLM |
| `howler-agents-service` | Python | Stateless gRPC + REST service layer; meta-LLM reflection runs server-side |
| `howler-agents-ts` | TypeScript | `@howler-agents/sdk` -- thin gRPC/Connect client for Node.js and browsers |
| `howler-agents-ui` | TypeScript | React + Vite dashboard for monitoring evolution runs in real time |
| `howler-agents-docs` | TypeScript | Documentation site |

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      Language SDKs                            │
│          Python  |  TypeScript  |  (future: Go, Rust, ...)    │
└──────────────────────────┬────────────────────────────────────┘
                           │
                 gRPC (primary) / REST (fallback)
                           │
┌──────────────────────────▼────────────────────────────────────┐
│                   howler-agents-service                        │
│        FastAPI + gRPC gateway -- stateless, scalable          │
│        Meta-LLM reflection runs here; API keys stay server    │
└───────────┬──────────────────────────────┬────────────────────┘
            │                              │
┌───────────▼───────────┐    ┌─────────────▼────────────────────┐
│  howler-agents-core   │    │       Experience Store            │
│  Python library       │    │   Postgres + pgvector (durable)   │
│  (pip installable)    │    │   Redis (hot cache)               │
└───────────┬───────────┘    └──────────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────────────┐
│                      Core Modules                             │
│                                                               │
│  AgentPool           -- manages the living population         │
│  PerformanceNoveltySeletor -- balanced parent selection        │
│  SharedExperiencePool -- aggregated group traces              │
│  GroupReproducer      -- parent group -> child group via LLM  │
│  ProbeEvaluator       -- builds binary capability vectors     │
│  LLMRouter            -- role-based model dispatch            │
└───────────────────────────────────────────────────────────────┘
```

## Documentation

- [Architecture deep dive](./GEA_Architecture.md)
- [Documentation site](https://howler-agents.github.io/howler-agents)
- [API reference](https://howler-agents.github.io/howler-agents/api)

## Build Commands

```bash
make setup          # Install Python (uv) and Node (pnpm) dependencies
make dev            # Start service, UI, and docs in parallel
make test           # Run all Python and Node tests
make build          # Build all packages
make lint           # Lint Python (ruff, mypy) and Node (eslint, tsc)
make proto          # Regenerate protobuf stubs for all SDKs
make docker-up      # Start Postgres + Redis via Docker Compose
make docker-down    # Stop Docker services
make clean          # Remove build artifacts and caches
```

## License

MIT
