# howler-agents-core

Core Python library for Group-Evolving Agents (GEA) -- an evolutionary framework for open-ended self-improvement via experience sharing (arXiv:2602.04837).

This package implements the full GEA algorithm: agent pool management, performance-novelty parent selection, shared experience aggregation, group reproduction via meta-LLM, and probe-based capability characterization. It can be used standalone or as the engine behind `howler-agents-service`.

## Installation

```bash
pip install howler-agents-core
```

### Optional dependencies

For production use with Postgres (experience store + pgvector KNN) and Redis (hot cache):

```bash
pip install howler-agents-core[postgres]   # SQLAlchemy + asyncpg + pgvector
pip install howler-agents-core[redis]       # redis with hiredis
pip install howler-agents-core[all]         # both
```

## Quick Usage

```python
import asyncio
import uuid

from howler_agents import HowlerConfig, EvolutionLoop
from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector


class MyAgent(Agent):
    """Implement your agent by subclassing Agent."""

    async def run_task(self, task: dict) -> TaskResult:
        # Your task execution logic here
        return TaskResult(success=True, score=0.8, output="done")

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)


async def main() -> None:
    config = HowlerConfig(
        population_size=10,
        group_size=3,
        num_iterations=5,
        alpha=0.5,
        num_probes=20,
    )

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    selector = PerformanceNoveltySelector(alpha=config.alpha)
    reproducer = GroupReproducer(llm, experience, config)
    registry = ProbeRegistry()
    registry.register_default_probes(num_probes=config.num_probes)
    probes = ProbeEvaluator(registry)

    pool = AgentPool()
    for _ in range(config.population_size):
        pool.add(MyAgent(AgentConfig(id=str(uuid.uuid4()))))

    loop = EvolutionLoop(config, pool, selector, reproducer, experience, probes)

    tasks = [{"description": "solve a coding problem", "type": "general"}]
    results = await loop.run("my-run", tasks)
    print(f"Best score: {results['best_score']:.3f}")


asyncio.run(main())
```

## Core Modules

| Module | Description |
|---|---|
| `howler_agents.agents` | Agent base class, AgentPool, FrameworkPatch |
| `howler_agents.selection` | Performance scorer, KNN novelty estimator, combined criterion |
| `howler_agents.experience` | Evolutionary traces, shared experience pool, pluggable stores |
| `howler_agents.evolution` | Evolution directives, group reproducer, main evolution loop |
| `howler_agents.probes` | Probe task interface, evaluator, registry |
| `howler_agents.llm` | LiteLLM-backed router with role-based model dispatch |
| `howler_agents.config` | `HowlerConfig` with all GEA parameters (K, M, alpha, iterations) |

## Configuration

`HowlerConfig` accepts all GEA parameters:

| Parameter | Default | Description |
|---|---|---|
| `population_size` (K) | 10 | Total agents in population |
| `group_size` (M) | 3 | Agents per parent group |
| `num_iterations` | 5 | Evolution generations |
| `alpha` | 0.5 | Performance vs novelty weight (0=novelty, 1=performance) |
| `num_probes` | 20 | Probe tasks for capability vector |
| `task_domain` | "general" | Domain identifier (e.g., "swe-bench", "polyglot") |
| `role_models` | Claude Sonnet defaults | Per-role LLM model configuration |

## License

MIT
