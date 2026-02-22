"""Basic example: Run a minimal evolution with mock agents."""

import asyncio
import uuid

from howler_agents import HowlerConfig
from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector


class SimpleAgent(Agent):
    """A simple agent for demonstration."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._base_skill = 0.5

    async def run_task(self, task: dict) -> TaskResult:
        import random
        score = min(1.0, self._base_skill + random.gauss(0, 0.1))
        return TaskResult(
            success=score > 0.4,
            score=max(0, score),
            output="completed",
            key_decisions=["used default strategy"],
            lessons_learned=["baseline approach"],
        )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)
        self._base_skill = min(1.0, self._base_skill + 0.05)


async def main() -> None:
    config = HowlerConfig(population_size=6, group_size=3, num_iterations=3)

    # Set up components
    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    selector = PerformanceNoveltySelector(alpha=config.alpha)
    reproducer = GroupReproducer(llm, experience, config)

    registry = ProbeRegistry()
    registry.register_default_probes(num_probes=5)
    probes = ProbeEvaluator(registry)

    # Create initial population
    pool = AgentPool()
    for _ in range(config.population_size):
        pool.add(SimpleAgent(AgentConfig(id=str(uuid.uuid4()))))

    loop = EvolutionLoop(config, pool, selector, reproducer, experience, probes)

    print(f"Starting evolution: K={config.population_size}, M={config.group_size}")
    print(f"Iterations: {config.num_iterations}, Alpha: {config.alpha}\n")

    tasks = [{"description": "solve a coding problem", "type": "general"}]
    results = await loop.run("demo-run", tasks)

    print(f"\nEvolution complete!")
    print(f"Best score: {results['best_score']:.3f}")
    for gen in results["generations"]:
        print(f"  Gen {gen['generation']}: best={gen['best_score']:.3f}, mean={gen['mean_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
