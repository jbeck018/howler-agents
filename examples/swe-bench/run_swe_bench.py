"""Run GEA on SWE-bench Verified to reproduce the paper's 71.0% result.

Configuration follows arXiv:2602.04837 Table 3:
- K=50 (population size)
- M=5 (group size)
- alpha=0.7 (performance-weighted -- SWE-bench benefits from exploitation)
- 60 iterations (multi-file coordination requires more generations)
- 30 probe tasks (binary capability vector dimensionality)

SWE-bench tasks involve multi-file edits across real Python repositories.
GEA adapts to this complexity by producing smaller, distributed patches
across more iterations compared to single-file benchmarks.
"""

import asyncio
import uuid

from howler_agents import HowlerConfig
from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.config import LLMRole, RoleModelConfig
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector


class SWEBenchAgent(Agent):
    """Agent implementation for SWE-bench Verified tasks.

    In a real setup, this would integrate with a SWE-bench harness
    (e.g., OpenHands, SWE-agent) to execute repository-level edits
    and run the test suite for validation.
    """

    async def run_task(self, task: dict) -> TaskResult:
        # TODO: Replace with actual SWE-bench task execution.
        # The task dict should contain:
        #   - "instance_id": SWE-bench instance identifier
        #   - "repo": repository URL
        #   - "base_commit": commit to check out
        #   - "problem_statement": issue description
        #   - "test_patch": patch to apply for validation
        raise NotImplementedError(
            "Implement SWE-bench task execution by integrating with your "
            "preferred agent harness (OpenHands, SWE-agent, etc.)."
        )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)


async def main() -> None:
    # Paper parameters for SWE-bench Verified (Table 3, arXiv:2602.04837)
    config = HowlerConfig(
        population_size=50,      # K=50: population size
        group_size=5,            # M=5: agents per parent group
        num_iterations=60,       # 60 iterations for multi-file tasks
        alpha=0.7,               # Higher alpha favors performance (exploitation)
        num_probes=30,           # 30 probe tasks for capability characterization
        task_domain="swe-bench",
        task_config={
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
        },
        # Paper uses Claude Haiku for acting, Claude Sonnet for evolving,
        # and GPT-o1 for reflection. Adjust models to your API access.
        role_models={
            LLMRole.ACTING: RoleModelConfig(model="claude-haiku-4-5-20250514"),
            LLMRole.EVOLVING: RoleModelConfig(model="claude-sonnet-4-20250514"),
            LLMRole.REFLECTING: RoleModelConfig(model="o1"),
        },
    )

    # Set up the evolution components
    store = InMemoryStore()  # Use PostgresStore for durable runs
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    selector = PerformanceNoveltySelector(alpha=config.alpha)
    reproducer = GroupReproducer(llm, experience, config)

    registry = ProbeRegistry()
    registry.register_default_probes(num_probes=config.num_probes)
    probes = ProbeEvaluator(registry)

    # Create initial population
    pool = AgentPool()
    for _ in range(config.population_size):
        pool.add(SWEBenchAgent(AgentConfig(id=str(uuid.uuid4()))))

    loop = EvolutionLoop(config, pool, selector, reproducer, experience, probes)

    print("SWE-bench Verified -- GEA Evolution")
    print(f"  Population (K):  {config.population_size}")
    print(f"  Group size (M):  {config.group_size}")
    print(f"  Alpha:           {config.alpha}")
    print(f"  Iterations:      {config.num_iterations}")
    print(f"  Probe tasks:     {config.num_probes}")
    print()

    # Load SWE-bench tasks. In production, these come from the dataset.
    tasks = [
        {
            "description": "SWE-bench task",
            "type": "swe-bench",
            "instance_id": "placeholder",
        }
    ]

    results = await loop.run("swe-bench-run", tasks)

    print(f"\nEvolution complete.")
    print(f"Best score: {results['best_score']:.3f}")
    print(f"Target:     0.710 (71.0% -- paper result)")
    print()
    for gen in results["generations"]:
        print(
            f"  Gen {gen['generation']:3d}: "
            f"best={gen['best_score']:.3f}, "
            f"mean={gen['mean_score']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
