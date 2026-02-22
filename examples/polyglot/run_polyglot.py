"""Run GEA on Polyglot benchmark to reproduce the paper's 88.3% result.

Configuration follows arXiv:2602.04837 Table 3:
- K=50 (population size)
- M=5 (group size)
- alpha=0.6 (slightly more exploration than SWE-bench)
- 40 iterations (single-file edits converge faster)
- 25 probe tasks (binary capability vector dimensionality)

Polyglot tasks involve single-file edits across multiple programming
languages. GEA adapts by producing larger, more concentrated patches
and converging in fewer iterations than multi-file benchmarks.
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


class PolyglotAgent(Agent):
    """Agent implementation for Polyglot benchmark tasks.

    In a real setup, this would integrate with a code editing harness
    capable of handling multiple programming languages (Python, Java,
    JavaScript, TypeScript, Go, Rust, etc.) and running language-specific
    test suites for validation.
    """

    async def run_task(self, task: dict) -> TaskResult:
        # TODO: Replace with actual Polyglot task execution.
        # The task dict should contain:
        #   - "instance_id": Polyglot instance identifier
        #   - "language": programming language of the task
        #   - "source_file": path to the file to edit
        #   - "problem_statement": description of the required change
        #   - "test_command": command to validate the solution
        raise NotImplementedError(
            "Implement Polyglot task execution by integrating with your "
            "preferred code editing agent."
        )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)


async def main() -> None:
    # Paper parameters for Polyglot (Table 3, arXiv:2602.04837)
    config = HowlerConfig(
        population_size=50,      # K=50: population size
        group_size=5,            # M=5: agents per parent group
        num_iterations=40,       # 40 iterations (faster convergence for single-file)
        alpha=0.6,               # Slightly more exploration than SWE-bench
        num_probes=25,           # 25 probe tasks for capability characterization
        task_domain="polyglot",
        task_config={
            "dataset": "polyglot-benchmark",
            "languages": [
                "python", "java", "javascript", "typescript",
                "go", "rust", "ruby", "cpp",
            ],
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
        pool.add(PolyglotAgent(AgentConfig(id=str(uuid.uuid4()))))

    loop = EvolutionLoop(config, pool, selector, reproducer, experience, probes)

    print("Polyglot Benchmark -- GEA Evolution")
    print(f"  Population (K):  {config.population_size}")
    print(f"  Group size (M):  {config.group_size}")
    print(f"  Alpha:           {config.alpha}")
    print(f"  Iterations:      {config.num_iterations}")
    print(f"  Probe tasks:     {config.num_probes}")
    print()

    # Load Polyglot tasks. In production, these come from the dataset.
    tasks = [
        {
            "description": "Polyglot coding task",
            "type": "polyglot",
            "instance_id": "placeholder",
        }
    ]

    results = await loop.run("polyglot-run", tasks)

    print(f"\nEvolution complete.")
    print(f"Best score: {results['best_score']:.3f}")
    print(f"Target:     0.883 (88.3% -- paper result)")
    print()
    for gen in results["generations"]:
        print(
            f"  Gen {gen['generation']:3d}: "
            f"best={gen['best_score']:.3f}, "
            f"mean={gen['mean_score']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
