"""Main evolution loop - orchestrates the full evolutionary cycle."""

from __future__ import annotations

import structlog

from howler_agents.agents.base import Agent
from howler_agents.agents.pool import AgentPool
from howler_agents.config import HowlerConfig
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.trace import EvolutionaryTrace
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.selection.criterion import PerformanceNoveltySelector

logger = structlog.get_logger()


class EvolutionLoop:
    """Main orchestration loop: evaluate -> select -> aggregate -> reproduce.

    Implements the Group-Evolving Agents (GEA) algorithm:
    1. Evaluate all agents on tasks + probes
    2. Score using combined performance + novelty criterion
    3. Select top agents as parents
    4. Aggregate group experience traces
    5. Reproduce via meta-LLM mutations
    """

    def __init__(
        self,
        config: HowlerConfig,
        pool: AgentPool,
        selector: PerformanceNoveltySelector,
        reproducer: GroupReproducer,
        experience: SharedExperiencePool,
        probe_evaluator: ProbeEvaluator,
    ) -> None:
        self._config = config
        self._pool = pool
        self._selector = selector
        self._reproducer = reproducer
        self._experience = experience
        self._probes = probe_evaluator

    async def step(self, run_id: str, generation: int, tasks: list[dict]) -> dict:
        """Execute one generation of evolution.

        Returns a summary dict with generation stats.
        """
        logger.info("generation_start", generation=generation, population=self._pool.size)

        # 1. Evaluate agents on tasks
        agents = self._pool.agents
        for agent in agents:
            total_score = 0.0
            for task in tasks:
                result = await agent.run_task(task)
                total_score += result.score

                trace = EvolutionaryTrace(
                    agent_id=agent.id,
                    run_id=run_id,
                    generation=generation,
                    task_description=task.get("description", ""),
                    outcome="success" if result.success else "failure",
                    score=result.score,
                    key_decisions=result.key_decisions,
                    lessons_learned=result.lessons_learned,
                    patches_applied=[p.intent for p in agent.patches],
                    parent_ids=list(agent.config.lineage),
                )
                await self._experience.submit(trace)

            agent.performance_score = total_score / len(tasks) if tasks else 0

        # 2. Evaluate probes for capability vectors
        for agent in agents:
            agent.capability_vector = await self._probes.evaluate(agent)

        # 3. Select survivors — use group-level selection when agents have group_ids
        num_survivors = max(1, self._config.population_size // 2)
        if any(a.config.group_id for a in agents):
            survivors = self._selector.select_groups(agents, num_survivors)
        else:
            survivors = self._selector.select(agents, num_survivors)

        # 4. Reproduce to fill population
        new_agents: list[Agent] = list(survivors)
        groups = self._pool.partition_groups(self._config.group_size)
        group_map = {a.id: g[0].config.group_id or "default" for g in groups for a in g}

        while len(new_agents) < self._config.population_size:
            for parent in survivors:
                if len(new_agents) >= self._config.population_size:
                    break
                group_id = group_map.get(parent.id, "default")
                patch, _directive = await self._reproducer.reproduce(
                    parent=parent,
                    run_id=run_id,
                    group_id=group_id,
                    generation=generation,
                )
                await parent.apply_patch(patch)
                # Count the patched parent as filling a slot toward population target
                new_agents.append(parent)

        # 5. Update pool
        self._pool.replace_population(new_agents)

        best = self._pool.top_k(1)[0] if self._pool.size > 0 else None
        summary = {
            "generation": generation,
            "population_size": self._pool.size,
            "best_score": best.combined_score if best else 0,
            "mean_score": sum(a.combined_score for a in self._pool.agents) / self._pool.size
            if self._pool.size > 0
            else 0,
            "best_agent_id": best.id if best else None,
        }

        logger.info("generation_complete", **summary)
        return summary

    async def run(self, run_id: str, tasks: list[dict]) -> dict:
        """Run the full evolution loop for all configured iterations."""
        results = []
        for gen in range(self._config.num_iterations):
            summary = await self.step(run_id, gen, tasks)
            results.append(summary)

        return {
            "run_id": run_id,
            "generations": results,
            "best_score": max(r["best_score"] for r in results) if results else 0,
        }
