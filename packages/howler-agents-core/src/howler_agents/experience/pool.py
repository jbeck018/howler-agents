"""Shared experience pool - aggregates group traces for LLM context."""

from __future__ import annotations

from howler_agents.experience.store.base import ExperienceStore
from howler_agents.experience.trace import EvolutionaryTrace


class SharedExperiencePool:
    """Aggregates evolutionary traces into LLM-readable context.

    This is the core mechanism for group evolution: traces from all agents
    in a group are aggregated and provided as context when generating
    mutations for the next generation.
    """

    def __init__(self, store: ExperienceStore) -> None:
        self._store = store

    async def submit(self, trace: EvolutionaryTrace) -> None:
        """Add a trace to the experience pool."""
        await self._store.save(trace)

    async def get_group_context(
        self, run_id: str, group_id: str, generation: int, max_traces: int = 50
    ) -> str:
        """Build LLM-readable context from group experience traces.

        Returns a formatted string summarizing lessons learned, key decisions,
        and outcomes from the group's evolutionary history.
        """
        traces = await self._store.get_by_run(run_id, limit=max_traces)

        if not traces:
            return "No prior experience available. This is the first generation."

        sections: list[str] = []
        sections.append(f"## Group Experience Summary (Generation {generation})")
        sections.append(f"Total traces: {len(traces)}\n")

        # Group by generation
        by_gen: dict[int, list[EvolutionaryTrace]] = {}
        for t in traces:
            by_gen.setdefault(t.generation, []).append(t)

        for gen in sorted(by_gen.keys()):
            gen_traces = by_gen[gen]
            scores = [t.score for t in gen_traces]
            avg_score = sum(scores) / len(scores) if scores else 0
            sections.append(f"### Generation {gen} (avg score: {avg_score:.3f})")

            for t in gen_traces[:5]:  # Limit per generation
                sections.append(f"- Task: {t.task_description}")
                sections.append(f"  Outcome: {t.outcome} (score: {t.score:.3f})")
                if t.patches_applied:
                    sections.append(f"  Patches: {', '.join(t.patches_applied)}")
                if t.lessons_learned:
                    for lesson in t.lessons_learned[:3]:
                        sections.append(f"  Lesson: {lesson}")
            sections.append("")

        return "\n".join(sections)

    async def get_agent_history(self, agent_id: str) -> list[EvolutionaryTrace]:
        """Get all traces for a specific agent."""
        return await self._store.get_by_agent(agent_id)
