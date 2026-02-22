"""In-memory experience store for testing."""

from __future__ import annotations

from howler_agents.experience.trace import EvolutionaryTrace


class InMemoryStore:
    """Simple in-memory experience store for testing and development."""

    def __init__(self) -> None:
        self._traces: list[EvolutionaryTrace] = []

    async def save(self, trace: EvolutionaryTrace) -> None:
        self._traces.append(trace)

    async def get_by_agent(self, agent_id: str) -> list[EvolutionaryTrace]:
        return [t for t in self._traces if t.agent_id == agent_id]

    async def get_by_run(self, run_id: str, limit: int = 100) -> list[EvolutionaryTrace]:
        traces = [t for t in self._traces if t.run_id == run_id]
        return sorted(traces, key=lambda t: t.recorded_at, reverse=True)[:limit]

    async def get_by_generation(self, run_id: str, generation: int) -> list[EvolutionaryTrace]:
        return [
            t for t in self._traces
            if t.run_id == run_id and t.generation == generation
        ]

    async def delete_by_run(self, run_id: str) -> int:
        before = len(self._traces)
        self._traces = [t for t in self._traces if t.run_id != run_id]
        return before - len(self._traces)
