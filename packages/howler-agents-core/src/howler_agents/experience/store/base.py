"""Protocol for pluggable experience storage backends."""

from __future__ import annotations

from typing import Protocol

from howler_agents.experience.trace import EvolutionaryTrace


class ExperienceStore(Protocol):
    """Backend interface for storing evolutionary traces."""

    async def save(self, trace: EvolutionaryTrace) -> None: ...
    async def get_by_agent(self, agent_id: str) -> list[EvolutionaryTrace]: ...
    async def get_by_run(self, run_id: str, limit: int = 100) -> list[EvolutionaryTrace]: ...
    async def get_by_generation(self, run_id: str, generation: int) -> list[EvolutionaryTrace]: ...
    async def delete_by_run(self, run_id: str) -> int: ...
