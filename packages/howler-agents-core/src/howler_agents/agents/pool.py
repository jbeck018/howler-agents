"""Agent population pool management."""

from __future__ import annotations

import heapq
from typing import Sequence

from howler_agents.agents.base import Agent


class AgentPool:
    """Manages a living population of agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def add(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def remove(self, agent_id: str) -> Agent | None:
        return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    @property
    def size(self) -> int:
        return len(self._agents)

    def top_k(self, k: int, key: str = "combined_score") -> list[Agent]:
        """Return the top-K agents by the given score attribute."""
        return heapq.nlargest(k, self._agents.values(), key=lambda a: getattr(a, key, 0.0))

    def by_generation(self, generation: int) -> list[Agent]:
        return [a for a in self._agents.values() if a.config.generation == generation]

    def partition_groups(self, group_size: int) -> list[list[Agent]]:
        """Partition current population into groups of given size."""
        agents = list(self._agents.values())
        return [agents[i:i + group_size] for i in range(0, len(agents), group_size)]

    def replace_population(self, new_agents: Sequence[Agent]) -> None:
        """Replace the entire population."""
        self._agents.clear()
        for agent in new_agents:
            self._agents[agent.id] = agent
