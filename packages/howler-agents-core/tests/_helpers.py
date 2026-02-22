"""Shared test helpers - importable from test modules."""

from __future__ import annotations

import uuid
from typing import Any

from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult


class MockAgent(Agent):
    """Concrete agent for testing."""

    def __init__(self, config: AgentConfig | None = None, score: float = 0.5) -> None:
        super().__init__(config or AgentConfig())
        self._score = score

    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        return TaskResult(success=self._score > 0.3, score=self._score, output="mock")

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)


def make_agent(score: float = 0.5, generation: int = 0) -> MockAgent:
    agent = MockAgent(
        AgentConfig(id=str(uuid.uuid4()), generation=generation, lineage=[]),
        score=score,
    )
    agent.performance_score = score
    agent.capability_vector = [float(i % 2) for i in range(10)]
    return agent
