"""Orchestrator protocol — abstraction over agent execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorConfig:
    """Configuration for an orchestrator backend."""

    backend: str = "auto"  # auto | local | claude-flow
    claude_flow_available: bool = False
    max_concurrent_agents: int = 8
    timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpawnedAgent:
    """Reference to an agent spawned by an orchestrator."""

    agent_id: str
    backend: str  # which orchestrator spawned it
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutcome:
    """Result from executing a task through an orchestrator."""

    agent_id: str
    task_id: str
    success: bool
    score: float
    output: str = ""
    key_decisions: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Orchestrator(ABC):
    """Abstract base for agent execution backends.

    Implementations handle how evolved agent prompts become
    working agents that execute real tasks and report outcomes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier: 'local', 'claude-flow', etc."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the backend (connect, verify availability, etc)."""
        ...

    @abstractmethod
    async def spawn_agent(
        self,
        prompt: str,
        task_domain: str,
        agent_config: dict[str, Any] | None = None,
    ) -> SpawnedAgent:
        """Spawn a new agent with the given evolved prompt."""
        ...

    @abstractmethod
    async def execute_task(
        self,
        agent: SpawnedAgent,
        task: dict[str, Any],
    ) -> TaskOutcome:
        """Execute a task using a spawned agent. Returns the outcome."""
        ...

    @abstractmethod
    async def terminate_agent(self, agent: SpawnedAgent) -> None:
        """Terminate/clean up a spawned agent."""
        ...

    @abstractmethod
    async def get_available_agents(self) -> list[SpawnedAgent]:
        """List currently active agents managed by this orchestrator."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the orchestrator and all its agents."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check if the backend is operational."""
        return {"backend": self.name, "status": "ok"}
