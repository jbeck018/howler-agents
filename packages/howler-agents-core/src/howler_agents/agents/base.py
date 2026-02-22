"""Abstract base agent and configuration."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration state of a single agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    parent_id: str | None = None
    group_id: str | None = None
    framework_config: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)  # full ancestor chain, newest first


@dataclass
class TaskResult:
    """Result from an agent performing a task."""

    success: bool
    score: float
    output: str = ""
    key_decisions: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)


class Agent(ABC):
    """Abstract base for an evolvable agent."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.performance_score: float = 0.0
        self.novelty_score: float = 0.0
        self.combined_score: float = 0.0
        self.capability_vector: list[float] = []
        self.patches: list[FrameworkPatch] = []

    @property
    def id(self) -> str:
        return self.config.id

    @abstractmethod
    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a task and return the result."""
        ...

    @abstractmethod
    async def apply_patch(self, patch: FrameworkPatch) -> None:
        """Apply an evolutionary mutation to this agent."""
        ...

    def clone(self, new_id: str | None = None) -> AgentConfig:
        """Create a child config from this agent."""
        new_lineage = [self.config.id] + list(self.config.lineage)
        return AgentConfig(
            id=new_id or str(uuid.uuid4()),
            generation=self.config.generation + 1,
            parent_id=self.config.id,
            group_id=self.config.group_id,
            framework_config=dict(self.config.framework_config),
            lineage=new_lineage,
        )


@dataclass
class FrameworkPatch:
    """A code/workflow mutation applied to an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    generation: int = 0
    intent: str = ""
    diff: str = ""
    category: str = "general"
