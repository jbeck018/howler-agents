"""Evolutionary trace - a record of an agent's history."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvolutionaryTrace:
    """A single record in an agent's evolutionary history."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    run_id: str = ""
    group_id: str = ""
    generation: int = 0
    task_description: str = ""
    outcome: str = ""
    score: float = 0.0
    key_decisions: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    patches_applied: list[str] = field(
        default_factory=list
    )  # patch intents applied this generation
    parent_ids: list[str] = field(default_factory=list)  # full lineage chain
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
