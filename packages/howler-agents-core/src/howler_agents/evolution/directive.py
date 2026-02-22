"""Evolution directive - LLM-generated mutation intent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvolutionDirective:
    """An LLM-generated intent for how to mutate an agent."""

    intent: str = ""
    target_areas: list[str] = field(default_factory=list)
    strategy: str = "incremental"  # "incremental", "exploratory", "targeted"
    confidence: float = 0.5
    reasoning: str = ""
