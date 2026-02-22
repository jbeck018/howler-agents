"""Configuration for Howler Agents evolution runs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LLMRole(str, Enum):
    """Roles for LLM model routing."""
    ACTING = "acting"       # Agent performing tasks
    EVOLVING = "evolving"   # Meta-LLM generating mutations
    REFLECTING = "reflecting"  # Analyzing experience traces


class RoleModelConfig(BaseModel):
    """Configuration for a single LLM role."""
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class HowlerConfig(BaseModel):
    """Top-level configuration for an evolution run."""
    population_size: int = Field(default=10, alias="K", description="Total agents in population")
    group_size: int = Field(default=3, alias="M", description="Agents per group")
    num_iterations: int = Field(default=5, description="Evolution generations")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Performance vs novelty weight")
    num_probes: int = Field(default=20, description="Probe tasks for capability vector")
    task_domain: str = "general"
    task_config: dict[str, object] = Field(default_factory=dict)
    role_models: dict[LLMRole, RoleModelConfig] = Field(default_factory=lambda: {
        LLMRole.ACTING: RoleModelConfig(),
        LLMRole.EVOLVING: RoleModelConfig(),
        LLMRole.REFLECTING: RoleModelConfig(),
    })

    model_config = {"populate_by_name": True}
