"""Orchestration layer — abstracted agent execution backends.

Auto-detects the best available backend:
- claude-flow: when installed (preferred)
- local: direct LLM calls via LiteLLM (always available)
"""

from howler_agents.orchestration.detector import detect_orchestrator
from howler_agents.orchestration.interface import (
    Orchestrator,
    OrchestratorConfig,
    SpawnedAgent,
    TaskOutcome,
)

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "SpawnedAgent",
    "TaskOutcome",
    "detect_orchestrator",
]
