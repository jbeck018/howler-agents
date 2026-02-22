"""Auto-detect the best available orchestrator backend."""

from __future__ import annotations

import structlog

from howler_agents.config import HowlerConfig
from howler_agents.orchestration.interface import Orchestrator

logger = structlog.get_logger()


def detect_orchestrator(
    howler_config: HowlerConfig,
    preferred: str = "auto",
) -> Orchestrator:
    """Return the best available orchestrator.

    Resolution order for 'auto':
    1. claude-flow (if installed) — richest orchestration
    2. local (always available) — direct LLM calls

    Args:
        howler_config: Evolution run configuration.
        preferred: 'auto', 'local', or 'claude-flow'.

    Returns:
        An Orchestrator instance ready to be initialized.
    """
    if preferred == "local":
        from howler_agents.orchestration.local import LocalOrchestrator

        logger.info("orchestrator_selected", backend="local", reason="explicit")
        return LocalOrchestrator(howler_config)

    if preferred == "claude-flow":
        from howler_agents.orchestration.claude_flow import (
            ClaudeFlowOrchestrator,
            _check_claude_flow_available,
        )

        if _check_claude_flow_available():
            logger.info("orchestrator_selected", backend="claude-flow", reason="explicit")
            return ClaudeFlowOrchestrator()
        else:
            logger.warning(
                "orchestrator_fallback",
                requested="claude-flow",
                actual="local",
                reason="not_installed",
            )
            from howler_agents.orchestration.local import LocalOrchestrator

            return LocalOrchestrator(howler_config)

    # Auto mode: prefer claude-flow if available
    if preferred == "auto":
        from howler_agents.orchestration.claude_flow import _check_claude_flow_available

        if _check_claude_flow_available():
            from howler_agents.orchestration.claude_flow import ClaudeFlowOrchestrator

            logger.info("orchestrator_selected", backend="claude-flow", reason="auto_detected")
            return ClaudeFlowOrchestrator()
        else:
            from howler_agents.orchestration.local import LocalOrchestrator

            logger.info("orchestrator_selected", backend="local", reason="auto_fallback")
            return LocalOrchestrator(howler_config)

    # Unknown backend
    logger.warning("orchestrator_unknown", backend=preferred)
    from howler_agents.orchestration.local import LocalOrchestrator

    return LocalOrchestrator(howler_config)
