"""Claude-flow orchestrator — executes agents via claude-flow MCP tools.

Requires claude-flow MCP server to be registered and running. Uses the
mcp__claude-flow__* tool namespace for swarm init, agent spawning, task
orchestration, and memory operations.

This orchestrator is the preferred backend when claude-flow is detected.
It provides richer coordination, adaptive topology, and hive-mind memory.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from typing import Any

import structlog

from howler_agents.orchestration.interface import (
    Orchestrator,
    OrchestratorConfig,
    SpawnedAgent,
    TaskOutcome,
)

logger = structlog.get_logger()


class ClaudeFlowOrchestrator(Orchestrator):
    """Execute agents via claude-flow's MCP tools.

    Integration model:
    - Uses claude-flow for agent spawning and task execution
    - Feeds outcomes back to howler-agents experience pool
    - Bridges memory between claude-flow and howler hive-minds

    Since MCP tools are invoked from Claude Code (the host), this
    orchestrator prepares tool call specifications that the MCP
    server can relay. For direct programmatic use (outside Claude Code),
    it falls back to CLI subprocess calls.
    """

    def __init__(self, orch_config: OrchestratorConfig | None = None) -> None:
        self._config = orch_config or OrchestratorConfig(backend="claude-flow")
        self._agents: dict[str, SpawnedAgent] = {}
        self._swarm_id: str | None = None
        self._available = False

    @property
    def name(self) -> str:
        return "claude-flow"

    async def initialize(self) -> None:
        """Verify claude-flow is available and initialize swarm."""
        self._available = _check_claude_flow_available()
        if not self._available:
            raise RuntimeError(
                "claude-flow not found. Install with: npm install -g claude-flow@alpha"
            )
        self._swarm_id = f"howler-{uuid.uuid4().hex[:8]}"
        logger.info("orchestrator_initialized", backend="claude-flow", swarm_id=self._swarm_id)

    async def spawn_agent(
        self,
        prompt: str,
        task_domain: str,
        agent_config: dict[str, Any] | None = None,
    ) -> SpawnedAgent:
        agent_id = f"howler-agent-{uuid.uuid4().hex[:8]}"
        agent = SpawnedAgent(
            agent_id=agent_id,
            backend="claude-flow",
            prompt=prompt,
            metadata={
                "task_domain": task_domain,
                "swarm_id": self._swarm_id,
                **(agent_config or {}),
            },
        )
        self._agents[agent_id] = agent

        # Store the agent prompt in claude-flow memory for retrieval by spawned agents
        _cf_cli(
            "memory",
            "store",
            "--key",
            f"agent-prompt-{agent_id}",
            "--value",
            prompt,
            "--namespace",
            "howler-agents",
        )

        logger.debug("agent_spawned", backend="claude-flow", agent_id=agent_id)
        return agent

    async def execute_task(
        self,
        agent: SpawnedAgent,
        task: dict[str, Any],
    ) -> TaskOutcome:
        description = task.get("description", str(task))
        task_id = task.get("id", str(uuid.uuid4()))
        start = time.monotonic()

        # Execute via claude-flow task orchestration
        try:
            result = _cf_cli(
                "task",
                "create",
                "--prompt",
                f"{agent.prompt}\n\nTask: {description}",
                "--id",
                task_id,
            )

            elapsed = int((time.monotonic() - start) * 1000)

            # Parse result — claude-flow returns JSON
            try:
                parsed = json.loads(result)
                return TaskOutcome(
                    agent_id=agent.agent_id,
                    task_id=task_id,
                    success=parsed.get("success", False),
                    score=float(parsed.get("score", 0.5)),
                    output=parsed.get("output", result),
                    key_decisions=parsed.get("key_decisions", []),
                    lessons_learned=parsed.get("lessons_learned", []),
                    duration_ms=elapsed,
                    metadata={"backend": "claude-flow"},
                )
            except (json.JSONDecodeError, TypeError):
                # Raw text output — score by length/content heuristic
                return TaskOutcome(
                    agent_id=agent.agent_id,
                    task_id=task_id,
                    success=len(result.strip()) > 10,
                    score=0.5,
                    output=result,
                    duration_ms=elapsed,
                    metadata={"backend": "claude-flow", "raw_output": True},
                )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning("cf_task_error", agent_id=agent.agent_id, error=str(exc))
            return TaskOutcome(
                agent_id=agent.agent_id,
                task_id=task_id,
                success=False,
                score=0.0,
                output=f"Error: {exc}",
                duration_ms=elapsed,
            )

    async def terminate_agent(self, agent: SpawnedAgent) -> None:
        self._agents.pop(agent.agent_id, None)
        # Clean up memory
        _cf_cli(
            "memory",
            "delete",
            "--key",
            f"agent-prompt-{agent.agent_id}",
            "--namespace",
            "howler-agents",
        )

    async def get_available_agents(self) -> list[SpawnedAgent]:
        return list(self._agents.values())

    async def shutdown(self) -> None:
        self._agents.clear()
        logger.info("orchestrator_shutdown", backend="claude-flow", swarm_id=self._swarm_id)

    async def health_check(self) -> dict[str, Any]:
        available = _check_claude_flow_available()
        return {
            "backend": "claude-flow",
            "status": "ok" if available else "unavailable",
            "swarm_id": self._swarm_id,
            "agents": len(self._agents),
        }

    # ------------------------------------------------------------------ #
    # Claude-flow memory bridge                                           #
    # ------------------------------------------------------------------ #

    async def store_memory(self, key: str, value: str, namespace: str = "howler-agents") -> None:
        """Store a value in claude-flow's memory system."""
        _cf_cli("memory", "store", "--key", key, "--value", value, "--namespace", namespace)

    async def search_memory(self, query: str, namespace: str = "howler-agents") -> str:
        """Search claude-flow's memory."""
        return _cf_cli("memory", "search", "--query", query, "--namespace", namespace)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _check_claude_flow_available() -> bool:
    """Check if claude-flow CLI is installed and accessible."""
    import shutil

    if shutil.which("claude-flow"):
        return True
    if shutil.which("npx"):
        try:
            result = subprocess.run(
                ["npx", "claude-flow", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    return False


def _cf_cli(*args: str) -> str:
    """Run a claude-flow CLI command and return stdout."""
    import shutil

    cmd: list[str]
    if shutil.which("claude-flow"):
        cmd = ["claude-flow", *args]
    else:
        cmd = ["npx", "claude-flow", *args]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("cf_cli_error", args=args, stderr=result.stderr[:500])
            return result.stderr or ""
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("cf_cli_timeout", args=args)
        return ""
    except FileNotFoundError:
        logger.warning("cf_cli_not_found")
        return ""
