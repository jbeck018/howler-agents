"""Local orchestrator — executes agents via direct LLM calls through LiteLLM."""

import time
import uuid
from typing import Any

import structlog

from howler_agents.config import HowlerConfig, LLMRole
from howler_agents.llm.router import LLMRouter
from howler_agents.orchestration.interface import (
    Orchestrator,
    OrchestratorConfig,
    SpawnedAgent,
    TaskOutcome,
)

logger = structlog.get_logger()


class LocalOrchestrator(Orchestrator):
    """Execute agents locally via LiteLLM.

    This is the zero-dependency default: no claude-flow, no external
    services. Each 'spawned agent' is just a prompt + LLM router call.
    """

    def __init__(
        self, howler_config: HowlerConfig, orch_config: OrchestratorConfig | None = None
    ) -> None:
        self._config = howler_config
        self._orch_config = orch_config or OrchestratorConfig(backend="local")
        self._llm: LLMRouter | None = None
        self._agents: dict[str, SpawnedAgent] = {}

    @property
    def name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        self._llm = LLMRouter(self._config)
        logger.info("orchestrator_initialized", backend="local")

    async def spawn_agent(
        self,
        prompt: str,
        task_domain: str,
        agent_config: dict[str, Any] | None = None,
    ) -> SpawnedAgent:
        agent_id = str(uuid.uuid4())
        agent = SpawnedAgent(
            agent_id=agent_id,
            backend="local",
            prompt=prompt,
            metadata={"task_domain": task_domain, **(agent_config or {})},
        )
        self._agents[agent_id] = agent
        logger.debug("agent_spawned", backend="local", agent_id=agent_id)
        return agent

    async def execute_task(
        self,
        agent: SpawnedAgent,
        task: dict[str, Any],
    ) -> TaskOutcome:
        if self._llm is None:
            raise RuntimeError("LocalOrchestrator not initialized. Call initialize() first.")

        description = task.get("description", str(task))
        task_id = task.get("id", str(uuid.uuid4()))
        start = time.monotonic()

        system_prompt = (
            f"{agent.prompt}\n\n"
            f"Task domain: {agent.metadata.get('task_domain', 'general')}\n"
            "After your answer, on a new line write:\n"
            "DECISION: <your key decision>\n"
            "LESSON: <what you learned>\n"
            "SCORE: <0.0-1.0 self-assessment>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ]

        try:
            response = await self._llm.complete(role=LLMRole.ACTING, messages=messages)
            lines = response.strip().splitlines()

            decision = ""
            lesson = ""
            score = 0.5
            answer_lines = []

            for line in lines:
                upper = line.upper()
                if upper.startswith("DECISION:"):
                    decision = line.split(":", 1)[1].strip()
                elif upper.startswith("LESSON:"):
                    lesson = line.split(":", 1)[1].strip()
                elif upper.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        score = 0.5
                else:
                    answer_lines.append(line)

            elapsed = int((time.monotonic() - start) * 1000)
            answer = "\n".join(answer_lines).strip()
            expected = task.get("expected", "")
            success = expected.lower() in answer.lower() if expected else True

            return TaskOutcome(
                agent_id=agent.agent_id,
                task_id=task_id,
                success=success,
                score=score,
                output=answer,
                key_decisions=[decision] if decision else [],
                lessons_learned=[lesson] if lesson else [],
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning("local_task_error", agent_id=agent.agent_id, error=str(exc))
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

    async def get_available_agents(self) -> list[SpawnedAgent]:
        return list(self._agents.values())

    async def shutdown(self) -> None:
        self._agents.clear()
        logger.info("orchestrator_shutdown", backend="local")
