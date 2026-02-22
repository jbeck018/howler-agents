"""Probe evaluator - builds binary capability vectors."""

from __future__ import annotations

from howler_agents.agents.base import Agent
from howler_agents.probes.registry import ProbeRegistry


class ProbeEvaluator:
    """Evaluates agents on probe tasks to build capability vectors."""

    def __init__(self, registry: ProbeRegistry) -> None:
        self._registry = registry

    async def evaluate(self, agent: Agent) -> list[float]:
        """Run all registered probes and return binary capability vector."""
        probes = self._registry.get_probes()
        vector: list[float] = []

        for probe in probes:
            result = await agent.run_task(probe)
            vector.append(1.0 if result.success else 0.0)

        return vector
