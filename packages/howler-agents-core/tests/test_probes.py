"""Tests for probe evaluator and registry."""

import pytest
from _helpers import MockAgent

from howler_agents.agents.base import AgentConfig
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry


@pytest.fixture
def registry() -> ProbeRegistry:
    reg = ProbeRegistry()
    reg.register({"description": "probe 1", "type": "test"})
    reg.register({"description": "probe 2", "type": "test"})
    reg.register({"description": "probe 3", "type": "test"})
    return reg


@pytest.mark.asyncio
async def test_probe_evaluator(registry: ProbeRegistry):
    agent = MockAgent(AgentConfig(), score=0.8)
    evaluator = ProbeEvaluator(registry)
    vector = await evaluator.evaluate(agent)
    assert len(vector) == 3
    assert all(v in (0.0, 1.0) for v in vector)


def test_registry_default_probes():
    reg = ProbeRegistry()
    reg.register_default_probes(num_probes=10)
    assert reg.count == 10


def test_registry_count():
    reg = ProbeRegistry()
    assert reg.count == 0
    reg.register({"type": "test"})
    assert reg.count == 1
