"""Tests for the evolution loop."""

from unittest.mock import AsyncMock

import pytest
from _helpers import make_agent

from howler_agents.agents.base import FrameworkPatch
from howler_agents.agents.pool import AgentPool
from howler_agents.config import HowlerConfig
from howler_agents.evolution.directive import EvolutionDirective
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector


@pytest.fixture
def evolution_loop() -> EvolutionLoop:
    config = HowlerConfig(population_size=4, group_size=2, num_iterations=2, num_probes=3)
    pool = AgentPool()
    for _ in range(4):
        pool.add(make_agent(score=0.5))

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    selector = PerformanceNoveltySelector(alpha=config.alpha)

    registry = ProbeRegistry()
    registry.register_default_probes(num_probes=3)
    probe_eval = ProbeEvaluator(registry)

    reproducer = AsyncMock(spec=GroupReproducer)
    reproducer.reproduce = AsyncMock(
        return_value=(
            FrameworkPatch(intent="test patch"),
            EvolutionDirective(intent="test"),
        )
    )

    return EvolutionLoop(
        config=config,
        pool=pool,
        selector=selector,
        reproducer=reproducer,
        experience=experience,
        probe_evaluator=probe_eval,
    )


@pytest.mark.asyncio
async def test_step(evolution_loop: EvolutionLoop):
    tasks = [{"description": "test task", "type": "test"}]
    summary = await evolution_loop.step("run-1", 0, tasks)
    assert summary["generation"] == 0
    assert summary["population_size"] > 0
    assert "best_score" in summary
