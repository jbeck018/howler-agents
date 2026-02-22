"""Tests for experience pool and stores."""

import pytest

from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.experience.trace import EvolutionaryTrace


@pytest.fixture
def pool() -> SharedExperiencePool:
    return SharedExperiencePool(InMemoryStore())


@pytest.mark.asyncio
async def test_submit_and_retrieve(pool: SharedExperiencePool):
    trace = EvolutionaryTrace(
        agent_id="a1", run_id="r1", generation=0,
        task_description="test task", outcome="success", score=0.9,
        key_decisions=["used tool A"], lessons_learned=["tool A works well"],
    )
    await pool.submit(trace)
    history = await pool.get_agent_history("a1")
    assert len(history) == 1
    assert history[0].score == 0.9


@pytest.mark.asyncio
async def test_group_context_empty(pool: SharedExperiencePool):
    ctx = await pool.get_group_context("r1", "g1", 0)
    assert "No prior experience" in ctx


@pytest.mark.asyncio
async def test_group_context_with_traces(pool: SharedExperiencePool):
    for i in range(3):
        await pool.submit(EvolutionaryTrace(
            agent_id=f"a{i}", run_id="r1", generation=0,
            task_description=f"task {i}", outcome="success", score=0.5 + i * 0.1,
            lessons_learned=[f"lesson {i}"],
        ))
    ctx = await pool.get_group_context("r1", "g1", 1)
    assert "Generation 0" in ctx
    assert "lesson" in ctx


@pytest.mark.asyncio
async def test_memory_store_delete():
    store = InMemoryStore()
    for i in range(5):
        await store.save(EvolutionaryTrace(agent_id="a1", run_id="r1", generation=0,
                                           task_description="t", outcome="ok", score=0.5))
    deleted = await store.delete_by_run("r1")
    assert deleted == 5
    remaining = await store.get_by_run("r1")
    assert len(remaining) == 0
