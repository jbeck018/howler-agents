"""Tests for AgentPool."""

from _helpers import make_agent

from howler_agents.agents.pool import AgentPool


def test_add_and_get():
    pool = AgentPool()
    agent = make_agent(score=0.8)
    pool.add(agent)
    assert pool.get(agent.id) is agent
    assert pool.size == 1


def test_remove():
    pool = AgentPool()
    agent = make_agent()
    pool.add(agent)
    removed = pool.remove(agent.id)
    assert removed is agent
    assert pool.size == 0


def test_top_k():
    pool = AgentPool()
    agents = [make_agent(score=s) for s in [0.3, 0.9, 0.5, 0.7, 0.1]]
    for a in agents:
        a.combined_score = a.performance_score
        pool.add(a)

    top = pool.top_k(3)
    scores = [a.combined_score for a in top]
    assert scores == [0.9, 0.7, 0.5]


def test_partition_groups():
    pool = AgentPool()
    for _ in range(6):
        pool.add(make_agent())
    groups = pool.partition_groups(3)
    assert len(groups) == 2
    assert all(len(g) == 3 for g in groups)


def test_replace_population():
    pool = AgentPool()
    pool.add(make_agent())
    new_agents = [make_agent() for _ in range(3)]
    pool.replace_population(new_agents)
    assert pool.size == 3
