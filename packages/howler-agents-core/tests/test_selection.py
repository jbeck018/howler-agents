"""Tests for selection criterion."""

import pytest
from _helpers import make_agent

from howler_agents.selection.criterion import PerformanceNoveltySelector
from howler_agents.selection.novelty import KNNNoveltyEstimator
from howler_agents.selection.performance import TaskPerformanceScorer


def test_performance_scorer_normalizes():
    scorer = TaskPerformanceScorer()
    agents = [make_agent(score=s) for s in [0.2, 0.8, 0.5]]
    scorer.score(agents)
    assert agents[0].performance_score == pytest.approx(0.0)
    assert agents[1].performance_score == pytest.approx(1.0)
    assert agents[2].performance_score == pytest.approx(0.5)


def test_performance_scorer_uniform():
    scorer = TaskPerformanceScorer()
    agents = [make_agent(score=0.5) for _ in range(3)]
    scorer.score(agents)
    assert all(a.performance_score == 1.0 for a in agents)


def test_novelty_scorer():
    estimator = KNNNoveltyEstimator(k_neighbors=2)
    agents = [make_agent() for _ in range(5)]
    # Give distinct vectors
    for i, a in enumerate(agents):
        a.capability_vector = [float(i == j) for j in range(5)]
    estimator.score(agents)
    assert all(0 <= a.novelty_score <= 1 for a in agents)


def test_combined_selector():
    selector = PerformanceNoveltySelector(alpha=0.5, k_neighbors=2)
    agents = [make_agent(score=s) for s in [0.9, 0.3, 0.6, 0.1, 0.7]]
    for i, a in enumerate(agents):
        a.capability_vector = [float(i == j) for j in range(5)]

    survivors = selector.select(agents, num_survivors=3)
    assert len(survivors) == 3
    # All should have combined scores assigned
    assert all(a.combined_score > 0 for a in survivors)
