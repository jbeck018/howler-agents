"""Task performance scoring and normalization."""

from __future__ import annotations

from howler_agents.agents.base import Agent


class TaskPerformanceScorer:
    """Normalizes raw task performance scores to [0, 1]."""

    def score(self, agents: list[Agent]) -> None:
        """Normalize performance scores across the population."""
        if not agents:
            return

        raw_scores = [a.performance_score for a in agents]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        if score_range == 0:
            for agent in agents:
                agent.performance_score = 1.0
            return

        for agent in agents:
            agent.performance_score = (agent.performance_score - min_score) / score_range
