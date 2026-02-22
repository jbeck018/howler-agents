"""Combined performance + novelty selection."""

from __future__ import annotations

from howler_agents.agents.base import Agent
from howler_agents.selection.novelty import KNNNoveltyEstimator
from howler_agents.selection.performance import TaskPerformanceScorer


class PerformanceNoveltySelector:
    """Selects agents using combined performance and novelty scores.

    Combined score = alpha * performance + (1 - alpha) * novelty
    """

    def __init__(self, alpha: float = 0.5, k_neighbors: int = 5) -> None:
        self.alpha = alpha
        self._novelty = KNNNoveltyEstimator(k_neighbors=k_neighbors)
        self._performance = TaskPerformanceScorer()

    def score_agents(self, agents: list[Agent]) -> None:
        """Compute and assign combined scores for all agents."""
        self._performance.score(agents)
        self._novelty.score(agents)
        for agent in agents:
            agent.combined_score = (
                self.alpha * agent.performance_score + (1 - self.alpha) * agent.novelty_score
            )

    def select(self, agents: list[Agent], num_survivors: int) -> list[Agent]:
        """Score and select the top agents."""
        self.score_agents(agents)
        return sorted(agents, key=lambda a: a.combined_score, reverse=True)[:num_survivors]

    def select_groups(self, agents: list[Agent], num_survivors: int) -> list[Agent]:
        """Score agents then select by group average score.

        Computes the average combined_score for each group, ranks groups by
        that average, and keeps ALL members of the top groups until at least
        num_survivors agents have been collected.  Falls back to individual
        selection when no agents carry a group_id.
        """
        self.score_agents(agents)

        # Build group -> members mapping
        group_map: dict[str, list[Agent]] = {}
        ungrouped: list[Agent] = []
        for agent in agents:
            gid = agent.config.group_id
            if gid:
                group_map.setdefault(gid, []).append(agent)
            else:
                ungrouped.append(agent)

        if not group_map:
            # No groups present — fall back to individual selection
            return sorted(agents, key=lambda a: a.combined_score, reverse=True)[:num_survivors]

        # Rank groups by average combined_score of their members
        group_scores: list[tuple[float, str]] = []
        for gid, members in group_map.items():
            avg = sum(m.combined_score for m in members) / len(members)
            group_scores.append((avg, gid))
        group_scores.sort(reverse=True)

        # Collect members of top groups until we have at least num_survivors
        survivors: list[Agent] = []
        for _, gid in group_scores:
            if len(survivors) >= num_survivors:
                break
            survivors.extend(group_map[gid])

        # Include any ungrouped agents ranked by individual score (rare edge case)
        if ungrouped:
            remaining = num_survivors - len(survivors)
            if remaining > 0:
                survivors.extend(
                    sorted(ungrouped, key=lambda a: a.combined_score, reverse=True)[:remaining]
                )

        return survivors
