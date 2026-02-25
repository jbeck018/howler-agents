"""KNN-based novelty estimation in capability vector space."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from howler_agents.agents.base import Agent


class KNNNoveltyEstimator:
    """Estimates novelty as mean KNN distance in capability vector space."""

    def __init__(self, k_neighbors: int = 5) -> None:
        self.k_neighbors = k_neighbors

    def score(self, agents: list[Agent]) -> None:
        """Compute and assign novelty scores for all agents."""
        vectors = [a.capability_vector for a in agents if a.capability_vector]
        if len(vectors) < 2:
            for agent in agents:
                agent.novelty_score = 1.0
            return

        matrix = np.array(vectors, dtype=np.float64)
        k = min(self.k_neighbors, len(vectors) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(matrix)
        distances, _ = nn.kneighbors(matrix)

        # Mean distance to K nearest neighbors (exclude self at index 0)
        mean_distances = distances[:, 1:].mean(axis=1)

        # Normalize to [0, 1]
        max_dist = mean_distances.max()
        if max_dist > 0:
            normalized = mean_distances / max_dist
        else:
            normalized = np.ones(len(mean_distances))

        vec_idx = 0
        for agent in agents:
            if agent.capability_vector:
                agent.novelty_score = float(normalized[vec_idx])
                vec_idx += 1
            else:
                agent.novelty_score = 1.0  # Novel by default (unknown capability)
