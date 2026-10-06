"""A* Lookahead Value Routing with Admissible Latent Heuristics.

Replaces greedy 1-hop similarity with A* search using accumulated path cost g(v)
and admissible latent distance heuristic h(v) relative to target task vector.
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.types import EdgeRelation, EntityNode


class AStarValueRouter:
    """A* Lookahead router with latent task heuristics."""

    def __init__(
        self,
        heuristic_weight: float = 1.0,
        corridor_penalty: float = 5.0,
    ):
        self.heuristic_weight = heuristic_weight
        self.corridor_penalty = corridor_penalty

    def compute_heuristic(self, node_emb: np.ndarray, target_emb: np.ndarray) -> float:
        """Admissible heuristic estimating remaining distance to target state in latent space.

        h(v) = 1.0 - cosine_similarity(z_v, z_target) in [0.0, 2.0].
        """
        norm_v = np.linalg.norm(node_emb)
        norm_t = np.linalg.norm(target_emb)
        if norm_v < 1e-9 or norm_t < 1e-9:
            return 1.0
        cosine_sim = float(np.dot(node_emb, target_emb) / (norm_v * norm_t))
        # Clamp to [-1, 1]
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return (1.0 - cosine_sim) * self.heuristic_weight

    def compute_edge_traversal_cost(self, edge: EdgeRelation) -> float:
        """Calculates traversal cost: higher weight means lower cost."""
        # Using soft inverse weight: cost in [0.05, 2.0]
        w = max(1e-4, edge.weight)
        return max(0.05, 1.0 / (1.0 + w))

    def compute_utility(self, g_cost: float, h_cost: float) -> float:
        """Maps f(v) = g(v) + h(v) to a normalized utility score in (0, 1] for hysteresis."""
        f_cost = g_cost + h_cost
        return float(1.0 / (1.0 + f_cost))

    def evaluate_node(
        self,
        node: EntityNode,
        target_emb: np.ndarray,
        parent_g: float,
        edge: EdgeRelation,
        corridor: Optional[Set[int]] = None,
    ) -> Tuple[float, float, float, float]:
        """Evaluates g(v), h(v), f(v), and utility U(v).

        Returns:
            Tuple of (f_cost, g_cost, h_cost, utility).
        """
        step_cost = self.compute_edge_traversal_cost(edge)
        g_cost = parent_g + step_cost

        # Penalize nodes that deviate from the coarse landmark corridor
        if corridor is not None and node.cluster_id not in corridor and node.cluster_id != -1:
            g_cost += self.corridor_penalty

        h_cost = self.compute_heuristic(node.embedding, target_emb)
        f_cost = g_cost + h_cost
        utility = self.compute_utility(g_cost, h_cost)
        return f_cost, g_cost, h_cost, utility
