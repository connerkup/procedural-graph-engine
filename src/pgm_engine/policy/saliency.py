"""Saliency-Based Autonomous Pruning via Gradient Accumulation.

Tracks downstream task sensitivity with respect to node representations across a sliding
window, identifying low-influence nodes for autonomous edge contraction and eviction.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set
import numpy as np
import torch


class SaliencyPruningTracker:
    """Accumulates task loss gradients to compute lifetime-normalized node influence.

    Influence formulation:
        I(v) = (1 / Delta t_v) * sum_{t=1}^W ||nabla_{z_v} L_task||_2

    Args:
        window_size (W): Sliding window length for gradient norm history.
        saliency_threshold (epsilon): Influence cutoff below which node is eviction-eligible.
    """

    def __init__(
        self,
        window_size: int = 10,
        saliency_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.saliency_threshold = saliency_threshold

        # entity_id -> deque of gradient norms across recent steps
        self.gradient_history: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        # entity_id -> creation step / lifetime counter
        self.creation_step: Dict[str, int] = {}
        self.current_step: int = 0

    def step(self) -> None:
        """Increments temporal counter."""
        self.current_step += 1

    def register_node(self, entity_id: str) -> None:
        """Registers a newly materialized node."""
        if entity_id not in self.creation_step:
            self.creation_step[entity_id] = self.current_step

    def record_gradient(self, entity_id: str, grad_norm: float) -> None:
        """Records an instantaneous gradient norm for an entity representation."""
        if entity_id not in self.creation_step:
            self.creation_step[entity_id] = self.current_step
        self.gradient_history[entity_id].append(float(grad_norm))

    def record_tensor_gradients(
        self,
        compact_id_map: Dict[str, int],
        X_grad: torch.Tensor,
    ) -> None:
        """Extracts and records gradient norms directly from node feature tensor grad.

        Args:
            compact_id_map: Map from entity_id to compact tensor row index.
            X_grad: Tensor of shape (num_active, node_dim) with gradients.
        """
        if X_grad is None:
            return

        with torch.no_grad():
            grad_norms = torch.norm(X_grad, p=2, dim=-1).cpu().numpy()

        for entity_id, row_idx in compact_id_map.items():
            if row_idx < len(grad_norms):
                self.record_gradient(entity_id, float(grad_norms[row_idx]))

    def compute_influence(self, entity_id: str) -> float:
        """Calculates normalized lifetime influence I(v)."""
        history = self.gradient_history.get(entity_id)
        if not history:
            return 0.0

        t_birth = self.creation_step.get(entity_id, self.current_step)
        delta_t = max(1, self.current_step - t_birth)

        grad_sum = sum(history)
        influence = grad_sum / float(delta_t)
        return float(influence)

    def get_low_influence_candidates(
        self,
        active_nodes: Set[str],
        min_age_steps: int = 3,
    ) -> List[tuple[str, float]]:
        """Returns active nodes whose influence falls below threshold, sorted by influence ascending."""
        candidates = []
        for eid in active_nodes:
            t_birth = self.creation_step.get(eid, self.current_step)
            age = self.current_step - t_birth
            if age >= min_age_steps:
                infl = self.compute_influence(eid)
                if infl < self.saliency_threshold:
                    candidates.append((eid, infl))

        # Sort ascending (lowest influence first)
        candidates.sort(key=lambda x: x[1])
        return candidates

    def deregister_node(self, entity_id: str) -> None:
        """Cleans up tracking state for an evicted entity."""
        self.gradient_history.pop(entity_id, None)
        self.creation_step.pop(entity_id, None)
