"""Differentiable Dynamic Subgraph Selection via Concrete / Gumbel-Softmax Relaxation.

Enables end-to-end backpropagation through graph topology and procedural edge materialization.
Supports Straight-Through (ST) binary decisions during forward inference while preserving
continuous gradients during backward optimization.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSubgraphSelector(nn.Module):
    """Predicts edge existence probabilities and samples differentiable edge masks.

    Args:
        node_dim: Dimensionality of node representations (d).
        rel_dim: Dimensionality of relation representations (d_r).
        hidden_dim: Hidden dimension for edge scorer MLP.
        temperature: Concrete/Gumbel-Softmax relaxation temperature tau.
    """

    def __init__(
        self,
        node_dim: int = 128,
        rel_dim: int = 32,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.rel_dim = rel_dim
        self.temperature = temperature

        input_dim = 2 * node_dim + rel_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        u_features: torch.Tensor,
        v_features: torch.Tensor,
        rel_features: Optional[torch.Tensor] = None,
        hard: bool = False,
        temp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes continuous relaxed edge existence weights.

        Args:
            u_features: Source node embeddings (N, node_dim).
            v_features: Target node embeddings (N, node_dim).
            rel_features: Optional relation embeddings (N, rel_dim).
            hard: If True, uses straight-through binary discretization.
            temp: Optional override for relaxation temperature.

        Returns:
            sampled_weights: Continuous weights in (0, 1) or ST binary weights.
            logits: Unnormalized edge scoring logits.
        """
        tau = temp or self.temperature
        N = u_features.size(0)

        if rel_features is None:
            rel_features = torch.zeros(
                (N, self.rel_dim),
                dtype=u_features.dtype,
                device=u_features.device,
            )

        cat_inputs = torch.cat([u_features, v_features, rel_features], dim=-1)
        logits = self.edge_mlp(cat_inputs).squeeze(-1)  # (N,)

        if self.training:
            # Gumbel-Sigmoid sampling: g1, g2 ~ Gumbel(0, 1)
            # Sample uniform u ~ (0, 1)
            u = torch.rand_like(logits).clamp(1e-7, 1.0 - 1e-7)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            relaxed = torch.sigmoid((logits + gumbel_noise) / tau)
        else:
            relaxed = torch.sigmoid(logits / tau)

        if hard:
            # Straight-Through Estimator: discrete forward, continuous backward
            hard_mask = (relaxed >= 0.5).float()
            sampled_weights = hard_mask - relaxed.detach() + relaxed
        else:
            sampled_weights = relaxed

        return sampled_weights, logits
