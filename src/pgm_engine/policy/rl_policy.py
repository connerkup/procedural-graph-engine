"""Push/Pop Policy Networks and Reward Formulations for RL Optimization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPolicyNetwork(nn.Module):
    """Predicts discrete push and pop admission/eviction probabilities.

    pi_push(e | Task, G_t)
    pi_pop(v | Task, G_t)
    """

    def __init__(
        self,
        node_dim: int = 128,
        task_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.task_dim = task_dim

        # Input: [entity_emb || task_emb || graph_context]
        feature_dim = node_dim + task_dim + node_dim

        self.push_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.pop_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        entity_embeddings: torch.Tensor,
        task_embedding: torch.Tensor,
        graph_summary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates push logits and pop logits for candidate/active entities.

        Args:
            entity_embeddings: (N, node_dim).
            task_embedding: (task_dim,) or (1, task_dim).
            graph_summary: (node_dim,) mean pool of active working set.

        Returns:
            push_probs: (N,) probability in [0, 1] of pushing to L1.
            pop_probs: (N,) probability in [0, 1] of popping from L1.
        """
        N = entity_embeddings.size(0)
        if task_embedding.dim() == 1:
            task_emb_expanded = task_embedding.unsqueeze(0).expand(N, -1)
        else:
            task_emb_expanded = task_embedding.expand(N, -1)

        if graph_summary.dim() == 1:
            summary_expanded = graph_summary.unsqueeze(0).expand(N, -1)
        else:
            summary_expanded = graph_summary.expand(N, -1)

        combined = torch.cat([entity_embeddings, task_emb_expanded, summary_expanded], dim=-1)

        push_logits = self.push_mlp(combined).squeeze(-1)
        pop_logits = self.pop_mlp(combined).squeeze(-1)

        return torch.sigmoid(push_logits), torch.sigmoid(pop_logits)


class PolicyRewardEngine:
    """Computes reinforcement learning rewards based on downstream task delta and memory efficiency.

    Reward Formulation:
        R_t = -Delta L_task - alpha * (|V_t| / K) - beta * Latency
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Memory penalty scale
        beta: float = 0.05,  # Latency penalty scale
        max_capacity_k: int = 2048,
    ):
        self.alpha = alpha
        self.beta = beta
        self.max_capacity_k = max_capacity_k

    def compute_reward(
        self,
        prev_loss: float,
        curr_loss: float,
        active_nodes: int,
        latency_ms: float,
    ) -> float:
        """Calculates scalar step reward."""
        delta_loss = curr_loss - prev_loss
        memory_ratio = float(active_nodes) / float(self.max_capacity_k)
        # Latency normalized by 10ms target
        latency_norm = latency_ms / 10.0

        reward = (-delta_loss) - (self.alpha * memory_ratio) - (self.beta * latency_norm)
        return float(reward)
