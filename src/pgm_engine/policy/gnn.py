"""Relational Message-Passing GNN for task reasoning and downstream loss computation."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalMessagePassingLayer(nn.Module):
    """Single-hop relational message passing with edge weights and relation representations."""

    def __init__(self, in_dim: int = 128, out_dim: int = 128, rel_dim: int = 32):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + rel_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.node_update = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_rel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward message passing step.

        Args:
            X: Active node feature matrix (num_nodes, in_dim).
            edge_index: COO tensor (2, num_edges).
            edge_weight: Scalar edge weights (num_edges,).
            edge_rel: Relation embeddings (num_edges, rel_dim).

        Returns:
            Updated node representations (num_nodes, out_dim).
        """
        num_nodes = X.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # Self-transformation only
            dummy_msg = torch.zeros((num_nodes, X.size(1)), dtype=X.dtype, device=X.device)
            return F.relu(self.node_update(torch.cat([X, dummy_msg], dim=-1)))

        src_indices = edge_index[0]
        tgt_indices = edge_index[1]

        u_feats = X[src_indices]
        v_feats = X[tgt_indices]

        if edge_rel is None:
            rel_feats = torch.zeros((num_edges, 32), dtype=X.dtype, device=X.device)
        else:
            rel_feats = edge_rel

        # Compute raw messages
        raw_msg = self.msg_mlp(torch.cat([u_feats, v_feats, rel_feats], dim=-1))
        # Scale messages by edge weight
        weighted_msg = raw_msg * edge_weight.unsqueeze(-1)

        # Aggregate messages into target nodes (scatter add)
        aggregated = torch.zeros((num_nodes, weighted_msg.size(1)), dtype=X.dtype, device=X.device)
        aggregated.index_add_(0, tgt_indices, weighted_msg)

        # Node update
        combined = torch.cat([X, aggregated], dim=-1)
        out = F.relu(self.node_update(combined))
        return out


class DownstreamTaskGNN(nn.Module):
    """Multi-layer GNN with readout head for downstream task loss evaluation."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 128,
        rel_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationalMessagePassingLayer(
                in_dim=node_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                rel_dim=rel_dim,
            )
            for i in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_rel: Optional[torch.Tensor] = None,
        source_idx: int = 0,
        target_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs message passing and computes downstream task score.

        Returns:
            score: Scalar prediction logit for task resolution.
            final_X: Final node representations (with gradients enabled).
        """
        curr_X = X
        for layer in self.layers:
            curr_X = layer(curr_X, edge_index, edge_weight, edge_rel)

        src_rep = curr_X[source_idx]
        tgt_rep = curr_X[target_idx]
        pair_rep = torch.cat([src_rep, tgt_rep], dim=-1)
        score = self.predictor(pair_rep.unsqueeze(0)).squeeze(-1)
        return score, curr_X
