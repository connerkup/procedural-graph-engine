"""Unit tests for Differentiable Subgraph Selection, GNN Loss, and Saliency Tracking."""

import pytest
import numpy as np
import torch

from pgm_engine.policy.differentiable import DifferentiableSubgraphSelector
from pgm_engine.policy.gnn import DownstreamTaskGNN
from pgm_engine.policy.saliency import SaliencyPruningTracker


def test_differentiable_subgraph_selector_gradients():
    selector = DifferentiableSubgraphSelector(node_dim=16, rel_dim=8, hidden_dim=32, temperature=0.5)

    u = torch.randn(4, 16, requires_grad=True)
    v = torch.randn(4, 16, requires_grad=True)
    r = torch.randn(4, 8, requires_grad=True)

    weights, logits = selector(u, v, r, hard=False)
    assert weights.shape == (4,)
    assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0)

    # Backprop through continuous weights
    loss = weights.sum()
    loss.backward()

    assert u.grad is not None
    assert v.grad is not None
    assert r.grad is not None
    assert torch.norm(u.grad) > 0


def test_straight_through_estimator():
    selector = DifferentiableSubgraphSelector(node_dim=8, rel_dim=4, hidden_dim=16)
    selector.eval()

    u = torch.randn(3, 8, requires_grad=True)
    v = torch.randn(3, 8, requires_grad=True)

    hard_weights, _ = selector(u, v, hard=True)
    # Check that forward pass outputs exact binary values {0.0, 1.0}
    for val in hard_weights.detach().cpu().tolist():
        assert val in (0.0, 1.0)


def test_saliency_influence_accumulation():
    tracker = SaliencyPruningTracker(window_size=5, saliency_threshold=0.1)

    tracker.register_node("critical_node")
    tracker.register_node("useless_node")

    # Simulate 5 steps of training
    for step in range(5):
        tracker.step()
        tracker.record_gradient("critical_node", grad_norm=0.85)
        tracker.record_gradient("useless_node", grad_norm=0.01)

    crit_infl = tracker.compute_influence("critical_node")
    use_infl = tracker.compute_influence("useless_node")

    assert crit_infl > 0.5
    assert use_infl < 0.05

    low_candidates = tracker.get_low_influence_candidates({"critical_node", "useless_node"}, min_age_steps=2)
    assert len(low_candidates) == 1
    assert low_candidates[0][0] == "useless_node"
