"""Unit tests for Bidirectional Frontier Expansion and Latent Boundary Intersection."""

import pytest
import numpy as np

from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.routing.bidirectional import BidirectionalFrontierRouter


def test_bidirectional_latent_boundary_intersection():
    """Validates that frontiers connect when latent boundary distance is <= epsilon."""
    dim = 8
    sub = L3LatentSubstrate(node_dim=dim, rel_dim=4)

    # Chain of entities in embedding space: S -> U ... W <- T
    # U and W are close in embedding space (||z_u - z_w|| <= 0.20)
    v_s = np.zeros(dim, dtype=np.float32); v_s[0] = 1.0
    v_u = np.zeros(dim, dtype=np.float32); v_u[0] = 0.9; v_u[1] = 0.1; v_u /= np.linalg.norm(v_u)
    v_w = np.zeros(dim, dtype=np.float32); v_w[0] = 0.9; v_w[1] = 0.15; v_w /= np.linalg.norm(v_w)
    v_t = np.zeros(dim, dtype=np.float32); v_t[1] = 1.0

    sub.add_entity("S", v_s)
    sub.add_entity("U", v_u)
    sub.add_entity("W", v_w)
    sub.add_entity("T", v_t)

    sub.add_relation("S", "U", "fwd", weight=1.0)
    sub.add_relation("W", "T", "bwd", weight=1.0)

    # Note: No explicit edge exists in substrate between U and W!
    # They should connect procedurally via latent boundary intersection!
    dist_uw = float(np.linalg.norm(v_u - v_w))
    assert dist_uw < 0.25

    router = BidirectionalFrontierRouter(latent_epsilon=0.25)
    l2 = L2HostGraph()

    res = router.search("S", "T", l2_graph=l2, substrate=sub)

    assert res.success is True
    assert res.virtual_shortcuts_traversed >= 1
    assert any(e.relation == "latent_boundary_bridge" for e in res.path_edges)
