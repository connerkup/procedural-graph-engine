"""Unit tests for Landmark Graph, A* Value Iteration, and Defeating Bridge Entity Blindness."""

import pytest
import numpy as np

from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.memory.manager import MemoryManager
from pgm_engine.routing.a_star import AStarValueRouter
from pgm_engine.routing.corridor import CorridorMaterializer
from pgm_engine.routing.landmark import GlobalLandmarkGraph


def test_bridge_entity_blindness_resolution():
    """Validates that coarse landmark corridor protects intermediate bridge nodes with 0 query overlap."""
    dim = 8
    sub = L3LatentSubstrate(node_dim=dim, rel_dim=4)

    # Unit orthogonal vectors
    # Source entity S: [1, 0, 0, 0, ...]
    v_s = np.zeros(dim, dtype=np.float32)
    v_s[0] = 1.0

    # Bridge entity B: [0, 1, 0, 0, ...] -> Orthogonal to both S and T!
    # Cosine similarity to target T is strictly 0.0!
    v_b = np.zeros(dim, dtype=np.float32)
    v_b[1] = 1.0

    # Target entity T: [0, 0, 1, 0, ...]
    v_t = np.zeros(dim, dtype=np.float32)
    v_t[2] = 1.0

    # Distractor node D: has 0.707 cosine similarity to T, but is a dead-end!
    v_d = np.zeros(dim, dtype=np.float32)
    v_d[0] = 0.5
    v_d[2] = 0.5
    v_d /= np.linalg.norm(v_d)

    sub.add_entity("S", v_s)
    sub.add_entity("B", v_b)
    sub.add_entity("T", v_t)
    sub.add_entity("D", v_d)

    # Ground-truth path: S -> B -> T
    # False greedy branch: S -> D (dead end)
    sub.add_relation("S", "B", relation="rel_sb", weight=1.0)
    sub.add_relation("B", "T", relation="rel_bt", weight=1.0)
    sub.add_relation("S", "D", relation="rel_sd", weight=0.9)

    # Initialize Landmark Graph
    landmark = GlobalLandmarkGraph(num_clusters=4, node_dim=dim)
    landmark.build_from_substrate(sub)

    mem_mgr = MemoryManager(
        l1_buffer=L1TensorBuffer(max_nodes=16, max_edges=32, node_dim=dim, rel_dim=4),
        l2_graph=L2HostGraph(),
        l3_substrate=sub,
    )

    materializer = CorridorMaterializer(
        memory_manager=mem_mgr,
        landmark_graph=landmark,
    )

    res = materializer.materialize_corridor(source_id="S", target_id="T")

    # The bridge node B must be discovered and recovered despite 0 direct target similarity
    assert res.success is True
    assert "S" in res.path_nodes
    assert "B" in res.path_nodes
    assert "T" in res.path_nodes
    assert "D" not in res.path_nodes
