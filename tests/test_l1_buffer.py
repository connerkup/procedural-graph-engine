"""Unit tests for L1TensorBuffer: Pre-allocated static tensors and free-index recycling."""

import pytest
import numpy as np
import torch

from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer


def test_l1_buffer_initialization():
    buf = L1TensorBuffer(max_nodes=16, max_edges=32, node_dim=64, rel_dim=16)
    assert buf.X.shape == (16, 64)
    assert buf.E.shape == (2, 32)
    assert buf.W.shape == (32,)
    assert buf.R.shape == (32, 16)
    assert buf.num_active_nodes == 0
    assert buf.num_active_edges == 0
    assert not buf.is_node_full
    assert not buf.is_edge_full


def test_l1_node_allocation_and_free():
    buf = L1TensorBuffer(max_nodes=4, max_edges=8, node_dim=8)
    emb1 = np.ones(8, dtype=np.float32) * 1.5
    emb2 = np.ones(8, dtype=np.float32) * 2.5

    slot1 = buf.allocate_node("entity_1", emb1)
    slot2 = buf.allocate_node("entity_2", emb2)

    assert slot1 != slot2
    assert buf.num_active_nodes == 2
    assert torch.allclose(buf.X[slot1], torch.from_numpy(emb1))
    assert torch.allclose(buf.X[slot2], torch.from_numpy(emb2))

    # Free node 1
    freed = buf.free_node("entity_1")
    assert freed == slot1
    assert buf.num_active_nodes == 1
    assert buf.node_mask[slot1] == False
    assert torch.all(buf.X[slot1] == 0)

    # Re-allocate into recycled slot
    emb3 = np.ones(8, dtype=np.float32) * 3.5
    slot3 = buf.allocate_node("entity_3", emb3)
    assert slot3 == slot1  # Reused from LIFO free list
    assert buf.num_active_nodes == 2


def test_l1_edge_allocation_and_cascade():
    buf = L1TensorBuffer(max_nodes=4, max_edges=8, node_dim=8, rel_dim=4)
    emb = np.ones(8, dtype=np.float32)

    buf.allocate_node("A", emb)
    buf.allocate_node("B", emb)
    buf.allocate_node("C", emb)

    e1 = buf.allocate_edge("A", "B", relation="to_b", weight=0.8)
    e2 = buf.allocate_edge("B", "C", relation="to_c", weight=0.9)

    assert buf.num_active_edges == 2
    assert buf.W[e1] == pytest.approx(0.8)
    assert buf.W[e2] == pytest.approx(0.9)

    # Freeing node B must cascade and free both incident edges
    buf.free_node("B")
    assert buf.num_active_nodes == 2
    assert buf.num_active_edges == 0
    assert not buf.edge_mask[e1]
    assert not buf.edge_mask[e2]


def test_l1_capacity_exhaustion():
    buf = L1TensorBuffer(max_nodes=2, max_edges=2, node_dim=4)
    emb = np.zeros(4, dtype=np.float32)

    buf.allocate_node("A", emb)
    buf.allocate_node("B", emb)
    assert buf.is_node_full

    with pytest.raises(RuntimeError, match="L1 Node Capacity Exhausted"):
        buf.allocate_node("C", emb)
