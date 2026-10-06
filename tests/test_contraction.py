"""Unit tests for Virtual Shortcut Synthesis and Topological Edge Contraction."""

import pytest
import numpy as np

from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.topology.contraction import EdgeContractionOperator
from pgm_engine.types import EdgeRelation, EntityNode


def test_edge_contraction_shortcut_synthesis():
    l2 = L2HostGraph()
    op = EdgeContractionOperator(rel_dim=8)

    # Build A -> B -> C
    dim = 16
    l2.add_node(EntityNode("A", np.ones(dim, dtype=np.float32)))
    l2.add_node(EntityNode("B", np.ones(dim, dtype=np.float32)))
    l2.add_node(EntityNode("C", np.ones(dim, dtype=np.float32)))

    rel1 = np.ones(8, dtype=np.float32) / np.sqrt(8)
    rel2 = np.ones(8, dtype=np.float32) / np.sqrt(8)

    e_ab = EdgeRelation("A", "B", relation="leads_to", weight=0.8, relation_embedding=rel1)
    e_bc = EdgeRelation("B", "C", relation="causes", weight=0.5, relation_embedding=rel2)
    l2.add_edge(e_ab)
    l2.add_edge(e_bc)

    # Contract node B
    synthesized = op.contract_node("B", l2_graph=l2, weight_decay_factor=0.9)

    assert len(synthesized) == 1
    shortcut = synthesized[0]

    assert shortcut.source_id == "A"
    assert shortcut.target_id == "C"
    # Weight: 0.8 * 0.5 * 0.9 = 0.36
    assert shortcut.weight == pytest.approx(0.36)
    assert shortcut.is_virtual is True
    assert shortcut.hop_count == 2
    assert shortcut.provenance == ["B"]
    assert "virtual_leads_to_causes" == shortcut.relation

    # Reachability check: A can reach C directly in L2
    assert "C" in l2.out_edges["A"]
