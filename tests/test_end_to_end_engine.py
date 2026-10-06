"""End-to-end integration tests for ProceduralGraphEngine."""

import pytest
import numpy as np
import torch

from pgm_engine.engine import ProceduralGraphEngine


def test_engine_full_workflow():
    dim = 32
    rel_dim = 16
    engine = ProceduralGraphEngine(
        max_nodes_k=64,
        max_edges_m=256,
        node_dim=dim,
        rel_dim=rel_dim,
        tau_push=0.50,
        tau_pop=0.20,
        t_cooldown=3,
        num_landmark_clusters=4,
    )

    # 1. Populate Substrate with multi-hop chain and distractors
    nodes = ["N0", "N1", "N2", "N3", "N4"]
    for i, nid in enumerate(nodes):
        emb = np.random.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        engine.l3_substrate.add_entity(nid, emb)

    # Chain edges: N0 -> N1 -> N2 -> N3 -> N4
    for i in range(len(nodes) - 1):
        engine.l3_substrate.add_relation(nodes[i], nodes[i+1], relation=f"step_{i}", weight=0.9)

    # 2. Build Landmarks
    engine.initialize_landmarks()
    assert engine.landmark_graph.num_clusters > 0

    # 3. Query Corridor
    result = engine.query_reasoning_corridor("N0", "N4")
    assert result.success is True
    assert result.path_nodes == nodes

    # Verify L1 tensor population
    active_features, active_ids, id_map = engine.l1_buffer.get_active_features()
    assert len(active_ids) == len(nodes)
    for nid in nodes:
        assert nid in id_map

    # 4. Forward GNN Pass
    task_score, final_X = engine.forward_gnn_step(source_id="N0", target_id="N4")
    assert task_score is not None

    # 5. Saliency Backprop
    target_logit = torch.tensor([1.0], dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(task_score, target_logit)
    engine.backprop_saliency(loss, final_X)

    # Influence of nodes in reasoning chain should be recorded
    for nid in nodes:
        assert nid in engine.saliency_tracker.gradient_history

    # 6. Contraction and Pruning
    # Prune down to 3 nodes: intermediate bridge nodes should contract!
    synthesized = engine.prune_and_contract(target_capacity=3)
    assert len(engine.l1_buffer.entity_to_slot) <= 4

    telemetry = engine.get_telemetry()
    assert telemetry["active_l1_nodes"] <= 4
    assert telemetry["peak_l1_nodes"] >= 5
