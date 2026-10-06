"""Procedural Graph Materialization Engine.

Unified production-grade engine integrating:
1. Non-Greedy Path Discovery (Global Landmark Scaffolding & Bidirectional A* Value Iteration)
2. Topological Graph Contraction (Articulation Shielding & Virtual Shortcut Synthesis)
3. High-Throughput Tiered Memory (Pre-allocated Static L1 Tensors & Dual-Threshold Hysteresis)
4. Policy Learning & Differentiable Optimization (Gumbel-Softmax Selection & Saliency Pruning)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from pgm_engine.memory.hysteresis import HysteresisConfig
from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.memory.manager import MemoryManager
from pgm_engine.policy.differentiable import DifferentiableSubgraphSelector
from pgm_engine.policy.gnn import DownstreamTaskGNN
from pgm_engine.policy.saliency import SaliencyPruningTracker
from pgm_engine.routing.a_star import AStarValueRouter
from pgm_engine.routing.bidirectional import BidirectionalFrontierRouter
from pgm_engine.routing.corridor import CorridorMaterializer
from pgm_engine.routing.landmark import GlobalLandmarkGraph
from pgm_engine.topology.guard import TopologyGuard
from pgm_engine.types import EdgeRelation, EntityNode, PathResult


class ProceduralGraphEngine:
    """Master Procedural Graph Materialization Engine."""

    def __init__(
        self,
        max_nodes_k: int = 2048,
        max_edges_m: int = 16384,
        node_dim: int = 128,
        rel_dim: int = 32,
        tau_push: float = 0.75,
        tau_pop: float = 0.25,
        t_cooldown: int = 5,
        num_landmark_clusters: int = 16,
        latent_epsilon: float = 0.25,
        substrate: Optional[L3LatentSubstrate] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        if substrate is not None:
            self.node_dim = substrate.node_dim
            self.rel_dim = substrate.rel_dim
        else:
            self.node_dim = node_dim
            self.rel_dim = rel_dim
        self.max_nodes_k = max_nodes_k
        self.max_edges_m = max_edges_m

        # Tiered Memory Subsystem
        self.l1_buffer = L1TensorBuffer(
            max_nodes=max_nodes_k,
            max_edges=max_edges_m,
            node_dim=self.node_dim,
            rel_dim=self.rel_dim,
            device=self.device,
        )
        self.l2_graph = L2HostGraph()
        self.l3_substrate = substrate or L3LatentSubstrate(node_dim=self.node_dim, rel_dim=self.rel_dim)

        self.hysteresis_config = HysteresisConfig(
            tau_push=tau_push,
            tau_pop=tau_pop,
            t_cooldown=t_cooldown,
        )
        self.memory_manager = MemoryManager(
            l1_buffer=self.l1_buffer,
            l2_graph=self.l2_graph,
            l3_substrate=self.l3_substrate,
            hysteresis_config=self.hysteresis_config,
        )

        # Routing & Landmark Subsystem
        self.landmark_graph = GlobalLandmarkGraph(
            num_clusters=num_landmark_clusters,
            node_dim=self.node_dim,
        )
        self.a_star_router = AStarValueRouter()
        self.corridor_materializer = CorridorMaterializer(
            memory_manager=self.memory_manager,
            landmark_graph=self.landmark_graph,
            latent_epsilon=latent_epsilon,
        )

        # Topological Guards & Contraction
        self.topology_guard = TopologyGuard(rel_dim=self.rel_dim)

        # Policy & Differentiable Modules
        self.differentiable_selector = DifferentiableSubgraphSelector(
            node_dim=self.node_dim,
            rel_dim=self.rel_dim,
        ).to(self.device)
        self.saliency_tracker = SaliencyPruningTracker()
        self.gnn = DownstreamTaskGNN(
            node_dim=self.node_dim,
            hidden_dim=self.node_dim,
            rel_dim=self.rel_dim,
        ).to(self.device)

        self.iteration = 0

    def set_substrate(self, substrate: L3LatentSubstrate) -> None:
        """Sets backing substrate and updates memory manager reference."""
        self.l3_substrate = substrate
        self.memory_manager.l3 = substrate

    def initialize_landmarks(self) -> None:
        """Builds coarse landmark clusters from current L3 substrate entities."""
        self.landmark_graph.build_from_substrate(self.l3_substrate)

    def query_reasoning_corridor(
        self,
        source_id: str,
        target_id: str,
        use_landmarks: bool = True,
        max_iterations: int = 500,
    ) -> PathResult:
        """Finds multi-hop reasoning corridor between source and target entities.

        Overcomes bridge entity blindness via coarse landmark scaffolding and bidirectional A*.
        """
        self.iteration += 1
        self.memory_manager.advance_iteration()
        self.saliency_tracker.step()

        start_time = time.perf_counter()
        result = self.corridor_materializer.materialize_corridor(
            source_id=source_id,
            target_id=target_id,
            use_landmarks=use_landmarks,
            max_iterations=max_iterations,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        for nid in result.path_nodes:
            self.saliency_tracker.register_node(nid)

        result.peak_l1_nodes = self.l1_buffer.peak_nodes
        result.peak_l1_edges = self.l1_buffer.peak_edges
        return result

    def prune_and_contract(
        self,
        target_capacity: Optional[int] = None,
        use_saliency: bool = True,
    ) -> List[EdgeRelation]:
        """Executes articulation-shielded eviction with virtual shortcut contraction.

        Evicts low-utility or low-saliency nodes when working set approaches capacity.
        If an evicted node is an articulation point, synthesizes virtual edges before removal.

        Args:
            target_capacity: Node count threshold to prune down to. Defaults to 80% of K.
            use_saliency: Whether to consider gradient influence.

        Returns:
            List of newly synthesized virtual shortcut edges.
        """
        limit = target_capacity or int(self.max_nodes_k * 0.8)
        active_nodes = set(self.l1_buffer.entity_to_slot.keys())
        if len(active_nodes) <= limit:
            return []

        # Find candidates for eviction
        # Check nodes with low utility (hysteresis pop condition)
        candidates: List[tuple[str, float]] = []
        for eid in active_nodes:
            node = self.l2_graph.get_node(eid)
            u = node.utility if node else 0.0
            if self.memory_manager.hysteresis.should_pop(u):
                candidates.append((eid, u))

        # If saliency enabled, supplement with low-influence nodes
        if use_saliency and len(candidates) < (len(active_nodes) - limit):
            low_infl = self.saliency_tracker.get_low_influence_candidates(active_nodes, min_age_steps=0)
            for eid, infl in low_infl:
                if eid not in [c[0] for c in candidates]:
                    candidates.append((eid, infl))

        # If still needing candidates to satisfy target memory budget, rank remaining active nodes by utility
        if len(candidates) < (len(active_nodes) - limit):
            active_sorted = sorted(
                active_nodes,
                key=lambda x: self.l2_graph.get_node(x).utility if self.l2_graph.get_node(x) else 0.0,
            )
            for eid in active_sorted:
                if eid not in [c[0] for c in candidates]:
                    u = self.l2_graph.get_node(eid).utility if self.l2_graph.get_node(eid) else 0.0
                    candidates.append((eid, u))

        # Sort candidates (lowest utility/influence first)
        candidates.sort(key=lambda x: x[1])

        all_synthesized: List[EdgeRelation] = []

        for eid, _ in candidates:
            if len(self.l1_buffer.entity_to_slot) <= limit:
                break

            permitted, synth_edges = self.topology_guard.safe_prune_or_contract(
                node_id=eid,
                active_nodes=set(self.l1_buffer.entity_to_slot.keys()),
                l2_graph=self.l2_graph,
                l1_buffer=self.l1_buffer,
                force_contraction=True,
            )

            if permitted:
                self.memory_manager.evict_from_l1(eid)
                self.saliency_tracker.deregister_node(eid)
                all_synthesized.extend(synth_edges)
                self.memory_manager.stats.contractions_synthesized += len(synth_edges)

        return all_synthesized

    def forward_gnn_step(
        self,
        source_id: str,
        target_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs downstream GNN forward pass on active L1 tensors and tracks gradient saliency.

        Returns:
            task_score: Predicted scalar logit for source-target connection.
            final_X: Compact node representation matrix.
        """
        X_act, active_ids, compact_id_map = self.l1_buffer.get_active_features()
        if len(active_ids) == 0:
            raise RuntimeError("Cannot execute GNN pass with 0 active L1 nodes.")

        # Ensure endpoints are active
        if source_id not in compact_id_map or target_id not in compact_id_map:
            raise KeyError(f"Endpoints {source_id} and {target_id} must be in L1")

        src_idx = compact_id_map[source_id]
        tgt_idx = compact_id_map[target_id]

        E_act, W_act, R_act = self.l1_buffer.get_active_adjacency_coo()

        # Map edge_index from L1 global slots to compact indices
        if E_act.size(1) > 0:
            src_slots = E_act[0].cpu().tolist()
            tgt_slots = E_act[1].cpu().tolist()

            compact_src = []
            compact_tgt = []
            valid_edge_mask = []
            for i, (s_slot, t_slot) in enumerate(zip(src_slots, tgt_slots)):
                s_eid = self.l1_buffer.slot_to_entity.get(s_slot)
                t_eid = self.l1_buffer.slot_to_entity.get(t_slot)
                if s_eid in compact_id_map and t_eid in compact_id_map:
                    compact_src.append(compact_id_map[s_eid])
                    compact_tgt.append(compact_id_map[t_eid])
                    valid_edge_mask.append(i)

            if compact_src:
                compact_E = torch.tensor([compact_src, compact_tgt], dtype=torch.long, device=self.device)
                valid_indices = torch.tensor(valid_edge_mask, dtype=torch.long, device=self.device)
                compact_W = W_act[valid_indices]
                compact_R = R_act[valid_indices]
            else:
                compact_E = torch.empty((2, 0), dtype=torch.long, device=self.device)
                compact_W = torch.empty((0,), dtype=torch.float32, device=self.device)
                compact_R = torch.empty((0, self.rel_dim), dtype=torch.float32, device=self.device)
        else:
            compact_E = E_act
            compact_W = W_act
            compact_R = R_act

        X_act = X_act.clone().detach().requires_grad_(True)

        task_score, final_X = self.gnn(
            X=X_act,
            edge_index=compact_E,
            edge_weight=compact_W,
            edge_rel=compact_R,
            source_idx=src_idx,
            target_idx=tgt_idx,
        )

        return task_score, X_act

    def backprop_saliency(
        self,
        task_loss: torch.Tensor,
        X_act: torch.Tensor,
    ) -> None:
        """Backpropagates loss to extract gradient sensitivity for autonomous pruning."""
        if X_act.grad is not None:
            X_act.grad.zero_()

        task_loss.backward(retain_graph=True)
        _, active_ids, compact_id_map = self.l1_buffer.get_active_features()
        self.saliency_tracker.record_tensor_gradients(compact_id_map, X_act.grad)

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides full operational telemetry of the materialization engine."""
        stats = self.memory_manager.get_stats()
        return {
            "active_l1_nodes": stats.active_l1_nodes,
            "active_l1_edges": stats.active_l1_edges,
            "peak_l1_nodes": self.l1_buffer.peak_nodes,
            "peak_l1_edges": self.l1_buffer.peak_edges,
            "l2_total_nodes": stats.l2_total_nodes,
            "l2_total_edges": stats.l2_total_edges,
            "contractions_synthesized": stats.contractions_synthesized,
            "refractory_blocks": stats.refractory_blocks,
            "evictions_executed": stats.evictions_executed,
            "landmark_clusters": self.landmark_graph.num_clusters,
            "iteration": self.iteration,
        }
