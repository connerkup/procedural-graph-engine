"""Corridor-Guided Materialization: Unifies Landmark Scaffolding, A*, and Frontier Discovery."""

from __future__ import annotations

from typing import Optional, Set
import numpy as np

from pgm_engine.memory.manager import MemoryManager
from pgm_engine.routing.a_star import AStarValueRouter
from pgm_engine.routing.bidirectional import BidirectionalFrontierRouter
from pgm_engine.routing.landmark import GlobalLandmarkGraph
from pgm_engine.types import PathResult


class CorridorMaterializer:
    """End-to-end corridor routing and materialization pipeline."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        landmark_graph: GlobalLandmarkGraph,
        latent_epsilon: float = 0.25,
        heuristic_weight: float = 1.0,
    ):
        self.mem = memory_manager
        self.landmark = landmark_graph
        self.router = AStarValueRouter(heuristic_weight=heuristic_weight)
        self.bidirectional = BidirectionalFrontierRouter(
            a_star_router=self.router,
            latent_epsilon=latent_epsilon,
        )

    def materialize_corridor(
        self,
        source_id: str,
        target_id: str,
        use_landmarks: bool = True,
        max_iterations: int = 500,
    ) -> PathResult:
        """Finds reasoning corridor and materializes it into tiered memory.

        1. Computes coarse landmark corridor across G_coarse.
        2. Executes bidirectional A* lookahead search.
        3. Admits discovered corridor entities into L2 and L1 via hysteresis.
        """
        src_node = self.mem.l3.get_entity(source_id)
        tgt_node = self.mem.l3.get_entity(target_id)
        if not src_node or not tgt_node:
            raise ValueError(f"Entities {source_id} or {target_id} not found in L3 substrate")

        corridor: Optional[Set[int]] = None
        if use_landmarks and self.landmark.num_clusters > 0:
            corridor = self.landmark.compute_routing_corridor(
                source_emb=src_node.embedding,
                target_emb=tgt_node.embedding,
                allow_neighboring_clusters=True,
            )

        # Run Bidirectional Frontier Expansion
        res = self.bidirectional.search(
            source_id=source_id,
            target_id=target_id,
            l2_graph=self.mem.l2,
            substrate=self.mem.l3,
            corridor=corridor,
            max_iterations=max_iterations,
        )

        if not res.success:
            return res

        # Materialize discovered path into L2 and promote high-utility nodes to L1
        for node_id in res.path_nodes:
            self.mem.prefetch_from_l3(node_id)
            node = self.mem.l2.get_node(node_id)
            if node:
                h = self.router.compute_heuristic(node.embedding, tgt_node.embedding)
                direct_utility = self.router.compute_utility(g_cost=0.1, h_cost=h)
                # Intermediate bridge nodes on the active corridor are protected from early rejection
                corridor_boost = max(0.85, self.mem.hysteresis.config.tau_push + 0.10)
                utility = max(direct_utility, corridor_boost)
                self.mem.push_to_l1(node_id, utility=utility)

        for edge in res.path_edges:
            self.mem.l2.add_edge(edge)
            if edge.source_id in self.mem.l1.entity_to_slot and edge.target_id in self.mem.l1.entity_to_slot:
                if not self.mem.l1.is_edge_full:
                    self.mem.l1.allocate_edge(
                        src_id=edge.source_id,
                        tgt_id=edge.target_id,
                        relation=edge.relation,
                        weight=edge.weight,
                        rel_embedding=edge.relation_embedding,
                    )

        res.peak_l1_nodes = self.mem.l1.peak_nodes
        res.peak_l1_edges = self.mem.l1.peak_edges
        return res
