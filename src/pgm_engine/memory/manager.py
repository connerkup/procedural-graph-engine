"""Tiered Memory Manager: Coordinates L1 (VRAM/Tensors), L2 (Host RAM), and L3 (Substrate)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from pgm_engine.memory.hysteresis import HysteresisConfig, HysteresisController
from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.types import BufferStats, EdgeRelation, EntityNode


class MemoryManager:
    """Orchestrates multi-tiered storage and dynamic buffer paging."""

    def __init__(
        self,
        l1_buffer: L1TensorBuffer,
        l2_graph: L2HostGraph,
        l3_substrate: L3LatentSubstrate,
        hysteresis_config: Optional[HysteresisConfig] = None,
    ):
        self.l1 = l1_buffer
        self.l2 = l2_graph
        self.l3 = l3_substrate
        self.hysteresis = HysteresisController(hysteresis_config)

        self.stats = BufferStats()

    def advance_iteration(self) -> None:
        """Ticks the global iteration counter across memory tiers."""
        self.hysteresis.step()
        self.l2.apply_temporal_decay()

    def prefetch_from_l3(self, entity_id: str) -> Optional[EntityNode]:
        """Loads an entity from persistent substrate L3 into host cache L2."""
        if self.l2.has_node(entity_id):
            return self.l2.get_node(entity_id)

        node = self.l3.get_entity(entity_id)
        if node:
            self.l2.add_node(node)
            # Also prefetch its immediate edges into L2
            neighbors = self.l3.expand_procedural_neighbors(entity_id, direction="both")
            for neighbor_node, edge in neighbors:
                self.l2.add_edge(edge)
                if not self.l2.has_node(neighbor_node.entity_id):
                    self.l2.register_candidate_neighbors(entity_id, [neighbor_node.entity_id])
        return node

    def can_push_to_l1(self, entity_id: str, utility: float) -> bool:
        """Checks admission eligibility against hysteresis and refractory rules."""
        return self.hysteresis.should_push(entity_id, utility)

    def push_to_l1(self, entity_id: str, utility: float) -> bool:
        """Promotes an entity from L2/L3 into the L1 active tensor buffer.

        Returns:
            True if admitted to L1, False if rejected by hysteresis or capacity.
        """
        if not self.can_push_to_l1(entity_id, utility):
            self.stats.refractory_blocks += 1
            return False

        # Ensure entity is in L2
        node = self.l2.get_node(entity_id)
        if not node:
            node = self.prefetch_from_l3(entity_id)
            if not node:
                return False

        node.utility = utility

        # If already in L1, update utility and return
        if entity_id in self.l1.entity_to_slot:
            return True

        # Check L1 node capacity
        if self.l1.is_node_full:
            return False

        # Allocate slot in L1
        slot = self.l1.allocate_node(entity_id, node.embedding)
        node.l1_slot = slot

        # Check for incident edges whose other endpoint is already in L1
        # and promote them to L1 edge slots
        for edge in self.l2.get_all_edges_for_node(entity_id):
            other_id = edge.target_id if edge.source_id == entity_id else edge.source_id
            if other_id in self.l1.entity_to_slot and not self.l1.is_edge_full:
                remb = edge.relation_embedding
                self.l1.allocate_edge(
                    src_id=edge.source_id,
                    tgt_id=edge.target_id,
                    relation=edge.relation,
                    weight=edge.weight,
                    rel_embedding=remb,
                )

        return True

    def evict_from_l1(self, entity_id: str) -> bool:
        """Evicts a node from L1 back into L2 cache, starting refractory quarantine."""
        if entity_id not in self.l1.entity_to_slot:
            return False

        self.l1.free_node(entity_id)
        node = self.l2.get_node(entity_id)
        if node:
            node.l1_slot = None

        self.hysteresis.record_eviction(entity_id)
        self.stats.evictions_executed += 1
        return True

    def get_stats(self) -> BufferStats:
        """Returns snapshot telemetry."""
        self.stats.active_l1_nodes = self.l1.num_active_nodes
        self.stats.active_l1_edges = self.l1.num_active_edges
        self.stats.free_l1_node_slots = len(self.l1._free_node_slots)
        self.stats.free_l1_edge_slots = len(self.l1._free_edge_slots)
        self.stats.l2_total_nodes = self.l2.total_nodes
        self.stats.l2_total_edges = self.l2.total_edges
        return self.stats
