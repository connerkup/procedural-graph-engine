"""Topological Guard: Manages structural shielding and safe eviction sequencing."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.topology.articulation import find_articulation_points, is_local_articulation_point
from pgm_engine.topology.contraction import EdgeContractionOperator
from pgm_engine.types import EdgeRelation


class TopologyGuard:
    """Safeguards graph biconnectivity and message-passing topology during dynamic pruning."""

    def __init__(self, rel_dim: int = 32):
        self.contraction_op = EdgeContractionOperator(rel_dim=rel_dim)
        self.shielded_nodes: Set[str] = set()

    def identify_shielded_nodes(
        self,
        active_nodes: Set[str],
        l2_graph: L2HostGraph,
    ) -> Set[str]:
        """Detects all articulation points across active nodes and marks them shielded."""
        def get_active_neighbors(node_id: str) -> List[str]:
            return [n for n in l2_graph.get_neighbors(node_id, direction="both") if n in active_nodes]

        self.shielded_nodes = find_articulation_points(active_nodes, get_active_neighbors)
        return self.shielded_nodes

    def is_shielded(
        self,
        node_id: str,
        active_nodes: Set[str],
        l2_graph: L2HostGraph,
        use_local_test: bool = True,
    ) -> bool:
        """Determines whether a specific node is currently protected by structural shielding."""
        if use_local_test:
            def get_active_neighbors(nid: str) -> List[str]:
                return [n for n in l2_graph.get_neighbors(nid, direction="both") if n in active_nodes]

            return is_local_articulation_point(node_id, active_nodes, get_active_neighbors)

        if not self.shielded_nodes:
            self.identify_shielded_nodes(active_nodes, l2_graph)
        return node_id in self.shielded_nodes

    def safe_prune_or_contract(
        self,
        node_id: str,
        active_nodes: Set[str],
        l2_graph: L2HostGraph,
        l1_buffer: Optional[L1TensorBuffer] = None,
        force_contraction: bool = True,
    ) -> Tuple[bool, List[EdgeRelation]]:
        """Safely prunes or contracts a node.

        If node is an articulation point:
            - If force_contraction is True: Synthesize virtual transitive edges first,
              guaranteeing reachability before freeing node slot.
            - If force_contraction is False: Eviction is blocked by structural shielding.

        Returns:
            Tuple of (eviction_permitted: bool, synthesized_edges: list).
        """
        is_cut = self.is_shielded(node_id, active_nodes, l2_graph, use_local_test=True)

        synthesized_edges: List[EdgeRelation] = []

        if is_cut:
            if not force_contraction:
                # Structural shielding vetoes eviction
                return False, []

            # Contract node to preserve topological reachability
            synthesized_edges = self.contraction_op.contract_node(
                node_id=node_id,
                l2_graph=l2_graph,
                l1_buffer=l1_buffer,
            )

        # Evict from L1 if buffer provided
        if l1_buffer is not None and node_id in l1_buffer.entity_to_slot:
            l1_buffer.free_node(node_id)

        # If node was shielded, remove from shielded cache
        self.shielded_nodes.discard(node_id)

        return True, synthesized_edges
