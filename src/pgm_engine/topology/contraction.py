"""Virtual Shortcut Synthesis & Topological Edge Contraction.

Preserves multi-hop reachability and relational chains across GNNs and LLM reasoning
engines by collapsing evicted bridge nodes into weighted composite virtual edges.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.types import EdgeRelation


class EdgeContractionOperator:
    """Executes transitive contractions when a bridge node is pruned."""

    def __init__(self, rel_dim: int = 32):
        self.rel_dim = rel_dim

    def synthesize_composite_relation_embedding(
        self,
        emb1: Optional[np.ndarray],
        emb2: Optional[np.ndarray],
    ) -> np.ndarray:
        """Computes composite relation representation via normalized bilinear Hadamard projection."""
        if emb1 is None and emb2 is None:
            composite = np.zeros(self.rel_dim, dtype=np.float32)
            composite[0] = 1.0
            return composite
        elif emb1 is None:
            return emb2.copy() if emb2 is not None else np.zeros(self.rel_dim, dtype=np.float32)
        elif emb2 is None:
            return emb1.copy()

        # Bilinear combination + circular correlation/Hadamard product
        composite = (emb1 + emb2) * 0.5 + (emb1 * emb2) * 0.5
        norm = np.linalg.norm(composite)
        if norm > 1e-9:
            composite = composite / norm
        return composite.astype(np.float32)

    def contract_node(
        self,
        node_id: str,
        l2_graph: L2HostGraph,
        l1_buffer: Optional[L1TensorBuffer] = None,
        weight_decay_factor: float = 0.95,
    ) -> List[EdgeRelation]:
        """Synthesizes virtual transitive shortcuts across all 2-hop paths through `node_id`.

        Original Subgraph:
            (u) ──(w_uv, r1)──► [node_id] ──(w_vw, r2)──► (w)

        Synthesized Virtual Shortcut:
            (u) ────────────(w_uw = w_uv * w_vw, r_trans)────────────► (w)

        Args:
            node_id: The bridge entity being evicted.
            l2_graph: Active L2 host graph.
            l1_buffer: Optional active L1 tensor buffer for slot-level synchronization.
            weight_decay_factor: Multiplier to penalize multi-hop traversal cost.

        Returns:
            List of newly synthesized EdgeRelation objects.
        """
        # Collect incoming and outgoing edges for node_id
        in_edges_map = l2_graph.in_edges.get(node_id, {})
        out_edges_map = l2_graph.out_edges.get(node_id, {})

        incoming_edges: List[EdgeRelation] = []
        for src, rel_map in in_edges_map.items():
            incoming_edges.extend(rel_map.values())

        outgoing_edges: List[EdgeRelation] = []
        for tgt, rel_map in out_edges_map.items():
            outgoing_edges.extend(rel_map.values())

        synthesized_edges: List[EdgeRelation] = []

        for in_edge in incoming_edges:
            u = in_edge.source_id
            for out_edge in outgoing_edges:
                w = out_edge.target_id
                if u == w:
                    # Do not synthesize trivial self-loops
                    continue

                contracted_weight = in_edge.weight * out_edge.weight * weight_decay_factor
                composite_rel_type = f"virtual_{in_edge.relation}_{out_edge.relation}"
                composite_emb = self.synthesize_composite_relation_embedding(
                    in_edge.relation_embedding,
                    out_edge.relation_embedding,
                )

                # Merge provenance chains
                provenance_chain = list(in_edge.provenance) + [node_id] + list(out_edge.provenance)

                virtual_edge = EdgeRelation(
                    source_id=u,
                    target_id=w,
                    relation=composite_rel_type,
                    weight=contracted_weight,
                    is_virtual=True,
                    hop_count=in_edge.hop_count + out_edge.hop_count,
                    provenance=provenance_chain,
                    relation_embedding=composite_emb,
                    metadata={"contracted_through": node_id},
                )

                l2_graph.add_edge(virtual_edge)
                synthesized_edges.append(virtual_edge)

                # If both endpoints are currently active in L1, allocate the shortcut in L1
                if l1_buffer is not None:
                    if u in l1_buffer.entity_to_slot and w in l1_buffer.entity_to_slot:
                        if not l1_buffer.is_edge_full:
                            l1_buffer.allocate_edge(
                                src_id=u,
                                tgt_id=w,
                                relation=virtual_edge.relation,
                                weight=virtual_edge.weight,
                                rel_embedding=virtual_edge.relation_embedding,
                            )

        return synthesized_edges
