"""Baseline Implementations for Comparative Benchmarking.

Includes standard Static k-Hop Greedy Nearest-Neighbor Expansion, representing typical
vector-RAG and greedy GNN retrieval pipelines vulnerable to bridge-entity blindness.
"""

from __future__ import annotations

import time
from typing import List, Set, Tuple
import numpy as np

from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.types import PathResult


class StaticKHopGreedyBaseline:
    """Standard k-hop greedy search expanding top-k entities by direct cosine similarity."""

    def __init__(self, top_k_beam: int = 3, max_hops: int = 6):
        self.top_k_beam = top_k_beam
        self.max_hops = max_hops

    def search(
        self,
        source_id: str,
        target_id: str,
        substrate: L3LatentSubstrate,
    ) -> PathResult:
        """Executes static greedy expansion towards target vector."""
        start_time = time.perf_counter()

        src_node = substrate.get_entity(source_id)
        tgt_node = substrate.get_entity(target_id)
        if not src_node or not tgt_node:
            return PathResult(
                success=False,
                path_nodes=[],
                path_edges=[],
                total_cost=float("inf"),
                iterations=0,
                expanded_nodes=0,
                virtual_shortcuts_traversed=0,
                peak_l1_nodes=0,
                peak_l1_edges=0,
            )

        # Beam: list of (current_node_id, path_so_far, score)
        beam: List[Tuple[str, List[str], float]] = [(source_id, [source_id], 0.0)]
        visited: Set[str] = {source_id}
        expanded_nodes = 0
        success = False
        winning_path: List[str] = []

        for hop in range(self.max_hops):
            candidates = []
            for curr_id, curr_path, curr_score in beam:
                expanded_nodes += 1
                neighbors = substrate.expand_procedural_neighbors(curr_id, direction="out")

                for nbr_node, edge in neighbors:
                    nid = nbr_node.entity_id
                    if nid == target_id:
                        success = True
                        winning_path = curr_path + [target_id]
                        break

                    if nid not in visited:
                        visited.add(nid)
                        # Static greedy relevance score = cosine similarity to target
                        cos_sim = float(np.dot(nbr_node.embedding, tgt_node.embedding))
                        candidates.append((nid, curr_path + [nid], cos_sim))

                if success:
                    break

            if success:
                break

            if not candidates:
                break

            # Sort greedily by cosine similarity descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[: self.top_k_beam]

        return PathResult(
            success=success,
            path_nodes=winning_path,
            path_edges=[],
            total_cost=0.0,
            iterations=hop + 1,
            expanded_nodes=expanded_nodes,
            virtual_shortcuts_traversed=0,
            peak_l1_nodes=len(visited),
            peak_l1_edges=0,
        )
