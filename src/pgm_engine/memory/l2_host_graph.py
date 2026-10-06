"""L2 Working Graph Cache: High-concurrency host RAM graph engine.

Maintains mutable adjacency representation, edge provenance for synthesized shortcuts,
temporal decay statistics, and candidate neighbor frontier inverted index.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from pgm_engine.types import EdgeRelation, EntityNode


class L2HostGraph:
    """In-memory topological cache residing in host RAM.

    Bridges fast static L1 tensors with external L3 persistent storage.
    Tracks structural invariants, provenance chains, and neighbor frontiers.
    """

    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate
        self.nodes: Dict[str, EntityNode] = {}
        # Forward adjacency: src -> dict of {tgt: dict of {relation: EdgeRelation}}
        self.out_edges: Dict[str, Dict[str, Dict[str, EdgeRelation]]] = defaultdict(lambda: defaultdict(dict))
        # Inward adjacency: tgt -> dict of {src: dict of {relation: EdgeRelation}}
        self.in_edges: Dict[str, Dict[str, Dict[str, EdgeRelation]]] = defaultdict(lambda: defaultdict(dict))

        # Inverted index of candidate unmaterialized neighbors: node_id -> list of candidate node_ids
        self.candidate_frontier: Dict[str, Set[str]] = defaultdict(set)

        # Global step counter for temporal decay
        self.current_step: int = 0

    def add_node(self, node: EntityNode) -> None:
        """Adds or updates an entity node in L2."""
        node.last_accessed_step = self.current_step
        self.nodes[node.entity_id] = node

    def get_node(self, entity_id: str) -> Optional[EntityNode]:
        node = self.nodes.get(entity_id)
        if node:
            node.last_accessed_step = self.current_step
        return node

    def has_node(self, entity_id: str) -> bool:
        return entity_id in self.nodes

    def add_edge(self, edge: EdgeRelation) -> None:
        """Adds an edge to L2 host graph."""
        self.out_edges[edge.source_id][edge.target_id][edge.relation] = edge
        self.in_edges[edge.target_id][edge.source_id][edge.relation] = edge

    def remove_edge(self, source_id: str, target_id: str, relation: str) -> Optional[EdgeRelation]:
        """Removes a specific edge from L2."""
        edge = self.out_edges.get(source_id, {}).get(target_id, {}).pop(relation, None)
        self.in_edges.get(target_id, {}).get(source_id, {}).pop(relation, None)
        return edge

    def remove_node(self, entity_id: str) -> Optional[EntityNode]:
        """Removes a node and all its direct edges from L2."""
        node = self.nodes.pop(entity_id, None)
        if not node:
            return None

        # Remove outgoing edges
        out_targets = list(self.out_edges.get(entity_id, {}).keys())
        for tgt in out_targets:
            for rel in list(self.out_edges[entity_id][tgt].keys()):
                self.in_edges[tgt][entity_id].pop(rel, None)
        self.out_edges.pop(entity_id, None)

        # Remove incoming edges
        in_sources = list(self.in_edges.get(entity_id, {}).keys())
        for src in in_sources:
            for rel in list(self.in_edges[entity_id][src].keys()):
                self.out_edges[src][entity_id].pop(rel, None)
        self.in_edges.pop(entity_id, None)

        self.candidate_frontier.pop(entity_id, None)
        return node

    def get_neighbors(self, entity_id: str, direction: str = "out") -> List[str]:
        """Returns direct neighbor entity IDs in specified direction ('out', 'in', or 'both')."""
        if direction == "out":
            return list(self.out_edges.get(entity_id, {}).keys())
        elif direction == "in":
            return list(self.in_edges.get(entity_id, {}).keys())
        elif direction == "both":
            out_nbrs = set(self.out_edges.get(entity_id, {}).keys())
            in_nbrs = set(self.in_edges.get(entity_id, {}).keys())
            return list(out_nbrs | in_nbrs)
        else:
            raise ValueError(f"Unknown direction '{direction}'")

    def get_all_edges_for_node(self, entity_id: str) -> List[EdgeRelation]:
        """Retrieves all incident edges (in and out) for an entity."""
        edges: List[EdgeRelation] = []
        for tgt_map in self.out_edges.get(entity_id, {}).values():
            edges.extend(tgt_map.values())
        for src_map in self.in_edges.get(entity_id, {}).values():
            edges.extend(src_map.values())
        return edges

    def register_candidate_neighbors(self, source_id: str, candidate_ids: List[str]) -> None:
        """Registers unmaterialized candidate frontier nodes for asynchronous pre-fetching."""
        for cid in candidate_ids:
            if cid not in self.nodes:
                self.candidate_frontier[source_id].add(cid)

    def pop_candidate_frontier(self, source_id: str) -> List[str]:
        candidates = list(self.candidate_frontier.get(source_id, set()))
        self.candidate_frontier[source_id].clear()
        return candidates

    def apply_temporal_decay(self, decay_rate: Optional[float] = None) -> None:
        """Applies exponential temporal decay to edge weights based on time since access."""
        rate = decay_rate if decay_rate is not None else self.decay_rate
        self.current_step += 1
        for u, tgt_map in self.out_edges.items():
            for v, rel_map in tgt_map.items():
                for rel, edge in rel_map.items():
                    # Virtual shortcut edges decay faster to encourage path pruning when not reinforced
                    factor = (1.0 - rate * (1.5 if edge.is_virtual else 1.0))
                    edge.weight = max(1e-4, edge.weight * factor)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_edges(self) -> int:
        count = 0
        for tgt_map in self.out_edges.values():
            for rel_map in tgt_map.values():
                count += len(rel_map)
        return count
