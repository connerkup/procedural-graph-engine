"""L1 Working Tensor Buffer: High-throughput, zero-allocation static tensor substrate.

Pre-allocates bounded static tensors (X in R^{K x d}, E in R^{2 x M}) with internal
free-index pools to completely prevent GPU/CPU memory reallocation during procedural
materialization and dynamic pruning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch


class L1TensorBuffer:
    """Pre-allocated static tensor buffer with LIFO free-list recycling.

    Attributes:
        max_nodes (K): Maximum node capacity (default 2048).
        max_edges (M): Maximum edge capacity (default 16384).
        node_dim (d): Dimensionality of node embeddings.
        rel_dim (d_r): Dimensionality of relation embeddings.
        device: PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        max_nodes: int = 2048,
        max_edges: int = 16384,
        node_dim: int = 128,
        rel_dim: int = 32,
        device: Optional[torch.device] = None,
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_dim = node_dim
        self.rel_dim = rel_dim
        self.device = device or torch.device("cpu")

        # Static tensor allocations (never resized at runtime)
        self.X = torch.zeros((max_nodes, node_dim), dtype=torch.float32, device=self.device)
        self.E = torch.zeros((2, max_edges), dtype=torch.long, device=self.device)
        self.W = torch.zeros(max_edges, dtype=torch.float32, device=self.device)
        self.R = torch.zeros((max_edges, rel_dim), dtype=torch.float32, device=self.device)

        # Boolean occupancy masks
        self.node_mask = torch.zeros(max_nodes, dtype=torch.bool, device=self.device)
        self.edge_mask = torch.zeros(max_edges, dtype=torch.bool, device=self.device)

        # Free-index stacks (LIFO) for O(1) slot allocation and recycling
        self._free_node_slots: List[int] = list(range(max_nodes - 1, -1, -1))
        self._free_edge_slots: List[int] = list(range(max_edges - 1, -1, -1))

        # Bidirectional ID <-> Slot mapping
        self.entity_to_slot: Dict[str, int] = {}
        self.slot_to_entity: Dict[int, str] = {}

        # Edge key: (src_id, tgt_id, relation_key) -> edge_slot
        self.edge_to_slot: Dict[Tuple[str, str, str], int] = {}
        self.slot_to_edge: Dict[int, Tuple[str, str, str]] = {}

        # Incident edge tracking for fast cascade eviction: slot -> set of edge_slots
        self.node_incident_edges: Dict[int, Set[int]] = {i: set() for i in range(max_nodes)}

        # Peak usage metrics
        self.peak_nodes = 0
        self.peak_edges = 0

    @property
    def num_active_nodes(self) -> int:
        return len(self.entity_to_slot)

    @property
    def num_active_edges(self) -> int:
        return len(self.edge_to_slot)

    @property
    def is_node_full(self) -> bool:
        return len(self._free_node_slots) == 0

    @property
    def is_edge_full(self) -> bool:
        return len(self._free_edge_slots) == 0

    def allocate_node(self, entity_id: str, embedding: torch.Tensor | np.ndarray) -> int:
        """Allocates an L1 tensor slot for an entity.

        Args:
            entity_id: Unique string identifier.
            embedding: Vector of shape (node_dim,).

        Returns:
            The allocated slot index in [0, max_nodes - 1].

        Raises:
            RuntimeError: If L1 node capacity K is exhausted.
        """
        if entity_id in self.entity_to_slot:
            slot = self.entity_to_slot[entity_id]
            # In-place update of embedding
            if isinstance(embedding, np.ndarray):
                emb_tensor = torch.from_numpy(embedding).float().to(self.device)
            else:
                emb_tensor = embedding.to(self.device)
            self.X[slot] = emb_tensor
            return slot

        if not self._free_node_slots:
            raise RuntimeError(
                f"L1 Node Capacity Exhausted ({self.max_nodes}). Eviction/Contraction required before insertion."
            )

        slot = self._free_node_slots.pop()
        if isinstance(embedding, np.ndarray):
            emb_tensor = torch.from_numpy(embedding).float().to(self.device)
        else:
            emb_tensor = embedding.to(self.device)

        self.X[slot] = emb_tensor
        self.node_mask[slot] = True
        self.entity_to_slot[entity_id] = slot
        self.slot_to_entity[slot] = entity_id
        self.node_incident_edges[slot].clear()

        if self.num_active_nodes > self.peak_nodes:
            self.peak_nodes = self.num_active_nodes

        return slot

    def allocate_edge(
        self,
        src_id: str,
        tgt_id: str,
        relation: str = "rel",
        weight: float = 1.0,
        rel_embedding: Optional[torch.Tensor | np.ndarray] = None,
    ) -> int:
        """Allocates an L1 edge slot connecting two active nodes.

        Args:
            src_id: Source entity ID (must already reside in L1).
            tgt_id: Target entity ID (must already reside in L1).
            relation: Relation label/key.
            weight: Edge weight scalar.
            rel_embedding: Optional relation vector of shape (rel_dim,).

        Returns:
            The allocated edge slot index in [0, max_edges - 1].
        """
        if src_id not in self.entity_to_slot or tgt_id not in self.entity_to_slot:
            raise KeyError(f"Both endpoints must be allocated in L1 ({src_id} -> {tgt_id})")

        u_slot = self.entity_to_slot[src_id]
        v_slot = self.entity_to_slot[tgt_id]
        edge_key = (src_id, tgt_id, relation)

        if edge_key in self.edge_to_slot:
            e_slot = self.edge_to_slot[edge_key]
            self.W[e_slot] = weight
            if rel_embedding is not None:
                if isinstance(rel_embedding, np.ndarray):
                    remb = torch.from_numpy(rel_embedding).float().to(self.device)
                else:
                    remb = rel_embedding.to(self.device)
                self.R[e_slot] = remb
            return e_slot

        if not self._free_edge_slots:
            raise RuntimeError(
                f"L1 Edge Capacity Exhausted ({self.max_edges}). Edge pruning required."
            )

        e_slot = self._free_edge_slots.pop()
        self.E[0, e_slot] = u_slot
        self.E[1, e_slot] = v_slot
        self.W[e_slot] = weight
        self.edge_mask[e_slot] = True

        if rel_embedding is not None:
            if isinstance(rel_embedding, np.ndarray):
                remb = torch.from_numpy(rel_embedding).float().to(self.device)
            else:
                remb = rel_embedding.to(self.device)
            self.R[e_slot] = remb
        else:
            self.R[e_slot].zero_()

        self.edge_to_slot[edge_key] = e_slot
        self.slot_to_edge[e_slot] = edge_key

        self.node_incident_edges[u_slot].add(e_slot)
        self.node_incident_edges[v_slot].add(e_slot)

        if self.num_active_edges > self.peak_edges:
            self.peak_edges = self.num_active_edges

        return e_slot

    def free_edge_by_slot(self, e_slot: int) -> Optional[Tuple[str, str, str]]:
        """Recycles an edge slot back to the free pool."""
        if not self.edge_mask[e_slot]:
            return None

        edge_key = self.slot_to_edge.pop(e_slot, None)
        if edge_key:
            self.edge_to_slot.pop(edge_key, None)

        u_slot = int(self.E[0, e_slot].item())
        v_slot = int(self.E[1, e_slot].item())

        self.node_incident_edges[u_slot].discard(e_slot)
        self.node_incident_edges[v_slot].discard(e_slot)

        # Zero tensors
        self.E[:, e_slot] = 0
        self.W[e_slot] = 0.0
        self.R[e_slot].zero_()
        self.edge_mask[e_slot] = False

        self._free_edge_slots.append(e_slot)
        return edge_key

    def free_node(self, entity_id: str) -> Optional[int]:
        """Evicts a node from L1 and recycles its slot and incident edges.

        Args:
            entity_id: The entity ID to remove.

        Returns:
            The freed slot index or None if node was not in L1.
        """
        if entity_id not in self.entity_to_slot:
            return None

        slot = self.entity_to_slot.pop(entity_id)
        self.slot_to_entity.pop(slot, None)

        # Cascade free all incident edges
        incident = list(self.node_incident_edges[slot])
        for e_slot in incident:
            self.free_edge_by_slot(e_slot)
        self.node_incident_edges[slot].clear()

        # Zero out node state
        self.X[slot].zero_()
        self.node_mask[slot] = False

        # Return slot to free pool
        self._free_node_slots.append(slot)
        return slot

    def get_active_adjacency_coo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns active (edge_index, edge_weight, edge_rel) without reallocation.

        Returns:
            edge_index: (2, num_active_edges)
            edge_weight: (num_active_edges,)
            edge_relation: (num_active_edges, rel_dim)
        """
        active_edge_indices = torch.nonzero(self.edge_mask, as_tuple=False).squeeze(-1)
        if active_edge_indices.numel() == 0:
            empty_e = torch.empty((2, 0), dtype=torch.long, device=self.device)
            empty_w = torch.empty((0,), dtype=torch.float32, device=self.device)
            empty_r = torch.empty((0, self.rel_dim), dtype=torch.float32, device=self.device)
            return empty_e, empty_w, empty_r

        return (
            self.E[:, active_edge_indices],
            self.W[active_edge_indices],
            self.R[active_edge_indices],
        )

    def get_active_features(self) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """Returns contiguous active feature matrix and ID mappings.

        Returns:
            X_active: Shape (num_active, node_dim)
            active_ids: List of entity_ids in row order
            id_to_compact_idx: Map from entity_id to index 0..num_active-1
        """
        active_ids = list(self.entity_to_slot.keys())
        if not active_ids:
            return (
                torch.empty((0, self.node_dim), dtype=torch.float32, device=self.device),
                [],
                {},
            )

        slots = [self.entity_to_slot[eid] for eid in active_ids]
        slots_tensor = torch.tensor(slots, dtype=torch.long, device=self.device)
        X_active = self.X[slots_tensor]
        id_to_compact = {eid: idx for idx, eid in enumerate(active_ids)}
        return X_active, active_ids, id_to_compact
