"""Core data types and specifications for the Procedural Graph Materialization Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch


@dataclass
class EntityNode:
    """Represents a node in the graph knowledge substrate."""
    entity_id: str
    embedding: np.ndarray  # Shape: (d,)
    cluster_id: int = -1
    utility: float = 1.0
    creation_step: int = 0
    last_accessed_step: int = 0
    eviction_count: int = 0
    refractory_until: int = 0
    l1_slot: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_torch(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.from_numpy(self.embedding).float().to(device)


@dataclass
class EdgeRelation:
    """Represents a direct or contracted virtual relationship."""
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    is_virtual: bool = False
    hop_count: int = 1
    provenance: List[str] = field(default_factory=list)  # chain of collapsed bridge node IDs
    relation_embedding: Optional[np.ndarray] = None  # Shape: (d_r,)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.source_id, self.target_id, self.relation)


@dataclass
class BufferStats:
    """Telemetry counters for tiered memory and graph operations."""
    active_l1_nodes: int = 0
    active_l1_edges: int = 0
    free_l1_node_slots: int = 0
    free_l1_edge_slots: int = 0
    l2_total_nodes: int = 0
    l2_total_edges: int = 0
    contractions_synthesized: int = 0
    articulation_shields_triggered: int = 0
    evictions_executed: int = 0
    refractory_blocks: int = 0


@dataclass
class PathResult:
    """Represents the discovery result of a multi-hop reasoning search."""
    success: bool
    path_nodes: List[str]
    path_edges: List[EdgeRelation]
    total_cost: float
    iterations: int
    expanded_nodes: int
    virtual_shortcuts_traversed: int
    peak_l1_nodes: int
    peak_l1_edges: int
