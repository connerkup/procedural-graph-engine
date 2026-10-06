"""L3 Persistent Latent Substrate: Persistent knowledge store and vector substrate.

Simulates or interfaces with external vector databases, Parquet stores, or dynamic
knowledge extraction pipelines (e.g. LLM extractors / document stores).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pgm_engine.types import EdgeRelation, EntityNode


class L3LatentSubstrate:
    """Persistent storage and extraction interface for the backing graph substrate.

    Manages latent embeddings, ground-truth relations, and dynamic entity expansions.
    """

    def __init__(self, node_dim: int = 128, rel_dim: int = 32):
        self.node_dim = node_dim
        self.rel_dim = rel_dim
        self.entities: Dict[str, EntityNode] = {}
        # Ground-truth adjacency in substrate: src -> dict of {tgt: list of EdgeRelation}
        self.adjacency: Dict[str, Dict[str, List[EdgeRelation]]] = {}

        # Matrix cache of all substrate embeddings for fast cosine similarity lookups
        self._cached_ids: List[str] = []
        self._cached_matrix: Optional[np.ndarray] = None
        self._dirty: bool = False

    def add_entity(
        self,
        entity_id: str,
        embedding: np.ndarray,
        cluster_id: int = -1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityNode:
        """Registers an entity in the persistent substrate."""
        assert embedding.shape[0] == self.node_dim, f"Embedding dim must match {self.node_dim}"
        # Normalize embedding for unit cosine sphere
        norm = np.linalg.norm(embedding)
        if norm > 1e-9:
            embedding = embedding / norm

        node = EntityNode(
            entity_id=entity_id,
            embedding=embedding,
            cluster_id=cluster_id,
            metadata=metadata or {},
        )
        self.entities[entity_id] = node
        if entity_id not in self.adjacency:
            self.adjacency[entity_id] = {}
        self._dirty = True
        return node

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str = "rel",
        weight: float = 1.0,
        rel_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EdgeRelation:
        """Adds a directional relation in the persistent substrate."""
        if rel_embedding is None:
            rel_embedding = np.random.randn(self.rel_dim).astype(np.float32)
            rel_embedding /= np.linalg.norm(rel_embedding) + 1e-9

        edge = EdgeRelation(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            relation_embedding=rel_embedding,
            metadata=metadata or {},
        )
        if source_id not in self.adjacency:
            self.adjacency[source_id] = {}
        if target_id not in self.adjacency[source_id]:
            self.adjacency[source_id][target_id] = []
        self.adjacency[source_id][target_id].append(edge)
        return edge

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        return self.entities.get(entity_id)

    def expand_procedural_neighbors(
        self,
        entity_id: str,
        direction: str = "out",
    ) -> List[Tuple[EntityNode, EdgeRelation]]:
        """Dynamically materializes 1-hop outgoing or incoming neighbors from substrate.

        Simulates calling a vector database or edge extraction service.
        """
        results: List[Tuple[EntityNode, EdgeRelation]] = []
        if direction in ("out", "both"):
            targets = self.adjacency.get(entity_id, {})
            for tgt_id, edge_list in targets.items():
                tgt_node = self.entities.get(tgt_id)
                if tgt_node:
                    for edge in edge_list:
                        results.append((tgt_node, edge))

        if direction in ("in", "both"):
            for src_id, targets in self.adjacency.items():
                if entity_id in targets:
                    src_node = self.entities.get(src_id)
                    if src_node:
                        for edge in targets[entity_id]:
                            results.append((src_node, edge))

        return results

    def _sync_matrix_cache(self) -> None:
        if self._dirty or self._cached_matrix is None:
            self._cached_ids = list(self.entities.keys())
            if self._cached_ids:
                embeddings = [self.entities[eid].embedding for eid in self._cached_ids]
                self._cached_matrix = np.vstack(embeddings)
            else:
                self._cached_matrix = np.empty((0, self.node_dim), dtype=np.float32)
            self._dirty = False

    def query_knn_latent(self, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[EntityNode, float]]:
        """Performs vector similarity search against the latent substrate."""
        self._sync_matrix_cache()
        if self._cached_matrix is None or len(self._cached_ids) == 0:
            return []

        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        # Cosine similarities
        sims = np.dot(self._cached_matrix, q_norm)
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(self.entities[self._cached_ids[idx]], float(sims[idx])) for idx in top_indices]
