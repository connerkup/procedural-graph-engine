"""Global Landmark Graph: Dual-resolution latent scaffolding for coarse corridor routing.

Clusters substrate embeddings into coarse semantic landmarks (k-means or Louvain).
Maps query and target vectors to landmark corridors, shielding low-similarity bridge
entities from early greedy rejection.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from pgm_engine.memory.l3_substrate import L3LatentSubstrate


class GlobalLandmarkGraph:
    """Coarse semantic scaffolding over latent substrate.

    Attributes:
        num_clusters: Number of landmark centroids.
        centroids: Array of shape (num_clusters, node_dim).
        cluster_adjacency: Coarse cluster transition graph (c_i -> set of c_j).
    """

    def __init__(self, num_clusters: int = 16, node_dim: int = 128):
        self.num_clusters = num_clusters
        self.node_dim = node_dim
        self.centroids: np.ndarray = np.zeros((num_clusters, node_dim), dtype=np.float32)
        # Cluster graph adjacency: cluster_id -> set of adjacent cluster_ids
        self.cluster_adjacency: Dict[int, Set[int]] = defaultdict(set)
        # Entity ID -> cluster ID
        self.entity_to_cluster: Dict[str, int] = {}
        # Cluster ID -> set of entity IDs
        self.cluster_to_entities: Dict[int, Set[str]] = defaultdict(set)

    def build_from_substrate(self, substrate: L3LatentSubstrate, max_iters: int = 30) -> None:
        """Constructs landmark clusters and coarse connectivity from backing substrate."""
        entity_ids = list(substrate.entities.keys())
        if not entity_ids:
            return

        embeddings = np.array([substrate.entities[eid].embedding for eid in entity_ids], dtype=np.float32)
        n_samples = len(entity_ids)
        k = min(self.num_clusters, n_samples)
        self.num_clusters = k

        # K-Means clustering over normalized latent vectors
        np.random.seed(42)
        initial_indices = np.random.choice(n_samples, size=k, replace=False)
        self.centroids = embeddings[initial_indices].copy()

        labels = np.zeros(n_samples, dtype=np.int32)
        for _ in range(max_iters):
            # Compute cosine similarities to centroids: (n_samples, k)
            sims = np.dot(embeddings, self.centroids.T)
            new_labels = np.argmax(sims, axis=1)

            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_mean = np.mean(embeddings[mask], axis=0)
                    norm = np.linalg.norm(new_mean)
                    self.centroids[c] = new_mean / (norm + 1e-9)

        # Store entity-cluster assignments
        self.cluster_to_entities.clear()
        self.entity_to_cluster.clear()
        for idx, eid in enumerate(entity_ids):
            c_id = int(labels[idx])
            self.entity_to_cluster[eid] = c_id
            self.cluster_to_entities[c_id].add(eid)
            # Update node's cluster_id attribute
            substrate.entities[eid].cluster_id = c_id

        # Build coarse adjacency from substrate relationships
        self.cluster_adjacency.clear()
        for src_id, targets in substrate.adjacency.items():
            src_cluster = self.entity_to_cluster.get(src_id)
            if src_cluster is None:
                continue
            for tgt_id in targets.keys():
                tgt_cluster = self.entity_to_cluster.get(tgt_id)
                if tgt_cluster is not None:
                    self.cluster_adjacency[src_cluster].add(tgt_cluster)
                    self.cluster_adjacency[tgt_cluster].add(src_cluster)

    def map_embedding_to_cluster(self, embedding: np.ndarray) -> int:
        """Finds closest landmark cluster for a query or target embedding."""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        sims = np.dot(self.centroids, emb_norm)
        return int(np.argmax(sims))

    def compute_routing_corridor(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        allow_neighboring_clusters: bool = True,
    ) -> Set[int]:
        """Calculates shortest coarse path across landmark graph.

        Returns:
            Set of cluster IDs designated as the active routing corridor.
        """
        src_cluster = self.map_embedding_to_cluster(source_emb)
        tgt_cluster = self.map_embedding_to_cluster(target_emb)

        if src_cluster == tgt_cluster:
            corridor = {src_cluster}
        else:
            # BFS shortest path on coarse cluster graph
            queue: deque[Tuple[int, List[int]]] = deque([(src_cluster, [src_cluster])])
            visited = {src_cluster}
            path: Optional[List[int]] = None

            while queue:
                curr, curr_path = queue.popleft()
                if curr == tgt_cluster:
                    path = curr_path
                    break

                for nbr_cluster in self.cluster_adjacency.get(curr, set()):
                    if nbr_cluster not in visited:
                        visited.add(nbr_cluster)
                        queue.append((nbr_cluster, curr_path + [nbr_cluster]))

            if path is not None:
                corridor = set(path)
            else:
                # If coarse graph is disconnected, fall back to endpoints
                corridor = {src_cluster, tgt_cluster}

        if allow_neighboring_clusters:
            # Expand corridor to 1-hop cluster neighbors for resilience
            expanded: Set[int] = set(corridor)
            for c in corridor:
                expanded.update(self.cluster_adjacency.get(c, set()))
            return expanded

        return corridor

    def is_in_corridor(self, entity_id: str, corridor: Set[int]) -> bool:
        """Verifies if an entity belongs to an active routing corridor."""
        c_id = self.entity_to_cluster.get(entity_id)
        if c_id is None:
            return True  # If unknown, do not filter aggressively
        return c_id in corridor
