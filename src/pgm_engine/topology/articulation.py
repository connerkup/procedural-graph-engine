"""Articulation point and cut-vertex detection for structural shielding.

Implements Tarjan's low-point discovery and 2-hop localized biconnected component checks
to ensure dynamic pruning never partitions the active reasoning subgraph.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Set


def find_articulation_points(
    active_nodes: Set[str],
    neighbors_fn: Callable[[str], List[str]],
) -> Set[str]:
    """Finds all articulation points (cut vertices) across active nodes using Tarjan's algorithm.

    An articulation point is a node whose removal increases the number of connected components.

    Args:
        active_nodes: Set of active node IDs to inspect.
        neighbors_fn: Function returning adjacent node IDs for a given entity.

    Returns:
        Set of entity IDs that are cut vertices in the active subgraph.
    """
    if len(active_nodes) <= 2:
        return set()

    discovery_time: Dict[str, int] = {}
    low_point: Dict[str, int] = {}
    parent: Dict[str, str | None] = {}
    cut_vertices: Set[str] = set()
    timer = 0

    def dfs(u: str) -> None:
        nonlocal timer
        timer += 1
        discovery_time[u] = timer
        low_point[u] = timer
        children = 0

        # Only consider neighbors that are in active_nodes
        adj = [v for v in neighbors_fn(u) if v in active_nodes and v != u]

        for v in adj:
            if v not in discovery_time:
                parent[v] = u
                children += 1
                dfs(v)
                low_point[u] = min(low_point[u], low_point[v])

                # Root condition
                if parent[u] is None and children > 1:
                    cut_vertices.add(u)
                # Non-root condition
                elif parent[u] is not None and low_point[v] >= discovery_time[u]:
                    cut_vertices.add(u)
            elif v != parent[u]:
                # Back-edge
                low_point[u] = min(low_point[u], discovery_time[v])

    for node in active_nodes:
        if node not in discovery_time:
            parent[node] = None
            dfs(node)

    return cut_vertices


def is_local_articulation_point(
    node: str,
    active_nodes: Set[str],
    neighbors_fn: Callable[[str], List[str]],
    k_hops: int = 2,
) -> bool:
    """Fast localized cut-vertex test over a k-hop egocentric neighborhood.

    Instead of global O(|V| + |E|) traversal, checks if removing `node` disconnects
    its immediate active neighbors within a localized k-hop ego-network.

    Args:
        node: Candidate node to evaluate.
        active_nodes: Set of active node IDs.
        neighbors_fn: Neighbor retrieval function.
        k_hops: Radius of the local ego-network (default 2).

    Returns:
        True if removing `node` breaks connectivity between its neighbors.
    """
    immediate_nbrs = [nbr for nbr in neighbors_fn(node) if nbr in active_nodes and nbr != node]
    if len(immediate_nbrs) <= 1:
        # A node with 0 or 1 neighbor cannot partition the remaining graph
        return False

    # Extract k-hop neighborhood excluding `node`
    visited: Set[str] = {node}
    queue: deque[tuple[str, int]] = deque([(immediate_nbrs[0], 0)])
    reached_from_first: Set[str] = set()

    while queue:
        curr, depth = queue.popleft()
        if curr in reached_from_first:
            continue
        reached_from_first.add(curr)

        if depth < k_hops:
            for nbr in neighbors_fn(curr):
                if nbr in active_nodes and nbr not in visited and nbr not in reached_from_first:
                    queue.append((nbr, depth + 1))

    # Check if all other immediate neighbors can be reached without traversing `node`
    for other_nbr in immediate_nbrs[1:]:
        if other_nbr not in reached_from_first:
            return True

    return False
