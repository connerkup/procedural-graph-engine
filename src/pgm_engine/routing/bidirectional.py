"""Bidirectional Frontier Expansion with Latent Boundary Intersection.

Simultaneously steps a forward source frontier and a backward target frontier inward,
triggering materialization when latent boundary regions intersect (||z_u - z_w||_2 <= epsilon).
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.routing.a_star import AStarValueRouter
from pgm_engine.types import EdgeRelation, EntityNode, PathResult


class BidirectionalFrontierRouter:
    """Manages concurrent inward-stepping frontiers with latent boundary intersection."""

    def __init__(
        self,
        a_star_router: Optional[AStarValueRouter] = None,
        latent_epsilon: float = 0.25,  # Euclidean intersection threshold on unit sphere
    ):
        self.router = a_star_router or AStarValueRouter()
        self.latent_epsilon = latent_epsilon

    def search(
        self,
        source_id: str,
        target_id: str,
        l2_graph: L2HostGraph,
        substrate: L3LatentSubstrate,
        corridor: Optional[Set[int]] = None,
        max_iterations: int = 500,
    ) -> PathResult:
        """Executes bidirectional lookahead path discovery.

        Args:
            source_id: Root query entity ID.
            target_id: Goal target entity ID.
            l2_graph: In-memory L2 graph.
            substrate: Backing L3 latent substrate.
            corridor: Optional set of allowed cluster IDs.
            max_iterations: Safety limit on total frontier steps.

        Returns:
            PathResult with discovered path and traversal metadata.
        """
        source_node = substrate.get_entity(source_id)
        target_node = substrate.get_entity(target_id)
        if not source_node or not target_node:
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

        # Forward search state (from source towards target)
        # Heap entries: (f_score, counter, node_id)
        counter = 0
        forward_heap: List[Tuple[float, int, str]] = []
        forward_g: Dict[str, float] = {source_id: 0.0}
        forward_parent: Dict[str, Tuple[Optional[str], Optional[EdgeRelation]]] = {source_id: (None, None)}
        forward_visited: Set[str] = set()

        h_src = self.router.compute_heuristic(source_node.embedding, target_node.embedding)
        heapq.heappush(forward_heap, (h_src, counter, source_id))
        counter += 1

        # Backward search state (from target towards source)
        backward_heap: List[Tuple[float, int, str]] = []
        backward_g: Dict[str, float] = {target_id: 0.0}
        backward_parent: Dict[str, Tuple[Optional[str], Optional[EdgeRelation]]] = {target_id: (None, None)}
        backward_visited: Set[str] = set()

        h_tgt = self.router.compute_heuristic(target_node.embedding, source_node.embedding)
        heapq.heappush(backward_heap, (h_tgt, counter, target_id))
        counter += 1

        meeting_node: Optional[str] = None
        bridge_edge: Optional[EdgeRelation] = None
        iterations = 0
        expanded_nodes = 0

        while forward_heap and backward_heap and iterations < max_iterations:
            iterations += 1

            # 1. Step Forward Frontier
            if forward_heap:
                f_score, _, curr_fwd_id = heapq.heappop(forward_heap)
                if curr_fwd_id not in forward_visited:
                    forward_visited.add(curr_fwd_id)
                    expanded_nodes += 1

                    # Check intersection with backward frontier
                    if curr_fwd_id in backward_visited:
                        meeting_node = curr_fwd_id
                        break

                    curr_node = substrate.get_entity(curr_fwd_id)
                    if curr_node:
                        # Check latent boundary intersection against backward visited nodes
                        for b_id in backward_visited:
                            b_node = substrate.get_entity(b_id)
                            if b_node:
                                dist = float(np.linalg.norm(curr_node.embedding - b_node.embedding))
                                if dist <= self.latent_epsilon:
                                    # Intersected! Materialize bridge edge
                                    bridge_edge = EdgeRelation(
                                        source_id=curr_fwd_id,
                                        target_id=b_id,
                                        relation="latent_boundary_bridge",
                                        weight=max(0.1, 1.0 - dist),
                                        is_virtual=True,
                                        metadata={"latent_distance": dist},
                                    )
                                    meeting_node = curr_fwd_id
                                    break
                        if meeting_node:
                            break

                        # Expand forward neighbors
                        neighbors = substrate.expand_procedural_neighbors(curr_fwd_id, direction="out")
                        for nbr_node, edge in neighbors:
                            nbr_id = nbr_node.entity_id
                            if nbr_id in forward_visited:
                                continue

                            # Evaluate A* cost
                            _, g_val, _, _ = self.router.evaluate_node(
                                node=nbr_node,
                                target_emb=target_node.embedding,
                                parent_g=forward_g[curr_fwd_id],
                                edge=edge,
                                corridor=corridor,
                            )

                            if nbr_id not in forward_g or g_val < forward_g[nbr_id]:
                                forward_g[nbr_id] = g_val
                                h_val = self.router.compute_heuristic(nbr_node.embedding, target_node.embedding)
                                forward_parent[nbr_id] = (curr_fwd_id, edge)
                                heapq.heappush(forward_heap, (g_val + h_val, counter, nbr_id))
                                counter += 1

            # 2. Step Backward Frontier
            if backward_heap:
                f_score, _, curr_bwd_id = heapq.heappop(backward_heap)
                if curr_bwd_id not in backward_visited:
                    backward_visited.add(curr_bwd_id)
                    expanded_nodes += 1

                    # Check intersection with forward frontier
                    if curr_bwd_id in forward_visited:
                        meeting_node = curr_bwd_id
                        break

                    curr_node = substrate.get_entity(curr_bwd_id)
                    if curr_node:
                        # Check latent boundary intersection against forward visited nodes
                        for f_id in forward_visited:
                            f_node = substrate.get_entity(f_id)
                            if f_node:
                                dist = float(np.linalg.norm(curr_node.embedding - f_node.embedding))
                                if dist <= self.latent_epsilon:
                                    bridge_edge = EdgeRelation(
                                        source_id=f_id,
                                        target_id=curr_bwd_id,
                                        relation="latent_boundary_bridge",
                                        weight=max(0.1, 1.0 - dist),
                                        is_virtual=True,
                                        metadata={"latent_distance": dist},
                                    )
                                    meeting_node = f_id
                                    break
                        if meeting_node:
                            break

                        # Expand backward neighbors (inward traversal)
                        neighbors = substrate.expand_procedural_neighbors(curr_bwd_id, direction="in")
                        for nbr_node, edge in neighbors:
                            nbr_id = nbr_node.entity_id
                            if nbr_id in backward_visited:
                                continue

                            _, g_val, _, _ = self.router.evaluate_node(
                                node=nbr_node,
                                target_emb=source_node.embedding,
                                parent_g=backward_g[curr_bwd_id],
                                edge=edge,
                                corridor=corridor,
                            )

                            if nbr_id not in backward_g or g_val < backward_g[nbr_id]:
                                backward_g[nbr_id] = g_val
                                h_val = self.router.compute_heuristic(nbr_node.embedding, source_node.embedding)
                                backward_parent[nbr_id] = (curr_bwd_id, edge)
                                heapq.heappush(backward_heap, (g_val + h_val, counter, nbr_id))
                                counter += 1

        if not meeting_node:
            return PathResult(
                success=False,
                path_nodes=[],
                path_edges=[],
                total_cost=float("inf"),
                iterations=iterations,
                expanded_nodes=expanded_nodes,
                virtual_shortcuts_traversed=0,
                peak_l1_nodes=0,
                peak_l1_edges=0,
            )

        # Reconstruct path from source to meeting_node, then meeting_node to target
        path_nodes: List[str] = []
        path_edges: List[EdgeRelation] = []

        # Trace forward: meeting_node back to source
        fwd_chain: List[Tuple[str, Optional[EdgeRelation]]] = []
        curr: Optional[str] = meeting_node
        while curr is not None:
            p_node, p_edge = forward_parent[curr]
            fwd_chain.append((curr, p_edge))
            curr = p_node

        fwd_chain.reverse()
        for nid, edge in fwd_chain:
            path_nodes.append(nid)
            if edge is not None:
                path_edges.append(edge)

        # If a boundary bridge edge was used, attach it
        if bridge_edge is not None:
            path_edges.append(bridge_edge)
            bwd_start = bridge_edge.target_id
        else:
            bwd_start = meeting_node

        # Trace backward: from bwd_start to target
        bwd_curr: Optional[str] = bwd_start
        bwd_nodes: List[str] = []
        bwd_edges: List[EdgeRelation] = []
        while bwd_curr is not None and bwd_curr != target_id:
            p_node, p_edge = backward_parent.get(bwd_curr, (None, None))
            if p_edge is not None:
                bwd_edges.append(p_edge)
            if p_node is not None:
                bwd_nodes.append(p_node)
            bwd_curr = p_node

        path_nodes.extend(bwd_nodes)
        path_edges.extend(bwd_edges)

        virtual_count = sum(1 for e in path_edges if e.is_virtual)
        total_cost = forward_g.get(meeting_node, 0.0) + backward_g.get(bwd_start, 0.0)

        return PathResult(
            success=True,
            path_nodes=path_nodes,
            path_edges=path_edges,
            total_cost=total_cost,
            iterations=iterations,
            expanded_nodes=expanded_nodes,
            virtual_shortcuts_traversed=virtual_count,
            peak_l1_nodes=0,
            peak_l1_edges=0,
        )
