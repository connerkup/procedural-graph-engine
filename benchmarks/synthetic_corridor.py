"""Synthetic Multi-Hop Reasoning Benchmark Generator.

Generates realistic knowledge graphs with bridge entity corridors, high-similarity
distractor dead-ends, and semantic clustering designed to rigorously evaluate:
1. Greedy Horizon / Bridge Entity Blindness
2. Graph Contraction under tight memory budgets
3. Anti-thrashing hysteresis during iterative search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from pgm_engine.memory.l3_substrate import L3LatentSubstrate


@dataclass
class ReasoningTask:
    task_id: str
    source_id: str
    target_id: str
    ground_truth_path: List[str]
    num_hops: int


def generate_benchmark_substrate(
    num_tasks: int = 30,
    hops_per_task: int = 4,
    num_distractors_per_node: int = 3,
    background_nodes: int = 200,
    node_dim: int = 64,
    rel_dim: int = 16,
    seed: int = 42,
) -> Tuple[L3LatentSubstrate, List[ReasoningTask]]:
    """Generates an evaluation substrate and a set of multi-hop reasoning tasks.

    Each reasoning path S -> B_1 -> ... -> B_{H-1} -> T has:
    - S and T in distinct clusters.
    - Intermediate bridge nodes B_i orthogonal to T (zero direct cosine similarity).
    - Distractors branching off S and B_i with high direct similarity to T, but dead ends.
    """
    np.random.seed(seed)
    sub = L3LatentSubstrate(node_dim=node_dim, rel_dim=rel_dim)
    tasks: List[ReasoningTask] = []

    for t_idx in range(num_tasks):
        s_id = f"T{t_idx}_SRC"
        t_id = f"T{t_idx}_TGT"

        # Directional basis vectors
        v_s = np.zeros(node_dim, dtype=np.float32)
        v_s[0] = 1.0  # Axis 0 for source

        v_t = np.zeros(node_dim, dtype=np.float32)
        v_t[1] = 1.0  # Axis 1 for target

        sub.add_entity(s_id, v_s, cluster_id=0, metadata={"type": "source", "task": t_idx})
        sub.add_entity(t_id, v_t, cluster_id=1, metadata={"type": "target", "task": t_idx})

        path_nodes = [s_id]
        prev_node = s_id

        # Generate intermediate bridge nodes
        for h in range(1, hops_per_task):
            b_id = f"T{t_idx}_B{h}"
            # Orthogonal axis for bridge (axes 2..node_dim-1) -> strict 0.0 cosine sim with target!
            axis = 2 + (h % (node_dim - 2))
            v_b = np.zeros(node_dim, dtype=np.float32)
            v_b[axis] = 1.0

            sub.add_entity(b_id, v_b, cluster_id=2 + h, metadata={"type": "bridge", "task": t_idx, "hop": h})
            sub.add_relation(prev_node, b_id, relation=f"path_step_{h-1}", weight=0.95)
            path_nodes.append(b_id)

            # Add high-similarity distractors attached to previous node
            for d in range(num_distractors_per_node):
                d_id = f"T{t_idx}_D_{h}_{d}"
                # Distractor has high superficial similarity to target (cos sim ~0.60), tempting greedy search,
                # but Euclidean distance to target is ~0.89 (well above epsilon=0.25) and belongs to a dead-end cluster
                v_d = np.zeros(node_dim, dtype=np.float32)
                v_d[1] = 0.60 + np.random.uniform(0.01, 0.05)
                v_d[axis] = 0.80
                v_d /= np.linalg.norm(v_d)

                sub.add_entity(d_id, v_d, cluster_id=15, metadata={"type": "distractor", "task": t_idx})
                sub.add_relation(prev_node, d_id, relation=f"distractor_step_{d}", weight=0.90)

            prev_node = b_id

        # Final edge to target
        sub.add_relation(prev_node, t_id, relation=f"path_step_{hops_per_task-1}", weight=0.95)
        path_nodes.append(t_id)

        tasks.append(
            ReasoningTask(
                task_id=f"Task_{t_idx}",
                source_id=s_id,
                target_id=t_id,
                ground_truth_path=path_nodes,
                num_hops=hops_per_task,
            )
        )

    # Add background clutter entities
    for bg_idx in range(background_nodes):
        bg_id = f"BG_{bg_idx}"
        v_bg = np.random.randn(node_dim).astype(np.float32)
        v_bg /= np.linalg.norm(v_bg)
        sub.add_entity(bg_id, v_bg, cluster_id=np.random.randint(0, 10))

    return sub, tasks
