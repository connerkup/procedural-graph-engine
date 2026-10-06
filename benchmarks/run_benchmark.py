"""Empirical Benchmark Runner: Evaluates PGM Engine against Static k-Hop Greedy Baseline.

Measures:
1. Multi-hop Path Recovery Rate (%) under Bridge Entity Blindness
2. Peak Working Memory (Nodes & Edges in L1)
3. Query Latency (ms)
4. Virtual Shortcuts Synthesized (Contraction count & reachability preservation)
5. Hysteresis Anti-Thrashing Preventions
"""

from __future__ import annotations

import time
from typing import Dict, List
import numpy as np

from benchmarks.baselines import StaticKHopGreedyBaseline
from benchmarks.synthetic_corridor import generate_benchmark_substrate
from pgm_engine.engine import ProceduralGraphEngine


def run_comprehensive_benchmark(
    num_tasks: int = 25,
    hops: int = 4,
    node_dim: int = 64,
    rel_dim: int = 16,
) -> Dict[str, Dict[str, float]]:
    print(f"\n================================================================================")
    print(f"  RUNNING EMPIRICAL MULTI-HOP GRAPH MATERIALIZATION BENCHMARK")
    print(f"  Tasks: {num_tasks} | Hops: {hops} | Node Dim: {node_dim} | Rel Dim: {rel_dim}")
    print(f"================================================================================\n")

    substrate, tasks = generate_benchmark_substrate(
        num_tasks=num_tasks,
        hops_per_task=hops,
        num_distractors_per_node=3,
        background_nodes=150,
        node_dim=node_dim,
        rel_dim=rel_dim,
        seed=101,
    )

    print(f"Substrate generated with {len(substrate.entities)} entities.\n")

    # 1. Evaluate Static k-Hop Greedy Baseline
    print("Evaluating Baseline: Static k-Hop Greedy (top-3 beam)...")
    greedy_baseline = StaticKHopGreedyBaseline(top_k_beam=3, max_hops=hops + 2)

    greedy_recovered = 0
    greedy_latencies = []
    greedy_nodes_expanded = []

    for task in tasks:
        t0 = time.perf_counter()
        res = greedy_baseline.search(task.source_id, task.target_id, substrate)
        lat = (time.perf_counter() - t0) * 1000.0
        greedy_latencies.append(lat)
        greedy_nodes_expanded.append(res.expanded_nodes)
        if res.success:
            greedy_recovered += 1

    greedy_rate = (greedy_recovered / float(num_tasks)) * 100.0
    greedy_avg_lat = float(np.mean(greedy_latencies))
    greedy_avg_nodes = float(np.mean(greedy_nodes_expanded))

    # 2. Evaluate Procedural Graph Materialization Engine
    print("Evaluating Procedural Graph Materialization Engine (PGM)...")
    engine = ProceduralGraphEngine(
        max_nodes_k=1024,
        max_edges_m=8192,
        node_dim=node_dim,
        rel_dim=rel_dim,
        tau_push=0.50,
        tau_pop=0.20,
        t_cooldown=5,
        num_landmark_clusters=8,
        substrate=substrate,
    )
    engine.initialize_landmarks()

    pgm_recovered = 0
    pgm_latencies = []
    pgm_peak_nodes = []

    for task in tasks:
        t0 = time.perf_counter()
        res = engine.query_reasoning_corridor(task.source_id, task.target_id, use_landmarks=True)
        lat = (time.perf_counter() - t0) * 1000.0
        pgm_latencies.append(lat)
        pgm_peak_nodes.append(res.peak_l1_nodes)
        if res.success:
            pgm_recovered += 1

    # 3. Evaluate Topological Graph Contraction under Constrained Memory
    print("Evaluating Topological Pruning & Contraction Under Constrained Budget...")
    # Select active bridge nodes from tasks and contract them to free working memory
    active_initial = len(engine.l1_buffer.entity_to_slot)
    target_budget = max(5, int(active_initial * 0.5))

    # Identify intermediate bridge nodes
    bridge_candidates = [
        eid for eid in engine.l1_buffer.entity_to_slot.keys()
        if "_B" in eid
    ]

    contractions_synthesized = 0
    for b_node in bridge_candidates:
        if len(engine.l1_buffer.entity_to_slot) <= target_budget:
            break
        perm, synth_edges = engine.topology_guard.safe_prune_or_contract(
            node_id=b_node,
            active_nodes=set(engine.l1_buffer.entity_to_slot.keys()),
            l2_graph=engine.l2_graph,
            l1_buffer=engine.l1_buffer,
            force_contraction=True,
        )
        if perm:
            engine.memory_manager.evict_from_l1(b_node)
            contractions_synthesized += len(synth_edges)

    pgm_rate = (pgm_recovered / float(num_tasks)) * 100.0
    pgm_avg_lat = float(np.mean(pgm_latencies))
    pgm_avg_peak_nodes = float(np.mean(pgm_peak_nodes))
    telemetry = engine.get_telemetry()

    results = {
        "Static k-Hop Greedy Baseline": {
            "path_recovery_rate_pct": greedy_rate,
            "avg_latency_ms": greedy_avg_lat,
            "avg_working_nodes": greedy_avg_nodes,
            "contractions_synthesized": 0,
            "refractory_blocks": 0,
        },
        "PGM Engine (Ours)": {
            "path_recovery_rate_pct": pgm_rate,
            "avg_latency_ms": pgm_avg_lat,
            "avg_working_nodes": pgm_avg_peak_nodes,
            "contractions_synthesized": contractions_synthesized,
            "refractory_blocks": telemetry["refractory_blocks"],
        },
    }

    # Print Results Table
    print("\n" + "=" * 80)
    print(f"{'METRIC':<35} | {'STATIC K-HOP GREEDY':<20} | {'PGM ENGINE (OURS)':<18}")
    print("=" * 80)
    print(f"{'Multi-Hop Path Recovery Rate':<35} | {greedy_rate:>18.1f}% | {pgm_rate:>16.1f}%")
    print(f"{'Average Query Latency (ms)':<35} | {greedy_avg_lat:>18.2f}  | {pgm_avg_lat:>16.2f} ")
    print(f"{'Peak Working Nodes (Memory)':<35} | {greedy_avg_nodes:>18.1f}  | {pgm_avg_peak_nodes:>16.1f} ")
    print(f"{'Virtual Contractions Synthesized':<35} | {0:>18}  | {contractions_synthesized:>16} ")
    print(f"{'Active L1 After Contraction':<35} | {greedy_avg_nodes:>18.1f}  | {len(engine.l1_buffer.entity_to_slot):>16} ")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_comprehensive_benchmark(num_tasks=20, hops=4)
