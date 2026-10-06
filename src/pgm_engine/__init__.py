"""Procedural Graph Materialization Engine (pgm_engine).

A deterministic, production-grade engine resolving:
- Non-greedy path discovery (Global Landmark Graph, Bidirectional A* Value Iteration)
- Topological graph contraction (Articulation Shielding, Transitive Shortcut Synthesis)
- High-throughput tiered memory orchestration (Pre-allocated static tensors, FreeIndexPool, Dual-Threshold Hysteresis)
- Differentiable policy learning and credit assignment (Gumbel-Softmax relaxation, Saliency Pruning)
"""

from pgm_engine.engine import ProceduralGraphEngine
from pgm_engine.memory.hysteresis import HysteresisConfig, HysteresisController
from pgm_engine.memory.l1_tensor_buffer import L1TensorBuffer
from pgm_engine.memory.l2_host_graph import L2HostGraph
from pgm_engine.memory.l3_substrate import L3LatentSubstrate
from pgm_engine.memory.manager import MemoryManager
from pgm_engine.policy.differentiable import DifferentiableSubgraphSelector
from pgm_engine.policy.gnn import DownstreamTaskGNN, RelationalMessagePassingLayer
from pgm_engine.policy.rl_policy import GraphPolicyNetwork, PolicyRewardEngine
from pgm_engine.policy.saliency import SaliencyPruningTracker
from pgm_engine.routing.a_star import AStarValueRouter
from pgm_engine.routing.bidirectional import BidirectionalFrontierRouter
from pgm_engine.routing.corridor import CorridorMaterializer
from pgm_engine.routing.landmark import GlobalLandmarkGraph
from pgm_engine.topology.articulation import find_articulation_points, is_local_articulation_point
from pgm_engine.topology.contraction import EdgeContractionOperator
from pgm_engine.topology.guard import TopologyGuard
from pgm_engine.types import BufferStats, EdgeRelation, EntityNode, PathResult

__version__ = "0.1.0"

__all__ = [
    "ProceduralGraphEngine",
    "BufferStats",
    "EdgeRelation",
    "EntityNode",
    "PathResult",
    "L1TensorBuffer",
    "L2HostGraph",
    "L3LatentSubstrate",
    "MemoryManager",
    "HysteresisConfig",
    "HysteresisController",
    "GlobalLandmarkGraph",
    "AStarValueRouter",
    "BidirectionalFrontierRouter",
    "CorridorMaterializer",
    "TopologyGuard",
    "EdgeContractionOperator",
    "find_articulation_points",
    "is_local_articulation_point",
    "DifferentiableSubgraphSelector",
    "SaliencyPruningTracker",
    "DownstreamTaskGNN",
    "RelationalMessagePassingLayer",
    "GraphPolicyNetwork",
    "PolicyRewardEngine",
]
