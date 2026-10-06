"""Topological guards, articulation-point detection, and virtual edge contraction."""

from pgm_engine.topology.articulation import find_articulation_points, is_local_articulation_point
from pgm_engine.topology.contraction import EdgeContractionOperator
from pgm_engine.topology.guard import TopologyGuard

__all__ = [
    "find_articulation_points",
    "is_local_articulation_point",
    "EdgeContractionOperator",
    "TopologyGuard",
]
