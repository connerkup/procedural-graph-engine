"""Unit tests for Articulation Point detection and Structural Shielding."""

import pytest
from pgm_engine.topology.articulation import find_articulation_points, is_local_articulation_point


def test_chain_articulation_points():
    # Linear graph: A - B - C - D
    # Removing B disconnects A from C,D -> B is cut vertex
    # Removing C disconnects A,B from D -> C is cut vertex
    adj = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C"],
    }
    active_nodes = {"A", "B", "C", "D"}

    cut_pts = find_articulation_points(active_nodes, lambda u: adj[u])
    assert cut_pts == {"B", "C"}

    # Local tests
    assert is_local_articulation_point("B", active_nodes, lambda u: adj[u]) is True
    assert is_local_articulation_point("C", active_nodes, lambda u: adj[u]) is True
    assert is_local_articulation_point("A", active_nodes, lambda u: adj[u]) is False
    assert is_local_articulation_point("D", active_nodes, lambda u: adj[u]) is False


def test_cycle_no_articulation_points():
    # Ring graph: A - B - C - A
    # Removing any node leaves a simple path -> 0 cut vertices
    adj = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"],
    }
    active_nodes = {"A", "B", "C"}
    cut_pts = find_articulation_points(active_nodes, lambda u: adj[u])
    assert len(cut_pts) == 0
