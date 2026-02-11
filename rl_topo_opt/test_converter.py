"""
Unit tests for converter module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ATOP'))

import pytest
from converter import SimplifiedTopology, convert_atop_to_simplified, is_connected
from generator.network import construct_topology


def test_convert_atop_to_simplified():
    """Test ATOP to SimplifiedTopology conversion."""
    # Generate a small topology
    net_topo = construct_topology(n_gpus=8, depth=2, width=2)

    # Convert
    simplified = convert_atop_to_simplified(net_topo)

    # Check basic properties
    assert len(simplified.nodes) > 0
    assert len(simplified.edges) > 0
    assert len(simplified.node_types) == len(simplified.nodes)

    # Check node types
    gpu_count = sum(1 for t in simplified.node_types.values() if t == 'GPU')
    assert gpu_count == 8  # Should have 8 GPU nodes

    # Check edge bandwidths
    for edge in simplified.edges:
        assert edge in simplified.edge_bandwidths
        assert simplified.edge_bandwidths[edge] > 0


def test_is_connected():
    """Test connectivity check."""
    # Create a simple connected topology
    topo = SimplifiedTopology(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'SWITCH', 3: 'GPU'},
        edge_bandwidths={(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0},
        original_net_topo=None
    )

    assert is_connected(topo) == True

    # Remove an edge to disconnect
    topo.remove_edge((1, 2))
    assert is_connected(topo) == False


def test_remove_edge():
    """Test edge removal."""
    topo = SimplifiedTopology(
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'GPU'},
        edge_bandwidths={(0, 1): 1.0, (1, 2): 1.0},
        original_net_topo=None
    )

    initial_edge_count = len(topo.edges)
    topo.remove_edge((0, 1))

    assert len(topo.edges) < initial_edge_count
    assert (0, 1) not in topo.edges
    assert (0, 1) not in topo.edge_bandwidths


def test_adjacency_matrix():
    """Test adjacency matrix generation."""
    topo = SimplifiedTopology(
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'GPU'},
        edge_bandwidths={(0, 1): 1.0, (1, 2): 1.0},
        original_net_topo=None
    )

    adj = topo.get_adjacency_matrix()

    assert adj.shape == (3, 3)
    assert adj[0, 1] == 1.0  # Edge exists
    assert adj[1, 0] == 1.0  # Undirected
    assert adj[0, 2] == 0.0  # No direct edge


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
