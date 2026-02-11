"""
Pytest configuration and shared fixtures for RL topology optimization tests.
"""

import pytest
import sys
import os
from typing import Dict, List, Tuple

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
atop_root = os.path.join(os.path.dirname(project_root), 'ATOP')

sys.path.insert(0, project_root)
sys.path.insert(0, atop_root)

from converter import SimplifiedTopology, convert_atop_to_simplified
from env.topo_env import TopoEnv
from env.reward import RewardCalculator

try:
    from generator.network import construct_topology, GraphNode
    from NSGAII.solution import NetTopology
    ATOP_AVAILABLE = True
except ImportError:
    ATOP_AVAILABLE = False


def _calculate_net_topo_edges(net_topo):
    """
    Calculate edges from NetTopology.nodes GraphNode.siblings.
    NetTopology doesn't have edges attribute by default, so we calculate it.
    Uses unordered pairs to avoid double-counting bidirectional edges.
    """
    edges = set()
    for node_id, node in net_topo.topology.nodes.items():
        for sibling in node.siblings:
            sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
            # Use unordered pair to avoid duplicates
            edge = tuple(sorted([node_id, sibling_id]))
            edges.add(edge)
    return list(edges)


def create_net_topo(total_gpus=8, total_layers=2, d_max=2, max_attempts=10):
    """
    Helper function to create a valid NetTopology with edges attribute.

    Returns:
        NetTopology with .edges attribute populated
    """
    if not ATOP_AVAILABLE:
        return None

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    for attempt in range(max_attempts):
        topology, connection_blocks, blueprint = construct_topology(
            total_gpus=total_gpus,
            total_layers=total_layers,
            d_max=d_max,
            generator=full_generator
        )
        net_topo = NetTopology(topology, connection_blocks, blueprint)

        # Calculate edges from GraphNode.siblings
        edges = _calculate_net_topo_edges(net_topo)

        if len(edges) > 0:
            net_topo.edges = edges
            return net_topo

    return None


# ============================================================================
# Fixtures for ATOP NetTopology
# ============================================================================

@pytest.fixture
def small_net_topo():
    """Generate small ATOP NetTopology (8 GPU, depth=2, width=2)."""
    if not ATOP_AVAILABLE:
        pytest.skip("ATOP module not available")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    # Retry loop to ensure valid topology with edges
    max_attempts = 10
    for attempt in range(max_attempts):
        topology, connection_blocks, blueprint = construct_topology(
            total_gpus=8,
            total_layers=2,
            d_max=2,
            generator=full_generator
        )
        net_topo = NetTopology(topology, connection_blocks, blueprint)

        # Calculate edges from GraphNode.siblings (use unordered pairs)
        edges = set()
        for node_id, node in topology.nodes.items():
            for sibling in node.siblings:
                sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                edge = tuple(sorted([node_id, sibling_id]))
                edges.add(edge)

        if len(edges) > 0:
            # Valid topology found, add edges attribute
            net_topo.edges = list(edges)
            return net_topo

    pytest.fail(f"Could not generate valid topology with edges after {max_attempts} attempts")


@pytest.fixture
def medium_net_topo():
    """Generate medium ATOP NetTopology (16 GPU, depth=3, width=2)."""
    if not ATOP_AVAILABLE:
        pytest.skip("ATOP module not available")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    # Retry loop to ensure valid topology with edges
    max_attempts = 10
    for attempt in range(max_attempts):
        topology, connection_blocks, blueprint = construct_topology(
            total_gpus=16,
            total_layers=3,
            d_max=2,
            generator=full_generator
        )
        net_topo = NetTopology(topology, connection_blocks, blueprint)

        # Calculate edges from GraphNode.siblings (use unordered pairs)
        edges = set()
        for node_id, node in topology.nodes.items():
            for sibling in node.siblings:
                sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                edge = tuple(sorted([node_id, sibling_id]))
                edges.add(edge)

        if len(edges) > 0:
            # Valid topology found, add edges attribute
            net_topo.edges = list(edges)
            return net_topo

    pytest.fail(f"Could not generate valid topology with edges after {max_attempts} attempts")


# ============================================================================
# Fixtures for SimplifiedTopology (manual construction)
# ============================================================================

@pytest.fixture
def simple_linear_topo() -> SimplifiedTopology:
    """
    Create a simple linear topology: 0 - 1 - 2 - 3
    """
    return SimplifiedTopology(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3),
               (1, 0), (2, 1), (3, 2)],  # Both directions
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'SWITCH', 3: 'GPU'},
        edge_bandwidths={
            (0, 1): 1.0, (1, 0): 1.0,
            (1, 2): 1.0, (2, 1): 1.0,
            (2, 3): 1.0, (3, 2): 1.0
        },
        original_net_topo=None
    )


@pytest.fixture
def simple_star_topo() -> SimplifiedTopology:
    """
    Create a star topology with center node.
    Structure: 1 - 0 - 2
               |   |
               3   4
    """
    return SimplifiedTopology(
        nodes=[0, 1, 2, 3, 4],
        edges=[(0, 1), (1, 0),
               (0, 2), (2, 0),
               (0, 3), (3, 0),
               (0, 4), (4, 0)],
        node_types={0: 'SWITCH', 1: 'GPU', 2: 'GPU', 3: 'GPU', 4: 'GPU'},
        edge_bandwidths={
            (0, 1): 1.0, (1, 0): 1.0,
            (0, 2): 1.0, (2, 0): 1.0,
            (0, 3): 1.0, (3, 0): 1.0,
            (0, 4): 1.0, (4, 0): 1.0
        },
        original_net_topo=None
    )


@pytest.fixture
def disconnected_topo() -> SimplifiedTopology:
    """
    Create a disconnected topology with two separate components.
    Component 1: 0 - 1
    Component 2: 2 - 3
    """
    return SimplifiedTopology(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 0),
               (2, 3), (3, 2)],
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'GPU', 3: 'SWITCH'},
        edge_bandwidths={
            (0, 1): 1.0, (1, 0): 1.0,
            (2, 3): 1.0, (3, 2): 1.0
        },
        original_net_topo=None
    )


@pytest.fixture
def single_edge_topo() -> SimplifiedTopology:
    """
    Create a topology with only two nodes and one edge.
    """
    return SimplifiedTopology(
        nodes=[0, 1],
        edges=[(0, 1), (1, 0)],
        node_types={0: 'GPU', 1: 'GPU'},
        edge_bandwidths={(0, 1): 1.0, (1, 0): 1.0},
        original_net_topo=None
    )


@pytest.fixture
def isolated_node_topo() -> SimplifiedTopology:
    """
    Create a topology where one node is isolated.
    Connected: 0 - 1 - 2
    Isolated: 3
    """
    return SimplifiedTopology(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 0),
               (1, 2), (2, 1)],
        node_types={0: 'GPU', 1: 'SWITCH', 2: 'GPU', 3: 'GPU'},
        edge_bandwidths={
            (0, 1): 1.0, (1, 0): 1.0,
            (1, 2): 1.0, (2, 1): 1.0
        },
        original_net_topo=None
    )


# ============================================================================
# Fixtures for TopoEnv
# ============================================================================

@pytest.fixture(scope='function')
def topo_env():
    """Create a TopoEnv instance with small topology (fresh for each test)."""
    # Create a fresh topology for each test to avoid state pollution
    if not ATOP_AVAILABLE:
        pytest.skip("ATOP module not available")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    # Retry loop to ensure valid topology with edges
    max_attempts = 10
    for attempt in range(max_attempts):
        topology, connection_blocks, blueprint = construct_topology(
            total_gpus=8,
            total_layers=2,
            d_max=2,
            generator=full_generator
        )
        fresh_topo = NetTopology(topology, connection_blocks, blueprint)

        # Calculate edges from GraphNode.siblings (use unordered pairs)
        edges = set()
        for node_id, node in topology.nodes.items():
            for sibling in node.siblings:
                sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                edge = tuple(sorted([node_id, sibling_id]))
                edges.add(edge)

        if len(edges) > 0:
            fresh_topo.edges = list(edges)
            try:
                env = TopoEnv(fresh_topo, max_steps=10)
                return env
            except Exception:
                continue

    pytest.fail(f"Could not create valid TopoEnv after {max_attempts} attempts")


@pytest.fixture
def topo_env_long():
    """Create a TopoEnv instance with longer episode length."""
    if not ATOP_AVAILABLE:
        pytest.skip("ATOP module not available")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    # Retry loop to ensure valid topology with edges
    max_attempts = 10
    for attempt in range(max_attempts):
        topology, connection_blocks, blueprint = construct_topology(
            total_gpus=8,
            total_layers=2,
            d_max=2,
            generator=full_generator
        )
        fresh_topo = NetTopology(topology, connection_blocks, blueprint)

        # Calculate edges from GraphNode.siblings (use unordered pairs)
        edges = set()
        for node_id, node in topology.nodes.items():
            for sibling in node.siblings:
                sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                edge = tuple(sorted([node_id, sibling_id]))
                edges.add(edge)

        if len(edges) > 0:
            fresh_topo.edges = list(edges)
            return TopoEnv(fresh_topo, max_steps=100)

    pytest.fail(f"Could not create valid TopoEnv after {max_attempts} attempts")


# ============================================================================
# Fixtures for RewardCalculator
# ============================================================================

@pytest.fixture
def reward_calculator():
    """Create a RewardCalculator with default weights."""
    return RewardCalculator()


@pytest.fixture
def custom_weights_calculator():
    """Create a RewardCalculator with custom weights."""
    weights = {'cost': 0.5, 'latency': 0.3, 'fault_tolerance': 0.2}
    return RewardCalculator(weights=weights)


# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "atop_required: marks tests that require ATOP module"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks integration tests"
    )
