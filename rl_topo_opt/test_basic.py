"""
Simple test script to verify basic functionality.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ATOP'))

from generator.network import construct_topology
from NSGAII.solution import NetTopology
from converter import convert_atop_to_simplified, is_connected


def test_basic_conversion():
    """Test basic ATOP to SimplifiedTopology conversion."""
    print("=" * 60)
    print("Test 1: Basic Conversion")
    print("=" * 60)

    # Generate a small topology
    print("\n1. Generating ATOP topology...")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    topology, connection_blocks, blueprint = construct_topology(
        total_gpus=8,
        total_layers=2,
        d_max=2,
        generator=full_generator
    )
    net_topo = NetTopology(topology, connection_blocks, blueprint)


    print(f"   Layers: {len(topology.layers)}")
    for i, layer in enumerate(topology.layers):
        print(f"   Layer {i}: {len(layer)} nodes")

    # Convert to simplified
    print("\n2. Converting to SimplifiedTopology...")
    simplified = convert_atop_to_simplified(net_topo)

    print(f"   Nodes: {len(simplified.nodes)}")
    print(f"   Edges: {len(simplified.edges)}")
    print(f"   Node types: {simplified.node_types}")

    # Check connectivity
    print("\n3. Checking connectivity...")
    connected = is_connected(simplified)
    print(f"   Connected: {connected}")

    # Test edge removal
    print("\n4. Testing edge removal...")
    if simplified.edges:
        edge_to_remove = simplified.edges[0]
        print(f"   Removing edge: {edge_to_remove}")
        simplified.remove_edge(edge_to_remove)
        print(f"   Edges after removal: {len(simplified.edges)}")
        connected_after = is_connected(simplified)
        print(f"   Connected after removal: {connected_after}")

    print("\n✓ Test 1 passed!")


def test_environment_creation():
    """Test environment creation."""
    print("\n" + "=" * 60)
    print("Test 2: Environment Creation")
    print("=" * 60)

    from env.topo_env import TopoEnv

    # Generate topology
    print("\n1. Generating topology...")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    topology, connection_blocks, blueprint = construct_topology(
        total_gpus=8,
        total_layers=2,
        d_max=2,
        generator=full_generator
    )
    net_topo = NetTopology(topology, connection_blocks, blueprint)


    # Create environment
    print("\n2. Creating environment...")
    env = TopoEnv(net_topo, max_steps=10)

    print(f"   Action space: {env.action_space}")
    print(f"   Observation space keys: {env.observation_space.spaces.keys()}")

    # Reset environment
    print("\n3. Resetting environment...")
    obs = env.reset()

    print(f"   Node features shape: {obs['node_features'].shape}")
    print(f"   Adjacency matrix shape: {obs['adjacency_matrix'].shape}")
    print(f"   Action mask shape: {obs['action_mask'].shape}")

    # Take a step
    print("\n4. Taking a step...")
    action = 0
    obs, reward, done, info = env.step(action)

    print(f"   Reward: {reward}")
    print(f"   Done: {done}")
    print(f"   Info: {info}")

    print("\n✓ Test 2 passed!")


if __name__ == '__main__':
    try:
        test_basic_conversion()
        test_environment_creation()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
