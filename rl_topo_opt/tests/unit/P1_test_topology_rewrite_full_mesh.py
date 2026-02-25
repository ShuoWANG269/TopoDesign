"""P1 tests for full-mesh topology rewrite module."""

import math
import numpy as np

from env.reward import RewardCalculator
from env.topo_env import TopoEnv
from add01_full_mesh_topology import TopologyRewriteMode, rewrite_topology


def _undirected_edge_set(net_topo):
    """Extract unique undirected edges from NetTopology siblings."""
    edges = set()
    for node_id, node in net_topo.topology.nodes.items():
        for sibling in node.siblings:
            sibling_id = sibling.node_id if hasattr(sibling, "node_id") else sibling
            if sibling_id == node_id:
                continue
            edges.add(tuple(sorted((node_id, sibling_id))))
    return edges


def test_rewrite_none_keeps_original_edges(small_net_topo):
    """Mode none should preserve original topology."""
    before = _undirected_edge_set(small_net_topo)
    same_obj = rewrite_topology(small_net_topo, mode=TopologyRewriteMode.NONE.value)
    after = _undirected_edge_set(same_obj)

    assert same_obj is small_net_topo
    assert before == after


def test_full_mesh_rewrite_structure_and_bandwidth(small_net_topo):
    """Full mesh should preserve nodes/layers and rebuild all links."""
    node_ids_before = set(small_net_topo.topology.nodes.keys())
    node_types_before = {
        node_id: node.node_type.value
        for node_id, node in small_net_topo.topology.nodes.items()
    }
    layers_before = [layer.copy() for layer in small_net_topo.topology.layers]

    rewritten = rewrite_topology(
        small_net_topo,
        mode=TopologyRewriteMode.FULL_MESH.value,
        default_bandwidth=2.5,
    )

    node_ids_after = set(rewritten.topology.nodes.keys())
    node_types_after = {
        node_id: node.node_type.value
        for node_id, node in rewritten.topology.nodes.items()
    }
    undirected_edges = _undirected_edge_set(rewritten)
    n = len(node_ids_after)

    assert node_ids_after == node_ids_before
    assert node_types_after == node_types_before
    assert rewritten.topology.layers == layers_before
    assert len(undirected_edges) == n * (n - 1) // 2

    for node_id, node in rewritten.topology.nodes.items():
        assert node_id not in node.siblings
        assert len(node.siblings) == n - 1

    for layer_idx in range(len(rewritten.topology.layers)):
        assert rewritten.blueprint[layer_idx]["Bii"] == 2.5

    for i in range(len(rewritten.topology.layers)):
        for j in range(i + 1, len(rewritten.topology.layers)):
            if rewritten.topology.layers[i] and rewritten.topology.layers[j]:
                assert rewritten.connection_blocks[(i, j)]["b_ij"] == 2.5


def test_full_mesh_env_and_reward_integration(small_net_topo):
    """Environment and reward calculator should work on rewritten topology."""
    rewritten = rewrite_topology(
        small_net_topo,
        mode=TopologyRewriteMode.FULL_MESH.value,
        default_bandwidth=1.0,
    )

    env = TopoEnv(rewritten, max_steps=5)
    expected_edges = len(rewritten.topology.nodes) * (len(rewritten.topology.nodes) - 1) // 2
    assert env.action_space.n == expected_edges

    obs = env.reset()
    assert obs["action_mask"].shape[0] == expected_edges

    calculator = RewardCalculator()
    reward = calculator.calculate_reward(rewritten, is_connected=True)

    assert isinstance(reward, (int, float, np.number))
    assert not math.isnan(float(reward))
    assert not math.isinf(float(reward))
