"""Topology rewrite utilities for full-mesh initialization."""

from __future__ import annotations

from itertools import combinations
from typing import Dict

from add01_full_mesh_topology.types import TopologyRewriteMode


def _as_node_id(value):
    """Normalize sibling entries that may be node ids or GraphNode objects."""
    return value.node_id if hasattr(value, "node_id") else value


def _count_undirected_edges(net_topo) -> int:
    """Count unique undirected edges from NetTopology siblings."""
    edges = set()
    for node_id, node in net_topo.topology.nodes.items():
        for sibling in node.siblings:
            sibling_id = _as_node_id(sibling)
            if sibling_id == node_id:
                continue
            edges.add(tuple(sorted((node_id, sibling_id))))
    return len(edges)


def _ensure_intra_layer_blueprint(net_topo, default_bandwidth: float) -> None:
    """Ensure each layer has Bii for intra-layer bandwidth lookups."""
    if not isinstance(net_topo.blueprint, dict):
        net_topo.blueprint = {}

    for layer_idx in range(len(net_topo.topology.layers)):
        layer_cfg = net_topo.blueprint.get(layer_idx)
        if not isinstance(layer_cfg, dict):
            layer_cfg = {}
        layer_cfg["Bii"] = float(default_bandwidth)
        net_topo.blueprint[layer_idx] = layer_cfg


def _build_inter_layer_connection_blocks(net_topo, default_bandwidth: float) -> Dict:
    """Build full inter-layer connection blocks required by ATOP scorers."""
    connection_blocks = {}
    layers = net_topo.topology.layers

    for i, j in combinations(range(len(layers)), 2):
        if not layers[i] or not layers[j]:
            continue

        # e_ij/i/j are retained for compatibility with ATOP-style block metadata.
        connection_blocks[(i, j)] = {
            "i": 1,
            "j": 1,
            "e_ij": len(layers[j]),
            "b_ij": float(default_bandwidth),
            "bandwidth": float(default_bandwidth),
        }

    return connection_blocks


def _rewrite_to_full_mesh(net_topo, default_bandwidth: float):
    """Mutate NetTopology to a full mesh while preserving all original nodes/layers."""
    all_node_ids = list(net_topo.topology.nodes.keys())
    node_id_set = set(all_node_ids)

    # Clear all existing edges first.
    for node in net_topo.topology.nodes.values():
        node.siblings = set()

    # Build undirected complete graph: every pair is connected both ways.
    for src_id, dst_id in combinations(all_node_ids, 2):
        net_topo.topology.nodes[src_id].siblings.add(dst_id)
        net_topo.topology.nodes[dst_id].siblings.add(src_id)

    # Safety: remove accidental self-loops or invalid sibling references.
    for node_id, node in net_topo.topology.nodes.items():
        node.siblings = {
            sibling_id
            for sibling_id in (_as_node_id(s) for s in node.siblings)
            if sibling_id != node_id and sibling_id in node_id_set
        }

    _ensure_intra_layer_blueprint(net_topo, default_bandwidth)
    net_topo.connection_blocks = _build_inter_layer_connection_blocks(net_topo, default_bandwidth)

    return net_topo


def rewrite_topology(net_topo, mode: str = "none", default_bandwidth: float = 1.0):
    """Rewrite topology according to selected mode.

    Args:
        net_topo: ATOP NetTopology
        mode: 'none' or 'full_mesh'
        default_bandwidth: Default edge bandwidth for rewritten links

    Returns:
        NetTopology (same object, possibly mutated)
    """
    resolved_mode = TopologyRewriteMode(mode)

    if resolved_mode == TopologyRewriteMode.NONE:
        return net_topo

    if default_bandwidth <= 0:
        raise ValueError("default_bandwidth must be positive")

    before_edges = _count_undirected_edges(net_topo)

    if resolved_mode == TopologyRewriteMode.FULL_MESH:
        rewritten = _rewrite_to_full_mesh(net_topo, default_bandwidth=default_bandwidth)
        after_edges = _count_undirected_edges(rewritten)
        print(
            "Topology rewrite applied: "
            f"mode={resolved_mode.value}, "
            f"nodes={len(rewritten.topology.nodes)}, "
            f"edges(before={before_edges}, after={after_edges}), "
            f"bandwidth={default_bandwidth}"
        )
        return rewritten

    raise ValueError(f"Unsupported topology rewrite mode: {mode}")
