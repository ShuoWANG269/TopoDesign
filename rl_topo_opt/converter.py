"""
ATOP NetTopology to SimplifiedTopology converter.
Converts ATOP's complex topology representation to a simplified graph format for RL.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class SimplifiedTopology:
    """Simplified topology representation for RL environment."""
    nodes: List[int]  # Node IDs
    edges: List[Tuple[int, int]]  # Edge list (source, target)
    node_types: Dict[int, str]  # {node_id: 'GPU' or 'SWITCH'}
    edge_bandwidths: Dict[Tuple[int, int], float]  # Edge bandwidth
    original_net_topo: object  # Reference to original ATOP NetTopology

    def copy(self):
        """Create a deep copy of the topology."""
        return SimplifiedTopology(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            node_types=self.node_types.copy(),
            edge_bandwidths=self.edge_bandwidths.copy(),
            original_net_topo=self.original_net_topo
        )

    def remove_edge(self, edge: Tuple[int, int]):
        """Remove an edge from the topology."""
        if edge in self.edges:
            self.edges.remove(edge)
        # Also remove reverse edge if exists
        reverse_edge = (edge[1], edge[0])
        if reverse_edge in self.edges:
            self.edges.remove(reverse_edge)

        # Remove from bandwidth dict
        self.edge_bandwidths.pop(edge, None)
        self.edge_bandwidths.pop(reverse_edge, None)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation."""
        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        for src, dst in self.edges:
            if src in node_to_idx and dst in node_to_idx:
                adj[node_to_idx[src], node_to_idx[dst]] = 1.0
                adj[node_to_idx[dst], node_to_idx[src]] = 1.0  # Undirected

        return adj


def convert_atop_to_simplified(net_topo) -> SimplifiedTopology:
    """
    Convert ATOP NetTopology to SimplifiedTopology.

    Args:
        net_topo: ATOP NetTopology object

    Returns:
        SimplifiedTopology object
    """
    nodes = []
    edges = []
    node_types = {}
    edge_bandwidths = {}

    # Extract nodes from topology
    topology = net_topo.topology
    all_nodes_dict = {}  # Map node_id to GraphNode

    for layer_idx, layer in enumerate(topology.layers):
        for node_id in layer:
            nodes.append(node_id)
            # Get the actual GraphNode from topology.nodes dict
            if node_id in topology.nodes:
                node = topology.nodes[node_id]
                all_nodes_dict[node_id] = node
                # Determine node type
                node_types[node_id] = 'GPU' if node.node_type.value == 'GPU' else 'SWITCH'
            else:
                # Fallback: GPU nodes are in first layer
                node_types[node_id] = 'GPU' if layer_idx == 0 else 'SWITCH'

    # Extract edges from GraphNode.siblings
    for node_id, node in all_nodes_dict.items():
        for sibling in node.siblings:
            sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
            edge = (node_id, sibling_id)
            reverse_edge = (sibling_id, node_id)
            # Avoid duplicate edges
            if edge not in edges and reverse_edge not in edges:
                edges.append(edge)

    # Extract edge bandwidths from connection_blocks
    # connection_blocks: dict with (src_layer, dst_layer) as key
    for (src_layer_idx, dst_layer_idx), params in net_topo.connection_blocks.items():
        # Get bandwidth from params (assuming it has a bandwidth attribute)
        bandwidth = params.get('bandwidth', 1.0) if isinstance(params, dict) else 1.0

        # Get nodes in these layers
        if src_layer_idx < len(topology.layers) and dst_layer_idx < len(topology.layers):
            src_layer = topology.layers[src_layer_idx]
            dst_layer = topology.layers[dst_layer_idx]

            # Assign bandwidth to edges between these layers
            for src_node_id in src_layer:
                if src_node_id in all_nodes_dict:
                    src_node = all_nodes_dict[src_node_id]
                    for sibling in src_node.siblings:
                        sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                        if sibling_id in dst_layer:
                            edge = (src_node_id, sibling_id)
                            edge_bandwidths[edge] = bandwidth
                            # Also add reverse edge
                            edge_bandwidths[(sibling_id, src_node_id)] = bandwidth

    # If no bandwidth info, set default
    for edge in edges:
        if edge not in edge_bandwidths:
            edge_bandwidths[edge] = 1.0
            edge_bandwidths[(edge[1], edge[0])] = 1.0

    return SimplifiedTopology(
        nodes=nodes,
        edges=edges,
        node_types=node_types,
        edge_bandwidths=edge_bandwidths,
        original_net_topo=net_topo
    )



def is_connected(topo: SimplifiedTopology) -> bool:
    """
    Check if the topology graph is connected using BFS.

    Args:
        topo: SimplifiedTopology object

    Returns:
        True if connected, False otherwise
    """
    if not topo.nodes or not topo.edges:
        return False

    # Build adjacency list
    adj_list = {node: [] for node in topo.nodes}
    for src, dst in topo.edges:
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    # BFS from first node
    visited = set()
    queue = [topo.nodes[0]]
    visited.add(topo.nodes[0])

    while queue:
        node = queue.pop(0)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(topo.nodes)

