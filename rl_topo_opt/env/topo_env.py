"""
RL Environment for topology optimization.
Action: Remove edges from the topology.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from converter import SimplifiedTopology, convert_atop_to_simplified, is_connected
from env.reward import RewardCalculator


class TopoEnv(gym.Env):
    """
    Gym environment for topology optimization.

    Action space: Discrete(n_edges) - select an edge to remove
    State space: Node features + Adjacency matrix
    """

    def __init__(self, initial_net_topo, max_steps: int = 100):
        """
        Initialize environment.

        Args:
            initial_net_topo: Initial ATOP NetTopology object or SimplifiedTopology
            max_steps: Maximum steps per episode
        """
        super(TopoEnv, self).__init__()

        self.initial_net_topo = initial_net_topo
        self.max_steps = max_steps
        self.current_step = 0

        # Convert to simplified topology if needed
        if isinstance(initial_net_topo, SimplifiedTopology):
            self.initial_topo = initial_net_topo
        else:
            self.initial_topo = convert_atop_to_simplified(initial_net_topo)
        self.current_topo = self.initial_topo.copy()

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Action space: select an edge to remove
        self.n_initial_edges = len(self.initial_topo.edges)
        self.action_space = spaces.Discrete(self.n_initial_edges)

        # State space: node features + adjacency matrix
        n_nodes = len(self.initial_topo.nodes)
        self.n_nodes = n_nodes

        # Node features: [node_type, degree, layer]
        node_feature_dim = 3
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=0, high=1, shape=(n_nodes, node_feature_dim), dtype=np.float32
            ),
            'adjacency_matrix': spaces.Box(
                low=0, high=1, shape=(n_nodes, n_nodes), dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=(self.n_initial_edges,), dtype=np.float32
            )
        })

        # Track removed edges
        self.removed_edges = set()

    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.current_topo = self.initial_topo.copy()
        self.current_step = 0
        self.removed_edges = set()
        self.reward_calculator.clear_cache()

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Edge index to remove

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        self.current_step += 1

        # Get the edge to remove
        if action >= len(self.initial_topo.edges):
            # Invalid action
            return self._get_observation(), -10.0, True, {'error': 'invalid_action'}

        edge = self.initial_topo.edges[action]

        # Check if edge already removed
        if action in self.removed_edges:
            # Already removed, give small penalty
            reward = -1.0
            done = False
            info = {'already_removed': True}
            return self._get_observation(), reward, done, info

        # Remove edge from current topology
        self.current_topo.remove_edge(edge)
        self.removed_edges.add(action)

        # Update original NetTopology (remove from siblings)
        self._update_net_topo_siblings(edge)

        # Check connectivity
        connected = is_connected(self.current_topo)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.current_topo.original_net_topo,
            is_connected=connected
        )

        # Check termination conditions
        done = False
        info = {}

        if not connected:
            done = True
            info['termination_reason'] = 'disconnected'
        elif self.current_step >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        elif len(self.current_topo.edges) < self.n_nodes - 1:
            # Too few edges (less than minimum spanning tree)
            done = True
            info['termination_reason'] = 'too_few_edges'

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Dict:
        """Get current observation."""
        node_features = self._get_node_features()
        adjacency_matrix = self.current_topo.get_adjacency_matrix()
        action_mask = self._get_action_mask()

        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency_matrix,
            'action_mask': action_mask
        }

    def _get_node_features(self) -> np.ndarray:
        """
        Extract node features.

        Features:
        - node_type: 1 for GPU, 0 for SWITCH
        - degree: normalized node degree
        - layer: normalized layer index
        """
        n_nodes = len(self.current_topo.nodes)
        features = np.zeros((n_nodes, 3), dtype=np.float32)

        # Build adjacency list for degree calculation
        adj_list = {node: [] for node in self.current_topo.nodes}
        for src, dst in self.current_topo.edges:
            adj_list[src].append(dst)
            adj_list[dst].append(src)

        # Get layer information from original topology
        node_to_layer = {}
        max_layer = 0

        # Handle both NetTopology and SimplifiedTopology
        if isinstance(self.initial_net_topo, SimplifiedTopology):
            # For SimplifiedTopology, use simple heuristic: GPU nodes are layer 0
            for node_id in self.current_topo.nodes:
                node_type = self.current_topo.node_types.get(node_id, 'SWITCH')
                node_to_layer[node_id] = 0 if node_type == 'GPU' else 1
            max_layer = 1
        else:
            # For NetTopology, use actual layer information
            for layer_idx, layer in enumerate(self.initial_net_topo.topology.layers):
                for node_id in layer:
                    node_to_layer[node_id] = layer_idx
            max_layer = len(self.initial_net_topo.topology.layers) - 1

        max_degree = max(len(neighbors) for neighbors in adj_list.values()) if adj_list else 1

        for idx, node_id in enumerate(self.current_topo.nodes):
            # Node type
            features[idx, 0] = 1.0 if self.current_topo.node_types[node_id] == 'GPU' else 0.0
            # Degree (normalized)
            features[idx, 1] = len(adj_list[node_id]) / max_degree if max_degree > 0 else 0
            # Layer (normalized)
            features[idx, 2] = node_to_layer.get(node_id, 0) / max_layer if max_layer > 0 else 0

        return features

    def _get_action_mask(self) -> np.ndarray:
        """
        Get action mask (1 for valid actions, 0 for invalid).

        Invalid actions:
        - already removed edges
        - edges whose removal would disconnect the graph (bridge edges)
        """
        mask = np.ones(self.n_initial_edges, dtype=np.float32)

        # Mark removed edges as invalid
        for action_idx in self.removed_edges:
            mask[action_idx] = 0.0

        # Mark bridge edges (removal would disconnect) as invalid
        # Use current topology (with already removed edges deleted)
        bridge_edges = self._find_bridge_edges()
        for action_idx, edge in enumerate(self.initial_topo.edges):
            if edge in bridge_edges or (edge[1], edge[0]) in bridge_edges:
                mask[action_idx] = 0.0

        return mask

    def _find_bridge_edges(self) -> set:
        """
        Find all bridge edges in the current topology.

        A bridge edge is an edge whose removal would disconnect the graph.
        Uses Tarjan's algorithm for bridge finding.
        """
        graph = {node: [] for node in self.current_topo.nodes}

        # Build adjacency list from current edges
        for src, dst in self.current_topo.edges:
            if src not in graph:
                graph[src] = []
            if dst not in graph:
                graph[dst] = []
            graph[src].append(dst)
            graph[dst].append(src)

        if not graph:
            return set()

        # Tarjan's algorithm for bridges
        discovery_time = {}
        low_link = {}
        bridges = set()
        visited = set()
        time_counter = [0]
        parent = {}

        def dfs(u):
            visited.add(u)
            discovery_time[u] = time_counter[0]
            low_link[u] = time_counter[0]
            time_counter[0] += 1

            for v in graph[u]:
                if v not in visited:
                    parent[v] = u
                    dfs(v)
                    low_link[u] = min(low_link[u], low_link[v])

                    # Check if edge (u, v) is a bridge
                    if low_link[v] > discovery_time[u]:
                        bridges.add((u, v))
                        bridges.add((v, u))
                elif v != parent.get(u):
                    low_link[u] = min(low_link[u], discovery_time[v])

        # Handle disconnected graph - run DFS from each component
        for node in graph:
            if node not in visited:
                dfs(node)

        return bridges

    def _update_net_topo_siblings(self, edge: Tuple[int, int]):
        """
        Update the original NetTopology by removing edge from siblings.

        Args:
            edge: Edge to remove (src, dst)
        """
        src_id, dst_id = edge

        # Get nodes dict from topology
        nodes_dict = self.current_topo.original_net_topo.topology.nodes

        # Remove edge from siblings
        if src_id in nodes_dict:
            src_node = nodes_dict[src_id]
            src_node.siblings = {s for s in src_node.siblings if (s.node_id if hasattr(s, 'node_id') else s) != dst_id}

        if dst_id in nodes_dict:
            dst_node = nodes_dict[dst_id]
            dst_node.siblings = {s for s in dst_node.siblings if (s.node_id if hasattr(s, 'node_id') else s) != src_id}


    def render(self, mode='human'):
        """Render the environment (optional)."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Nodes: {len(self.current_topo.nodes)}")
        print(f"Edges: {len(self.current_topo.edges)}/{self.n_initial_edges}")
        print(f"Removed edges: {len(self.removed_edges)}")
