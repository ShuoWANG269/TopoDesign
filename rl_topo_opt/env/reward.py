"""
Reward function integrating cost, latency, and fault tolerance metrics.
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os

# Add ATOP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ATOP'))

from simulator.networkcost_scorer import network_cost
from simulator.forestcoll_scorer import forestcoll_score
from simulator.faulttolerance_scorer import fault_tolerance_score
from NSGAII.solution import NSGASolution


class RewardCalculator:
    """Calculate reward based on three metrics: cost, latency, fault tolerance."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize reward calculator.

        Args:
            weights: Dict with keys 'cost', 'latency', 'fault_tolerance'.
                    Default is equal weights (1/3 each).
        """
        if weights is None:
            weights = {'cost': 1/3, 'latency': 1/3, 'fault_tolerance': 1/3}

        self.weights = weights
        # Normalization parameters (will be updated during training)
        self.cost_max = 1.0
        self.latency_max = 1.0
        self.ft_min = 0.0

        # Cache for metrics
        self.cache = {}

    def compute_metrics(self, net_topo) -> Tuple[float, float, float]:
        """
        Compute all three metrics for a given topology.

        Args:
            net_topo: ATOP NetTopology object or SimplifiedTopology

        Returns:
            Tuple of (cost, latency, fault_tolerance)
        """
        # Import here to avoid circular dependency
        from converter import SimplifiedTopology

        # Create cache key from topology structure
        cache_key = self._get_cache_key(net_topo)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # If SimplifiedTopology, use original_net_topo for ATOP scorers
        if isinstance(net_topo, SimplifiedTopology):
            if net_topo.original_net_topo is None:
                # No original topology, return default values
                return (1.0, 1.0, 0.5)
            actual_net_topo = net_topo.original_net_topo
        else:
            actual_net_topo = net_topo

        # Wrap NetTopology in NSGASolution for ATOP scorers
        solution = NSGASolution(actual_net_topo, (0, 0, 0))

        # Compute metrics using ATOP scorers
        cost = network_cost(solution)
        latency = forestcoll_score(solution)
        ft = fault_tolerance_score(solution)

        # Convert to float and handle NaN/Inf
        try:
            cost = float(cost)
            latency = float(latency)
            ft = float(ft)
        except (TypeError, ValueError):
            cost, latency, ft = 1.0, 1.0, 1.0  # Fallback values

        # Replace NaN/Inf with fallback values
        if np.isnan(cost) or np.isinf(cost):
            cost = 1.0
        if np.isnan(latency) or np.isinf(latency):
            latency = 1.0
        if np.isnan(ft) or np.isinf(ft):
            ft = 0.5  # Middle ground for fault tolerance

        # Cache result
        self.cache[cache_key] = (cost, latency, ft)

        return cost, latency, ft

    def calculate_reward(self, net_topo, is_connected: bool = True) -> float:
        """
        Calculate reward for a given topology.

        Args:
            net_topo: ATOP NetTopology object
            is_connected: Whether the topology is connected

        Returns:
            Reward value (higher is better)
        """
        # Penalty for disconnected graph
        if not is_connected:
            return -1000.0

        # Compute metrics
        cost, latency, ft = self.compute_metrics(net_topo)

        # Update normalization parameters
        self.cost_max = max(self.cost_max, cost)
        self.latency_max = max(self.latency_max, latency)
        self.ft_min = min(self.ft_min, ft)

        # Normalize metrics (lower is better for cost and latency, higher is better for ft)
        cost_norm = cost / self.cost_max if self.cost_max > 0 else 0
        latency_norm = latency / self.latency_max if self.latency_max > 0 else 0
        # ft_norm: handle edge case where ft_min >= 1.0 (all values are at or above 1)
        ft_range = max(1.0 - self.ft_min, 1e-6)  # Prevent division by zero
        ft_norm = (ft - self.ft_min) / ft_range

        # Weighted sum (negative because we want to minimize cost and latency)
        reward = -(
            self.weights['cost'] * cost_norm +
            self.weights['latency'] * latency_norm -
            self.weights['fault_tolerance'] * ft_norm  # Negative because higher ft is better
        )

        return reward

    def _get_cache_key(self, net_topo) -> str:
        """Generate cache key from topology structure."""
        # Import here to avoid circular dependency
        from converter import SimplifiedTopology

        # Handle SimplifiedTopology
        if isinstance(net_topo, SimplifiedTopology):
            edges = sorted(set(tuple(sorted(e)) for e in net_topo.edges))
            return str(edges)

        # Handle NetTopology
        edges = []
        nodes_dict = net_topo.topology.nodes

        for node_id, node in nodes_dict.items():
            for sibling in node.siblings:
                sibling_id = sibling.node_id if hasattr(sibling, 'node_id') else sibling
                edge = tuple(sorted([node_id, sibling_id]))
                edges.append(edge)

        edges = sorted(set(edges))
        return str(edges)


    def clear_cache(self):
        """Clear the metrics cache."""
        self.cache.clear()

