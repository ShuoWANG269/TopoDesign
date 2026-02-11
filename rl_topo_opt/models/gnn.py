"""
GNN-based feature extractor for topology state representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Tuple
from gym import spaces


class GNNFeatureExtractor(nn.Module):
    """
    GNN feature extractor using GCN layers.

    Extracts graph-level features from node features and adjacency matrix.
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Initialize GNN feature extractor.

        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of GCN layers
        """
        super(GNNFeatureExtractor, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment (num_nodes,) for batched graphs

        Returns:
            Graph-level features (batch_size, output_dim)
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

            # NaN protection
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # Global pooling
        if batch is None:
            # Single graph
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            # Batched graphs
            x = global_mean_pool(x, batch)

        # Final NaN check
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        return x


def obs_to_pyg_data(obs: Dict, device: torch.device = None) -> Data:
    """
    Convert observation dict to PyTorch Geometric Data object.

    Args:
        obs: Observation dict with 'node_features' and 'adjacency_matrix'
        device: Device to place tensors on

    Returns:
        PyG Data object
    """
    if device is None:
        device = torch.device('cpu')

    # Extract node features
    node_features = torch.from_numpy(obs['node_features']).float().to(device)

    # Extract edge index from adjacency matrix
    adj_matrix = obs['adjacency_matrix']
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.from_numpy(edge_index).long().to(device)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index)

    return data


def batch_obs_to_pyg_batch(obs_list: list, device: torch.device = None) -> Batch:
    """
    Convert list of observations to PyTorch Geometric Batch.

    Args:
        obs_list: List of observation dicts
        device: Device to place tensors on

    Returns:
        PyG Batch object
    """
    data_list = [obs_to_pyg_data(obs, device) for obs in obs_list]
    return Batch.from_data_list(data_list)

