"""
P1: Unit tests for GNN forward propagation.
Tests GCN feature extraction and graph embedding generation.
"""

import pytest
import torch
import numpy as np
import sys

sys.path.insert(0, '/Users/wangshuo/Projects/pcl_project/TopoDesign/rl_topo_opt')

from models.gnn import GNNFeatureExtractor, obs_to_pyg_data, batch_obs_to_pyg_batch


class TestGNNForwardShape:
    """Test GCN output shapes."""

    def test_forward_shape(self, simple_linear_topo):
        """
        P1-17: Verify GCN forward returns correct shape (1, 128).
        Single graph should produce graph-level embedding.
        """
        # Create GCN with correct parameters
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.eval()

        # Create observation from simple topology
        obs = create_linear_obs(simple_linear_topo)

        # Convert to PyG Data (returns numpy-based Data)
        data = obs_to_pyg_data(obs)

        # Forward pass
        with torch.no_grad():
            out = feature_extractor(data.x, data.edge_index, data.batch)

        # Output should be (1, 128) for single graph
        assert out.shape == torch.Size([1, 128]), \
            f"Expected shape (1, 128), got {out.shape}"

    def test_batched_forward_shape(self, simple_linear_topo):
        """
        P1-18: Verify GCN batched forward returns correct shape (batch_size, 128).
        Batch of graphs should produce batch of embeddings.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.eval()

        # Create batch of observations (3 identical graphs)
        batch_size = 3
        obs_list = [create_linear_obs(simple_linear_topo) for _ in range(batch_size)]

        # Convert to PyG Batch
        batch = batch_obs_to_pyg_batch(obs_list)

        # Forward pass
        with torch.no_grad():
            out = feature_extractor(batch.x, batch.edge_index, batch.batch)

        # Output should be (batch_size, 128)
        assert out.shape == torch.Size([batch_size, 128]), \
            f"Expected shape ({batch_size}, 128), got {out.shape}"


class TestObsToPyGData:
    """Test Gym observation to PyG Data conversion."""

    def test_obs_to_pyg_data(self, simple_linear_topo):
        """
        P1-19: Verify obs_to_pyg_data produces valid PyG Data object.
        """
        obs = create_linear_obs(simple_linear_topo)
        data = obs_to_pyg_data(obs)

        # Should have x attribute (node features)
        assert hasattr(data, 'x'), "Data should have x attribute"

        # Should have edge_index attribute
        assert hasattr(data, 'edge_index'), "Data should have edge_index"

        # Edge index should be 2 x n_edges
        assert data.edge_index.shape[0] == 2, "edge_index should be 2 x n_edges"

    def test_edge_index_extraction(self, simple_linear_topo):
        """
        P1-20: Verify that edge_index is correctly extracted from adjacency matrix.
        """
        obs = create_linear_obs(simple_linear_topo)
        data = obs_to_pyg_data(obs)

        adj = obs['adjacency_matrix']
        n_nodes = adj.shape[0]

        # Count edges in adjacency matrix
        n_edges_adj = (adj > 0).sum() // 2  # Undirected, divide by 2

        # Count edges in edge_index (undirected = 2 directed)
        n_edges_idx = data.edge_index.shape[1]

        # They should match (edge_index has both directions)
        assert n_edges_idx == n_edges_adj * 2, \
            f"edge_index edges {n_edges_idx} should be 2x adjacency edges {n_edges_adj}"


class TestGNNGradientFlow:
    """Test gradient flow through GCN."""

    def test_gradient_flow(self, simple_linear_topo):
        """
        P1-21: Verify that gradients can flow through GCN.
        Training mode should compute gradients.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.train()

        obs = create_linear_obs(simple_linear_topo)
        data = obs_to_pyg_data(obs)

        # Forward pass
        out = feature_extractor(data.x, data.edge_index, data.batch)

        # Loss backward
        loss = out.sum()
        loss.backward()

        # Gradients should exist (PyTorch Geometric uses lin.weight)
        assert feature_extractor.convs[0].lin.weight.grad is not None, \
            "Gradients should flow through first conv layer"
        assert feature_extractor.convs[0].lin.weight.grad.abs().sum() > 0, \
            "Gradients should be non-zero"


class TestGNNVariableGraphSize:
    """Test GCN robustness to different graph sizes."""

    def test_variable_graph_size(self, simple_linear_topo, simple_star_topo):
        """
        P1-22: Verify GCN handles different graph sizes.
        Should work for both linear (4 nodes) and star (5 nodes) topologies.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.eval()

        # Test linear topology (4 nodes)
        obs1 = create_linear_obs(simple_linear_topo)
        data1 = obs_to_pyg_data(obs1)
        with torch.no_grad():
            out1 = feature_extractor(data1.x, data1.edge_index, data1.batch)
        assert out1.shape == torch.Size([1, 128])

        # Test star topology (5 nodes)
        obs2 = create_linear_obs(simple_star_topo)
        data2 = obs_to_pyg_data(obs2)
        with torch.no_grad():
            out2 = feature_extractor(data2.x, data2.edge_index, data2.batch)
        assert out2.shape == torch.Size([1, 128])


class TestGNNNodeFeatures:
    """Test node feature handling."""

    def test_node_features_shape(self, simple_linear_topo):
        """
        P1-23: Verify node features have correct shape (n_nodes, 8).
        Each node should have 8-dimensional feature vector.
        """
        obs = create_linear_obs(simple_linear_topo)
        data = obs_to_pyg_data(obs)

        n_nodes = len(simple_linear_topo.nodes)

        # x should be n_nodes x in_channels
        assert data.x.shape == torch.Size([n_nodes, 8]), \
            f"Expected x shape ({n_nodes}, 8), got {data.x.shape}"

    def test_node_features_different_topologies(self, simple_linear_topo, simple_star_topo):
        """
        P1-24: Verify node features reflect topology differences.
        Different topologies should produce different embeddings.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.eval()

        obs1 = create_linear_obs(simple_linear_topo)
        obs2 = create_linear_obs(simple_star_topo)

        data1 = obs_to_pyg_data(obs1)
        data2 = obs_to_pyg_data(obs2)

        with torch.no_grad():
            out1 = feature_extractor(data1.x, data1.edge_index, data1.batch)
            out2 = feature_extractor(data2.x, data2.edge_index, data2.batch)

        # Different topologies should produce different embeddings
        # (unless they're isomorphic, but linear and star are not)
        assert not torch.allclose(out1, out2), \
            "Different topologies should produce different embeddings"


class TestGNNBatchNorm:
    """Test batch normalization in GCN."""

    def test_batch_norm_exists(self, simple_linear_topo):
        """
        P1-25: Verify that batch normalization layers exist in GCN.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )

        # Should have batch norm layers
        assert hasattr(feature_extractor, 'batch_norms'), "GCN should have batch_norms attribute"
        assert len(feature_extractor.batch_norms) == 3, "Should have 3 batch norm layers"


class TestGNNDeterministic:
    """Test GCN deterministic behavior."""

    def test_same_input_same_output(self, simple_linear_topo):
        """
        P1-26: Verify GCN produces deterministic outputs.
        Same input should produce same output.
        """
        feature_extractor = GNNFeatureExtractor(
            node_feature_dim=8,
            hidden_dim=32,
            output_dim=128,
            num_layers=3
        )
        feature_extractor.eval()

        obs = create_linear_obs(simple_linear_topo)
        data = obs_to_pyg_data(obs)

        # Run twice
        with torch.no_grad():
            out1 = feature_extractor(data.x, data.edge_index, data.batch)
            out2 = feature_extractor(data.x, data.edge_index, data.batch)

        assert torch.allclose(out1, out2), "Same input should produce same output"


# Helper function to create observation from SimplifiedTopology
def create_linear_obs(topo):
    """Create Gym observation dict from SimplifiedTopology using numpy arrays."""
    n_nodes = len(topo.nodes)
    n_edges = len(topo.edges)

    # Node features: one-hot encoding of node types
    node_features = np.zeros((n_nodes, 8), dtype=np.float32)
    for i, node_id in enumerate(topo.nodes):
        node_type = topo.node_types.get(node_id, 'GPU')
        if node_type == 'GPU':
            node_features[i, 0] = 1.0
        elif node_type == 'SWITCH':
            node_features[i, 1] = 1.0
        else:
            node_features[i, 2] = 1.0

    # Adjacency matrix
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in topo.edges:
        adj[u, v] = 1.0

    # Action mask (all valid initially)
    action_mask = np.ones(n_edges, dtype=np.float32)

    return {
        'node_features': node_features,
        'adjacency_matrix': adj,
        'action_mask': action_mask,
        'edge_list': topo.edges
    }
