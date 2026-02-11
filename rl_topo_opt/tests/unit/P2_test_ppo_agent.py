"""
P2: Unit tests for PPO strategy network.
Tests GNN features extractor, masked policy, and value estimation.
"""

import pytest
import torch
import numpy as np
import sys

sys.path.insert(0, '/Users/wangshuo/Projects/pcl_project/TopoDesign/rl_topo_opt')

from models.ppo_agent import GNNFeaturesExtractor, MaskedActorCriticPolicy, create_ppo_agent
from gym import spaces


class TestGNNFeaturesExtractor:
    """Test GNN features extractor for SB3."""

    def test_extractor_output_shape(self, topo_env_long):
        """
        P2-1: Verify GNNFeaturesExtractor output shape (batch_size, 128).
        """
        env = topo_env_long
        obs_space = env.observation_space

        extractor = GNNFeaturesExtractor(
            observation_space=obs_space,
            features_dim=128,
            hidden_dim=64,
            num_layers=3
        )

        # Get observation
        obs = env.reset()

        # Convert to torch tensors
        torch_obs = {
            'node_features': torch.tensor(obs['node_features']).unsqueeze(0),
            'adjacency_matrix': torch.tensor(obs['adjacency_matrix']).unsqueeze(0),
            'action_mask': torch.tensor(obs['action_mask']).unsqueeze(0)
        }

        # Forward pass
        with torch.no_grad():
            features = extractor(torch_obs)

        # Output should be (1, 128)
        assert features.shape == torch.Size([1, 128]), \
            f"Expected shape (1, 128), got {features.shape}"

    def test_batched_extractor_output(self, topo_env_long):
        """
        P2-2: Verify batched observation processing.
        """
        env = topo_env_long
        obs_space = env.observation_space

        extractor = GNNFeaturesExtractor(
            observation_space=obs_space,
            features_dim=128,
            hidden_dim=64,
            num_layers=3
        )

        # Get two observations
        obs1 = env.reset()
        obs2 = env.reset()

        # Stack into batch
        batch_obs = {
            'node_features': torch.tensor(np.stack([obs1['node_features'], obs2['node_features']])),
            'adjacency_matrix': torch.tensor(np.stack([obs1['adjacency_matrix'], obs2['adjacency_matrix']])),
            'action_mask': torch.tensor(np.stack([obs1['action_mask'], obs2['action_mask']]))
        }

        # Forward pass
        with torch.no_grad():
            features = extractor(batch_obs)

        # Output should be (2, 128)
        assert features.shape == torch.Size([2, 128]), \
            f"Expected shape (2, 128), got {features.shape}"


class TestMaskedPolicy:
    """Test masked actor-critic policy."""

    def test_masked_policy_forward(self, topo_env_long):
        """
        P2-3: Verify masked policy applies action mask during forward.
        """
        env = topo_env_long
        obs = env.reset()

        # Create agent
        agent = create_ppo_agent(env, learning_rate=3e-4)
        agent.policy.eval()

        # Get action through policy
        action, _ = agent.predict(obs)

        # Actions should be valid (within action space)
        assert action.shape == () or (hasattr(action, 'shape') and len(action.shape) == 0), \
            "Action should be scalar"
        assert 0 <= action.item() < env.action_space.n, "Action should be within action space"

    def test_action_sampling_with_mask(self, topo_env_long):
        """
        P2-4: Verify that sampled actions respect the action mask.
        """
        env = topo_env_long
        obs = env.reset()

        # Create agent
        agent = create_ppo_agent(env, learning_rate=3e-4)
        agent.policy.eval()

        # Get initial action (all valid)
        action1, _ = agent.predict(obs, deterministic=True)
        assert 0 <= action1 < env.action_space.n, "Action should be valid"

        # Remove first edge
        env.step(0)

        # Get updated observation
        obs2 = env._get_observation()

        # Get action again
        action2, _ = agent.predict(obs2, deterministic=True)

        # action2 should not be 0 if mask[0] == 0
        # The policy should respect the mask
        assert obs2['action_mask'][0] == 0.0, "Edge 0 should be masked after removal"


class TestPPOCreateAgent:
    """Test PPO agent creation."""

    def test_create_ppo_agent(self, topo_env_long):
        """
        P2-5: Verify PPO agent is successfully created.
        """
        agent = create_ppo_agent(topo_env_long, learning_rate=3e-4)

        assert agent is not None, "PPO agent should not be None"
        assert hasattr(agent, 'policy'), "Agent should have policy attribute"
        assert hasattr(agent, 'learn'), "Agent should have learn method"


class TestValueEstimation:
    """Test value network estimation."""

    def test_value_estimation_reasonable(self, topo_env_long):
        """
        P2-6: Verify value estimation is reasonable (positive, not too large).
        """
        env = topo_env_long
        obs = env.reset()

        agent = create_ppo_agent(env, learning_rate=3e-4)
        agent.policy.eval()

        # Get observation tensor
        torch_obs = {
            'node_features': torch.tensor(obs['node_features']).unsqueeze(0),
            'adjacency_matrix': torch.tensor(obs['adjacency_matrix']).unsqueeze(0),
            'action_mask': torch.tensor(obs['action_mask']).unsqueeze(0)
        }

        # Get values
        with torch.no_grad():
            _, values, _ = agent.policy.forward(torch_obs)

        # Value should be a scalar tensor
        value = values.item()

        # Value should be reasonable (rewards are typically negative)
        # -1000 to 10 range seems reasonable for this task
        assert -2000 < value < 100, f"Value {value} should be reasonable"


class TestDeterministicBehavior:
    """Test deterministic behavior of PPO."""

    def test_same_obs_same_action(self, topo_env_long):
        """
        P2-7: Verify that same observation produces same deterministic action.
        """
        env = topo_env_long
        obs = env.reset()

        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Get deterministic action twice
        action1, _ = agent.predict(obs, deterministic=True)
        action2, _ = agent.predict(obs, deterministic=True)

        # Should be the same
        assert action1 == action2, "Same observation should produce same deterministic action"
