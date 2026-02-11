"""
PPO agent with GNN policy network and action masking support.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import numpy as np

from models.gnn import GNNFeatureExtractor, obs_to_pyg_data


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor using GNN for SB3.

    Wraps GNNFeatureExtractor to be compatible with SB3's interface.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        """
        Initialize GNN features extractor.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
        """
        super(GNNFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Get node feature dimension from observation space
        node_feature_dim = observation_space['node_features'].shape[1]

        # Initialize GNN
        self.gnn = GNNFeatureExtractor(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=features_dim,
            num_layers=num_layers
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Dict with 'node_features', 'adjacency_matrix', 'action_mask'

        Returns:
            Features tensor (batch_size, features_dim)
        """
        device = next(self.gnn.parameters()).device

        # Handle batched observations
        if len(observations['node_features'].shape) == 3:
            # Batched: (batch_size, n_nodes, node_feature_dim)
            batch_size = observations['node_features'].shape[0]
            features_list = []

            for i in range(batch_size):
                obs = {
                    'node_features': observations['node_features'][i].cpu().numpy(),
                    'adjacency_matrix': observations['adjacency_matrix'][i].cpu().numpy()
                }
                data = obs_to_pyg_data(obs, device)
                features = self.gnn(data.x, data.edge_index)
                # NaN protection
                features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
                features_list.append(features)

            return torch.cat(features_list, dim=0)
        else:
            # Single observation
            obs = {
                'node_features': observations['node_features'].cpu().numpy(),
                'adjacency_matrix': observations['adjacency_matrix'].cpu().numpy()
            }
            data = obs_to_pyg_data(obs, device)
            features = self.gnn(data.x, data.edge_index)
            # NaN protection
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
            return features


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with action masking support.

    Masks invalid actions by setting their log probabilities to -inf.
    """

    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass with action masking.

        Args:
            obs: Observations dict
            deterministic: Whether to use deterministic actions

        Returns:
            Actions, values, log_probs
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply action mask
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            # Convert mask to log probabilities
            mask_log_probs = torch.log(action_mask + 1e-8)
            # Add mask to logits
            distribution.distribution.logits = distribution.distribution.logits + mask_log_probs

        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions with action masking.

        Args:
            obs: Observations dict
            actions: Actions to evaluate

        Returns:
            Values, log_probs, entropy
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply action mask
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            mask_log_probs = torch.log(action_mask + 1e-8)
            distribution.distribution.logits = distribution.distribution.logits + mask_log_probs

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy


def create_ppo_agent(env, learning_rate: float = 3e-4, **kwargs):
    """
    Create PPO agent with GNN policy.

    Args:
        env: Gym environment
        learning_rate: Learning rate
        **kwargs: Additional PPO arguments

    Returns:
        PPO agent
    """
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            hidden_dim=64,
            num_layers=3
        ),
    )

    model = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **kwargs
    )

    return model

