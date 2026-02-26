"""
Training loop for topology optimization.
"""

import numpy as np
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.topo_env import TopoEnv
from models.ppo_agent import create_ppo_agent
from training.checkpoint import CheckpointManager


class Trainer:
    """
    Trainer for topology optimization using PPO.
    """

    def __init__(
        self,
        initial_net_topo,
        total_timesteps: int = 10000,
        checkpoint_dir: str = './checkpoints',
        checkpoint_interval: int = 100,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            initial_net_topo: Initial ATOP NetTopology
            total_timesteps: Total training timesteps
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N steps
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            max_grad_norm: Max gradient norm
            tensorboard_log: Tensorboard log directory
        """
        self.initial_net_topo = initial_net_topo
        self.total_timesteps = total_timesteps

        # Create environment
        self.env = TopoEnv(initial_net_topo, max_steps=100)
        self.initial_edges = list(self.env.initial_topo.edges)

        # Create PPO agent
        self.model = create_ppo_agent(
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log
        )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_interval=checkpoint_interval
        )

        # Training state
        self.current_step = 0
        self.metrics_history = []

    def train(self, resume_from: Optional[str] = None):
        """
        Train the model.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_from, self.model)
            self.current_step = checkpoint['step']
            self.metrics_history = checkpoint.get('metrics_history', [])
            print(f"Resumed training from step {self.current_step}")

        # Custom callback for checkpointing
        class CheckpointCallback:
            def __init__(self, trainer):
                self.trainer = trainer

            def __call__(self, locals_dict, globals_dict):
                step = locals_dict['self'].num_timesteps
                self.trainer.current_step = step

                # Save checkpoint
                if self.trainer.checkpoint_manager.should_save(step):
                    metrics = {
                        'episode_reward': locals_dict.get('ep_info_buffer', []),
                        'loss': locals_dict.get('loss', None)
                    }
                    self.trainer.checkpoint_manager.save_checkpoint(
                        self.trainer.model,
                        step,
                        metrics=metrics,
                        extra_data={
                            'metrics_history': self.trainer.metrics_history,
                            'topology_params': {
                                'n_gpus': len(self.trainer.initial_net_topo.topology.layers[0]),
                                'depth': len(self.trainer.initial_net_topo.topology.layers),
                                'width': len(self.trainer.initial_net_topo.topology.layers[-1]) if len(self.trainer.initial_net_topo.topology.layers) > 1 else 3
                            },
                            'initial_edges': self.trainer.initial_edges
                        }
                    )

                return True

        callback = CheckpointCallback(self)

        # Train
        print(f"Starting training for {self.total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback
        )

        print("Training completed!")

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.current_step,
            metrics={'final': True},
            extra_data={
                'metrics_history': self.metrics_history,
                'topology_params': {
                    'n_gpus': len(self.initial_net_topo.topology.layers[0]),
                    'depth': len(self.initial_net_topo.topology.layers),
                    'width': len(self.initial_net_topo.topology.layers[-1]) if len(self.initial_net_topo.topology.layers) > 1 else 3
                },
                'initial_edges': self.initial_edges
            }
        )

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }

        print("\nEvaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

        return metrics

