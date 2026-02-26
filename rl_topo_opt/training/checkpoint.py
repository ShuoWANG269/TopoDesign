"""
Checkpoint manager for saving and loading training state.
"""

import os
import torch
import pickle
from typing import Dict, Optional
from datetime import datetime
import glob


class CheckpointManager:
    """
    Manage checkpoints for training.

    Saves model, optimizer, topology, and training state every N steps.
    """

    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        save_interval: int = 100,
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N steps
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model,
        step: int,
        metrics: Dict = None,
        extra_data: Dict = None
    ):
        """
        Save checkpoint.

        Args:
            model: PPO model
            step: Current training step
            metrics: Training metrics
            extra_data: Additional data to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_step_{step}_{timestamp}.pkl'
        )

        checkpoint = {
            'step': step,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'model_state': model.policy.state_dict(),
        }

        # Add extra data
        if extra_data:
            checkpoint.update(extra_data)

        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Also save the full model
        model_path = checkpoint_path.replace('.pkl', '_model.zip')
        model.save(model_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, checkpoint_path: str, model):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: PPO model to load state into

        Returns:
            Checkpoint dict
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Load model state
        model.policy.load_state_dict(checkpoint['model_state'])

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Step: {checkpoint['step']}")
        print(f"Timestamp: {checkpoint['timestamp']}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pkl'))
        if not checkpoints:
            return None

        # Sort by step number - extract step from filename
        def get_step(path):
            filename = os.path.basename(path)
            # Extract number after "checkpoint_step_" and before next underscore
            start = filename.find('checkpoint_step_') + len('checkpoint_step_')
            end = filename.find('_', start)
            if end == -1:
                end = len(filename)
            return int(filename[start:end])

        checkpoints.sort(key=get_step)
        return checkpoints[-1]

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pkl'))
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split('_step_')[1].split('_')[0]))

        # Remove oldest checkpoints
        for checkpoint_path in checkpoints[:-self.max_checkpoints]:
            os.remove(checkpoint_path)
            # Also remove corresponding model file
            model_path = checkpoint_path.replace('.pkl', '_model.zip')
            if os.path.exists(model_path):
                os.remove(model_path)
            print(f"Removed old checkpoint: {checkpoint_path}")

    def should_save(self, step: int) -> bool:
        """
        Check if checkpoint should be saved at this step.

        Args:
            step: Current training step

        Returns:
            True if should save
        """
        return step % self.save_interval == 0

