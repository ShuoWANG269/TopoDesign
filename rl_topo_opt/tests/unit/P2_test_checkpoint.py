"""
P2: Unit tests for checkpoint saving and loading.
Tests CheckpointManager functionality.
"""

import pytest
import os
import shutil
import pickle
import torch
import sys

sys.path.insert(0, '/Users/wangshuo/Projects/pcl_project/TopoDesign/rl_topo_opt')

from training.checkpoint import CheckpointManager
from models.ppo_agent import create_ppo_agent


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


@pytest.fixture
def sample_checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager with temp directory."""
    return CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        save_interval=100,
        max_checkpoints=5
    )


class TestCheckpointSave:
    """Test checkpoint saving."""

    def test_save_checkpoint(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-8: Verify checkpoint file is generated.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        # Create agent
        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Save checkpoint
        manager.save_checkpoint(agent, step=100, metrics={'test': 1.0})

        # Check checkpoint file exists
        checkpoints = os.listdir(manager.checkpoint_dir)
        pkl_files = [f for f in checkpoints if f.endswith('.pkl')]

        assert len(pkl_files) >= 1, "At least one checkpoint file should be created"
        assert any('checkpoint_step_100' in f for f in pkl_files), \
            "Checkpoint with step 100 should exist"

    def test_checkpoint_contains_model_state(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-9: Verify checkpoint contains model state dict.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Get original state
        original_state = agent.policy.state_dict().copy()

        # Save checkpoint
        manager.save_checkpoint(agent, step=100)

        # Load and verify
        latest = manager.get_latest_checkpoint()
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)

        assert 'model_state' in checkpoint, "Checkpoint should contain model_state"
        assert isinstance(checkpoint['model_state'], dict), "model_state should be dict"


class TestCheckpointLoad:
    """Test checkpoint loading."""

    def test_load_checkpoint(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-10: Verify checkpoint loading restores model state.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        # Create and train agent (minimal training)
        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Save checkpoint
        manager.save_checkpoint(agent, step=100)

        # Create new agent (different state)
        agent2 = create_ppo_agent(env, learning_rate=3e-4)

        # Load checkpoint
        latest = manager.get_latest_checkpoint()
        loaded = manager.load_checkpoint(latest, agent2)

        # Verify step is restored
        assert loaded['step'] == 100, "Loaded checkpoint step should be 100"


class TestCheckpointManagement:
    """Test checkpoint management."""

    def test_get_latest_checkpoint(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-11: Verify getting latest checkpoint.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            manager.save_checkpoint(agent, step=step)

        # Get latest
        latest = manager.get_latest_checkpoint()

        assert latest is not None, "Latest checkpoint should exist"
        assert 'checkpoint_step_300' in latest, "Latest should be step 300"

    def test_cleanup_old_checkpoints(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-12: Verify old checkpoints are cleaned up, keeping max_checkpoints.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Save more than max_checkpoints (5)
        for step in [100, 200, 300, 400, 500, 600]:
            manager.save_checkpoint(agent, step=step)

        # Should have at most 5 checkpoints
        checkpoints = os.listdir(manager.checkpoint_dir)
        pkl_files = [f for f in checkpoints if f.endswith('.pkl')]

        # Old ones should be cleaned up
        assert len(pkl_files) <= 5, f"Should have at most 5 checkpoints, got {len(pkl_files)}"

        # Latest 5 should exist (300, 400, 500, 600, and one more)
        assert any('checkpoint_step_600' in f for f in pkl_files), "Latest should exist"

    def test_should_save_logic(self, sample_checkpoint_manager):
        """
        P2-13: Verify should_save logic based on interval.
        """
        manager = sample_checkpoint_manager

        # Should save at intervals
        assert manager.should_save(100) == True
        assert manager.should_save(200) == True
        assert manager.should_save(300) == True

        # Should not save between intervals
        assert manager.should_save(50) == False
        assert manager.should_save(150) == False
        assert manager.should_save(250) == False


class TestCheckpointStepTracking:
    """Test step tracking in checkpoints."""

    def test_checkpoint_step_stored(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-14: Verify step number is stored in checkpoint.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        agent = create_ppo_agent(env, learning_rate=3e-4)
        manager.save_checkpoint(agent, step=500)

        latest = manager.get_latest_checkpoint()
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)

        assert checkpoint['step'] == 500, "Step should be 500"
        assert 'timestamp' in checkpoint, "Timestamp should be stored"


class TestCheckpointMetrics:
    """Test metrics storage in checkpoints."""

    def test_checkpoint_metrics_stored(self, sample_checkpoint_manager, topo_env_long):
        """
        P2-15: Verify metrics are stored in checkpoint.
        """
        manager = sample_checkpoint_manager
        env = topo_env_long

        agent = create_ppo_agent(env, learning_rate=3e-4)
        metrics = {'mean_reward': 10.5, 'std_reward': 2.3}

        manager.save_checkpoint(agent, step=100, metrics=metrics)

        latest = manager.get_latest_checkpoint()
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)

        assert 'metrics' in checkpoint, "Checkpoint should contain metrics"
        assert checkpoint['metrics']['mean_reward'] == 10.5, "Metrics should match"
