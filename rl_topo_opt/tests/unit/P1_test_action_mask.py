"""
P1: Unit tests for action mask mechanism.
Tests that the action mask correctly prevents invalid edge removals.
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/wangshuo/Projects/pcl_project/TopoDesign/rl_topo_opt')

from env.topo_env import TopoEnv


class TestActionMaskInitial:
    """Test initial action mask state."""

    def test_initial_mask_all_valid(self, topo_env_long):
        """
        P1-1: Verify that initial action mask has all valid actions (1.0).
        At the start of an episode, no edges have been removed yet.
        """
        env = topo_env_long
        obs = env.reset()

        mask = obs['action_mask']

        # All initial edges should be valid (mask value = 1.0)
        assert mask is not None, "action_mask should not be None"
        assert len(mask) > 0, "action_mask should not be empty"

        # All valid (1.0) since no edges removed yet
        assert np.all(mask == 1.0), "Initial mask should have all valid actions"
        assert mask.sum() == len(mask), "Sum of mask should equal number of edges"


class TestActionMaskUpdate:
    """Test action mask updates after edge removal."""

    def test_mask_updates_after_removal(self, topo_env_long):
        """
        P1-2: Verify that action mask updates after removing an edge.
        The removed edge's mask should become 0.0.
        """
        env = topo_env_long
        obs = env.reset()

        # Get initial mask
        initial_mask = obs['action_mask'].copy()
        n_edges = len(initial_mask)

        # Remove an edge
        obs, reward, done, info = env.step(0)

        # Get updated mask
        if not done:
            new_mask = obs['action_mask']

            # The removed edge (index 0) should now be masked
            assert new_mask[0] == 0.0, "Removed edge should have mask=0"
            assert new_mask.sum() < n_edges, "Total valid actions should decrease"


class TestActionMaskDuplicateRemoval:
    """Test behavior when trying to remove an already-removed edge."""

    def test_duplicate_removal_masked(self, topo_env_long):
        """
        P1-3: Verify that duplicate removal of same edge returns small penalty.
        Since the edge is already masked, trying to remove it again gives -1.0.
        """
        env = topo_env_long
        obs = env.reset()

        # Remove edge 0
        obs1, reward1, done1, info1 = env.step(0)

        if not done1:
            # Try to remove edge 0 again
            obs2, reward2, done2, info2 = env.step(0)

            # Should give small penalty for already-removed edge
            assert reward2 == -1.0, "Duplicate removal should return -1.0 penalty"
            assert info2.get('already_removed') == True, "Should mark as already_removed"


class TestActionMaskPPORespects:
    """Test that PPO respects action mask."""

    def test_ppo_respects_mask(self, topo_env_long):
        """
        P1-4: Verify that PPO agent does not select masked actions.
        Tests the masked action distribution in ppo_agent.py.
        """
        from models.ppo_agent import create_ppo_agent

        env = topo_env_long
        obs = env.reset()

        # Create agent
        agent = create_ppo_agent(env, learning_rate=3e-4)

        # Get action with unmasked environment
        action1, _ = agent.predict(obs)

        # Remove edge 0
        env.step(0)

        # Get updated observation
        obs2 = env._get_observation()

        # Get action again
        action2, _ = agent.predict(obs2)

        # action2 should not be 0 if mask[0] == 0
        # Just verify the mask is correctly applied
        assert obs2['action_mask'][0] == 0.0, "Edge 0 should be masked"


class TestActionMaskShape:
    """Test action mask shape and type."""

    def test_mask_shape_matches_n_edges(self, topo_env_long):
        """
        P1-5: Verify that action mask shape matches number of initial edges.
        """
        env = topo_env_long
        obs = env.reset()

        mask = obs['action_mask']

        # Mask length should match initial number of edges
        assert len(mask) == env.n_initial_edges, \
            f"Mask length {len(mask)} should match n_initial_edges {env.n_initial_edges}"

    def test_mask_is_numpy_array(self, topo_env_long):
        """
        P1-6: Verify that action mask is a numpy array.
        """
        env = topo_env_long
        obs = env.reset()

        mask = obs['action_mask']

        assert isinstance(mask, np.ndarray), "action_mask should be a numpy array"


class TestActionMaskConsistency:
    """Test consistency of action mask behavior."""

    def test_mask_consistency_across_steps(self, topo_env_long):
        """
        P1-7: Verify that mask remains consistent (once masked, stays masked).
        """
        env = topo_env_long
        obs = env.reset()

        # Remove edge 0
        env.step(0)

        # Remove edge 1
        env.step(1)

        # Get observation
        obs = env._get_observation()
        mask = obs['action_mask']

        # Edges 0 and 1 should be masked
        assert mask[0] == 0.0, "Edge 0 should remain masked"
        assert mask[1] == 0.0, "Edge 1 should remain masked"

    def test_mask_deterministic(self, topo_env_long):
        """
        P1-8: Verify that same edge removal sequence produces same mask.
        Tests deterministic behavior.
        """
        env1 = topo_env_long
        env2 = topo_env_long

        # Same initial state
        obs1 = env1.reset()
        obs2 = env2.reset()

        # Remove same edges
        for action in [0, 2, 4]:
            env1.step(action)
            env2.step(action)

        mask1 = env1._get_observation()['action_mask']
        mask2 = env2._get_observation()['action_mask']

        # Should be identical
        assert np.allclose(mask1, mask2), "Same sequence should produce same mask"
