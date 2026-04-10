"""Tests for neural action selector."""

import pytest
import numpy as np
import torch
import tempfile
import os

from .neural_action_selector import (
    ObservationActionEncoder,
    QNetwork,
    ReplayBuffer,
    NeuralActionSelector,
    Transition,
)


# ============================================================
# Encoder Tests
# ============================================================

class TestObservationActionEncoder:
    """Tests for ObservationActionEncoder."""

    def test_tuple_observation_encoding(self):
        """Test encoding of tuple observations (TaxiNet style)."""
        observations = [(0, 0), (1, 1), (2, -1)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # Check dimensions
        assert encoder.obs_dim == 2
        assert encoder.obs_mode == "tuple"

        # Encode observation
        obs_vec = encoder.encode_observation((0, 0))
        assert obs_vec.shape == (2,)
        assert np.allclose(obs_vec, np.array([0.0, 0.0]))

        obs_vec = encoder.encode_observation((1, -1))
        assert np.allclose(obs_vec, np.array([1.0, -1.0]))

    def test_discrete_action_encoding(self):
        """Test one-hot encoding of discrete actions."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # Check dimensions
        assert encoder.action_dim == 3
        assert encoder.action_mode == "discrete"

        # Encode actions
        action_vec = encoder.encode_action(-1)
        assert action_vec.shape == (3,)
        assert np.allclose(action_vec, np.array([1.0, 0.0, 0.0]))

        action_vec = encoder.encode_action(0)
        assert np.allclose(action_vec, np.array([0.0, 1.0, 0.0]))

        action_vec = encoder.encode_action(1)
        assert np.allclose(action_vec, np.array([0.0, 0.0, 1.0]))

    def test_history_encoding_with_padding(self):
        """Test history encoding with zero padding."""
        observations = [(0, 0), (1, 1)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # Short history (less than window)
        history = [((0, 0), -1), ((1, 1), 0)]
        window_size = 5

        history_vec = encoder.encode_history(history, window_size)

        # Expected dimension: 5 * (2 + 3) = 25
        assert history_vec.shape == (25,)

        # First 3 pairs should be zeros (padding)
        pair_dim = 5
        assert np.allclose(history_vec[:3*pair_dim], 0.0)

    def test_history_encoding_full_window(self):
        """Test history encoding with full window."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # Full window
        history = [((i, i), -1) for i in range(10)]
        window_size = 10

        history_vec = encoder.encode_history(history, window_size)

        # Expected dimension: 10 * (2 + 3) = 50
        assert history_vec.shape == (50,)

    def test_history_encoding_truncation(self):
        """Test that only last N pairs are used."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # History longer than window
        history = [((i, i), 0) for i in range(20)]
        window_size = 5

        history_vec = encoder.encode_history(history, window_size)

        # Should use last 5 pairs
        assert history_vec.shape == (25,)

    def test_get_input_dim(self):
        """Test input dimension calculation."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        encoder = ObservationActionEncoder(observations, actions)

        # obs_dim=2, action_dim=3, window=10
        expected_dim = 10 * (2 + 3)
        assert encoder.get_input_dim(10) == expected_dim

    def test_config_serialization(self):
        """Test encoder config serialization and loading."""
        observations = [(0, 0), (1, 1)]
        actions = [-1, 0, 1]

        encoder1 = ObservationActionEncoder(observations, actions)
        config = encoder1.get_config()

        # Check config contains necessary keys
        assert 'obs_vocab' in config
        assert 'action_vocab' in config
        assert 'obs_dim' in config
        assert 'action_dim' in config

        # Recreate encoder from config
        encoder2 = ObservationActionEncoder.from_config(
            config,
            observations,
            actions
        )

        # Should have same dimensions
        assert encoder2.obs_dim == encoder1.obs_dim
        assert encoder2.action_dim == encoder1.action_dim

        # Should encode the same way
        history = [((0, 0), -1), ((1, 1), 0)]
        vec1 = encoder1.encode_history(history, 5)
        vec2 = encoder2.encode_history(history, 5)
        assert np.allclose(vec1, vec2)


# ============================================================
# Network Tests
# ============================================================

class TestQNetwork:
    """Tests for QNetwork."""

    def test_forward_pass(self):
        """Test forward pass dimensions."""
        input_dim = 50
        num_actions = 3

        network = QNetwork(input_dim, num_actions)

        # Test with batch
        batch_size = 32
        x = torch.randn(batch_size, input_dim)
        q_values = network(x)

        assert q_values.shape == (batch_size, num_actions)

    def test_single_input(self):
        """Test with single input."""
        network = QNetwork(input_dim=10, num_actions=3)

        x = torch.randn(1, 10)
        q_values = network(x)

        assert q_values.shape == (1, 3)

    def test_custom_architecture(self):
        """Test with custom hidden dimensions."""
        network = QNetwork(
            input_dim=20,
            num_actions=5,
            hidden_dims=(64, 32),
            dropout=0.2
        )

        x = torch.randn(10, 20)
        q_values = network(x)

        assert q_values.shape == (10, 5)


# ============================================================
# Replay Buffer Tests
# ============================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_push_and_sample(self):
        """Test storing and sampling transitions."""
        buffer = ReplayBuffer(capacity=100)

        # Add transitions
        for i in range(50):
            buffer.push(
                history=tuple(),
                action=i % 3,
                reward=1.0,
                next_history=tuple(),
                done=False
            )

        assert len(buffer) == 50

        # Sample batch
        batch = buffer.sample(10)
        assert len(batch) == 10
        assert all(isinstance(t, Transition) for t in batch)

    def test_capacity_limit(self):
        """Test that buffer respects capacity."""
        buffer = ReplayBuffer(capacity=10)

        # Add more than capacity
        for i in range(20):
            buffer.push(
                history=tuple(),
                action=i,
                reward=1.0,
                next_history=tuple(),
                done=False
            )

        # Should only keep last 10
        assert len(buffer) == 10

    def test_transition_fields(self):
        """Test that transitions have correct fields."""
        buffer = ReplayBuffer(capacity=10)

        history = (((0, 0), -1),)
        next_history = (((0, 0), -1), ((1, 1), 0))

        buffer.push(
            history=history,
            action=0,
            reward=0.5,
            next_history=next_history,
            done=False
        )

        batch = buffer.sample(1)
        t = batch[0]

        assert t.history == history
        assert t.action == 0
        assert t.reward == 0.5
        assert t.next_history == next_history
        assert t.done == False


# ============================================================
# Neural Selector Tests
# ============================================================

class TestNeuralActionSelector:
    """Tests for NeuralActionSelector."""

    def test_initialization(self):
        """Test selector initialization."""
        observations = [(0, 0), (1, 1)]
        actions = [-1, 0, 1]

        selector = NeuralActionSelector(
            actions=actions,
            observations=observations,
            history_window=5
        )

        assert selector.history_window == 5
        assert len(selector.actions) == 3
        assert selector.q_network is not None
        assert selector.target_network is not None
        assert selector.optimizer is not None

    def test_action_selection_random(self):
        """Test random action selection (high exploration)."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        selector = NeuralActionSelector(
            actions=actions,
            observations=observations,
            exploration_rate=1.0  # Always explore
        )

        history = [((0, 0), -1)]
        action = selector.select(history, actions)

        assert action in actions

    def test_action_selection_greedy(self):
        """Test greedy action selection (no exploration)."""
        observations = [(0, 0)]
        actions = [-1, 0, 1]

        selector = NeuralActionSelector(
            actions=actions,
            observations=observations,
            exploration_rate=0.0  # Never explore
        )

        history = [((0, 0), -1)]
        action = selector.select(history, actions)

        assert action in actions

    def test_reward_computation(self):
        """Test reward calculation."""
        selector = NeuralActionSelector(
            actions=[-1, 0, 1],
            observations=[(0, 0)],
            maximize_safety=True
        )

        # Safe agent rewards
        assert selector._compute_reward("fail") < 0
        assert selector._compute_reward("safe") > 0
        assert selector._compute_reward("step") > 0

        # Adversarial agent rewards
        selector.maximize_safety = False
        assert selector._compute_reward("fail") > 0
        assert selector._compute_reward("safe") < 0
        assert selector._compute_reward("step") < 0

    def test_exploration_decay(self):
        """Test exploration rate decay."""
        selector = NeuralActionSelector(
            actions=[-1, 0, 1],
            observations=[(0, 0)],
            exploration_rate=1.0,
            exploration_decay=0.9,
            min_exploration=0.01
        )

        initial_rate = selector.exploration_rate

        # Decay
        selector._decay_exploration()
        assert selector.exploration_rate == initial_rate * 0.9

        # Multiple decays
        for _ in range(100):
            selector._decay_exploration()

        # Should not go below minimum
        assert selector.exploration_rate >= 0.01

    def test_save_and_load(self):
        """Test model persistence."""
        observations = [(0, 0), (1, 1)]
        actions = [-1, 0, 1]

        # Create and configure selector
        selector1 = NeuralActionSelector(
            actions=actions,
            observations=observations,
            history_window=5,
            maximize_safety=True
        )

        # Train for a bit to create non-random weights
        selector1.replay_buffer.push(
            history=tuple(),
            action=0,
            reward=1.0,
            next_history=(((0, 0), -1),),
            done=False
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            selector1.save(filepath)

            # Create mock IPOMDP for loading
            class MockIPOMDP:
                def __init__(self):
                    self.actions = actions
                    self.observations = observations

            ipomdp = MockIPOMDP()

            # Load
            selector2 = NeuralActionSelector.load(filepath, ipomdp)

            # Check hyperparameters preserved
            assert selector2.history_window == 5
            assert selector2.maximize_safety == True
            assert len(selector2.actions) == 3

            # Check networks have same weights
            for p1, p2 in zip(
                selector1.q_network.parameters(),
                selector2.q_network.parameters()
            ):
                assert torch.allclose(p1, p2)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests with simple environments."""

    def test_train_simple_environment(self):
        """Test training on a simple deterministic environment."""
        # Create simple IPOMDP
        class SimpleIPOMDP:
            def __init__(self):
                self.states = ["s0", "s1", "FAIL"]
                self.observations = ["obs0", "obs1"]
                self.actions = ["stay", "move"]

                # Deterministic dynamics
                self.T = {
                    ("s0", "stay"): {"s0": 1.0},
                    ("s0", "move"): {"s1": 1.0},
                    ("s1", "stay"): {"s1": 1.0},
                    ("s1", "move"): {"FAIL": 1.0},
                }

                # Observation model
                self.P_lower = {
                    "s0": {"obs0": 0.9, "obs1": 0.0},
                    "s1": {"obs0": 0.0, "obs1": 0.9},
                    "FAIL": {"obs0": 0.0, "obs1": 0.0}
                }
                self.P_upper = {
                    "s0": {"obs0": 1.0, "obs1": 0.1},
                    "s1": {"obs0": 0.1, "obs1": 1.0},
                    "FAIL": {"obs0": 0.0, "obs1": 0.0}
                }

            def evolve(self, state, action):
                if state == "FAIL":
                    return "FAIL"
                transitions = self.T.get((state, action), {})
                if not transitions:
                    return state
                states = list(transitions.keys())
                probs = list(transitions.values())
                import random
                return random.choices(states, weights=probs)[0]

        # Simple perception model
        class SimplePerception:
            def sample_observation(self, state, ipomdp, context=None):
                if state == "FAIL":
                    return "obs0"
                elif state == "s0":
                    return "obs0"
                else:
                    return "obs1"

        ipomdp = SimpleIPOMDP()
        perception = SimplePerception()

        selector = NeuralActionSelector(
            actions=ipomdp.actions,
            observations=ipomdp.observations,
            history_window=3,
            maximize_safety=True,
            batch_size=16
        )

        # Train for a few episodes
        metrics = selector.train(
            ipomdp=ipomdp,
            perception=perception,
            num_episodes=10,
            episode_length=5,
            verbose=False
        )

        # Check metrics structure
        assert 'episode_rewards' in metrics
        assert 'episode_outcomes' in metrics
        assert 'final_safe_rate' in metrics
        assert len(metrics['episode_rewards']) == 10
        assert len(metrics['episode_outcomes']) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
