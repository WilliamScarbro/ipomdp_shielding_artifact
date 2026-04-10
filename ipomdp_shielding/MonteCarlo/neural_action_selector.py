"""Neural Network RL Action Selector for IPOMDP Models.

This module implements a Deep Q-Network (DQN) based action selector that:
- Trains directly on IPOMDP dynamics (no runtime shield dependency)
- Uses observation-action history as input (not belief states)
- Optimizes for FAIL avoidance/seeking (binary safety objective)
- Supports model persistence (save/load)

Key components:
- ObservationActionEncoder: Encodes heterogeneous obs/action types to vectors
- QNetwork: Deep Q-Network for action value estimation
- ReplayBuffer: Experience replay for stable learning
- NeuralActionSelector: Main DQN-based selector with training loop
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import deque, namedtuple
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Models.ipomdp import IPOMDP
from .action_selectors import RLActionSelector


# ============================================================
# Data Structures
# ============================================================

Transition = namedtuple('Transition',
    ['history', 'action', 'reward', 'next_history', 'done'])


# ============================================================
# Observation and Action Encoding
# ============================================================

class ObservationActionEncoder:
    """Encodes heterogeneous observation/action types to fixed-size vectors.

    Handles:
    - Tuple observations: (cte, he) → [cte, he]
    - Scalar observations/actions: value → [value]
    - Discrete observations/actions: one-hot encoding

    Examples:
        TaxiNet observations: (0, 0) → np.array([0.0, 0.0])
        TaxiNet actions: {-1, 0, 1} → one-hot vectors
    """

    def __init__(
        self,
        observations: List[Any],
        actions: List[Any],
        obs_vocab: Optional[Dict[Any, int]] = None,
        action_vocab: Optional[Dict[Any, int]] = None
    ):
        """Initialize encoder from observation and action samples.

        Parameters
        ----------
        observations : list
            Sample observations (used to detect encoding mode)
        actions : list
            All possible actions
        obs_vocab : dict, optional
            Pre-built observation vocabulary for loading
        action_vocab : dict, optional
            Pre-built action vocabulary for loading
        """
        self.observations = observations
        self.actions = actions

        # Detect encoding modes
        if obs_vocab is not None:
            self.obs_vocab = obs_vocab
            self.obs_mode = self._detect_mode_from_vocab(obs_vocab)
        else:
            self.obs_mode = self._detect_encoding_mode_from_list(observations)
            self.obs_vocab = self._build_vocab(observations, self.obs_mode)

        if action_vocab is not None:
            self.action_vocab = action_vocab
            self.action_mode = self._detect_mode_from_vocab(action_vocab)
        else:
            self.action_mode = self._detect_encoding_mode_from_list(actions)
            self.action_vocab = self._build_vocab(actions, self.action_mode)

        # Determine dimensions
        self.obs_dim = self._get_dim(observations[0] if observations else None,
                                     self.obs_mode, self.obs_vocab)
        self.action_dim = self._get_dim(actions[0] if actions else None,
                                       self.action_mode, self.action_vocab)

    def _detect_encoding_mode_from_list(self, items: List[Any]) -> str:
        """Detect encoding mode from list of items.

        Checks if items are tuples, discrete (small set), or scalar.
        """
        if not items:
            return "scalar"

        sample = items[0]

        # Check if tuple
        if isinstance(sample, tuple):
            return "tuple"

        # Check if discrete (small number of unique values)
        unique_items = set(items)
        if len(unique_items) <= 20:  # Treat as discrete if <= 20 unique values
            return "discrete"

        # Otherwise treat as scalar
        return "scalar"

    def _detect_mode_from_vocab(self, vocab: Dict[Any, int]) -> str:
        """Detect encoding mode from vocabulary."""
        if not vocab:
            return "scalar"
        # Check first key to determine mode
        sample = next(iter(vocab.keys()))
        if isinstance(sample, tuple):
            return "tuple"
        # If we have a vocab, it means it was discrete
        return "discrete"

    def _build_vocab(self, items: List[Any], mode: str) -> Dict[Any, int]:
        """Build vocabulary for discrete items."""
        if mode == "discrete":
            return {item: idx for idx, item in enumerate(sorted(set(items)))}
        return {}

    def _get_dim(self, sample: Any, mode: str, vocab: Dict[Any, int]) -> int:
        """Get dimension for encoding."""
        if mode == "tuple":
            return len(sample) if sample is not None else 0
        elif mode == "discrete":
            return len(vocab) if vocab else 1
        else:  # scalar
            return 1

    def encode_observation(self, obs: Any) -> np.ndarray:
        """Encode single observation to vector.

        Parameters
        ----------
        obs : any
            Observation to encode

        Returns
        -------
        np.ndarray
            Encoded observation vector
        """
        if self.obs_mode == "tuple":
            if not isinstance(obs, tuple):
                return np.zeros(self.obs_dim, dtype=np.float32)
            return np.array(obs, dtype=np.float32)
        elif self.obs_mode == "discrete":
            vec = np.zeros(self.obs_dim, dtype=np.float32)
            if obs in self.obs_vocab:
                vec[self.obs_vocab[obs]] = 1.0
            return vec
        else:  # scalar
            return np.array([float(obs)], dtype=np.float32)

    def encode_action(self, action: Any) -> np.ndarray:
        """Encode single action to vector.

        Parameters
        ----------
        action : any
            Action to encode

        Returns
        -------
        np.ndarray
            Encoded action vector
        """
        if self.action_mode == "discrete":
            vec = np.zeros(self.action_dim, dtype=np.float32)
            if action in self.action_vocab:
                vec[self.action_vocab[action]] = 1.0
            return vec
        else:  # scalar
            return np.array([float(action)], dtype=np.float32)

    def encode_history(
        self,
        history: List[Tuple[Any, Any]],
        window_size: int
    ) -> np.ndarray:
        """Encode observation-action history to fixed-size vector.

        Takes last window_size (obs, action) pairs, encodes each,
        and concatenates. Zero-pads if history is shorter.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Observation-action history
        window_size : int
            Number of recent pairs to use

        Returns
        -------
        np.ndarray
            Encoded history vector of shape (window_size * (obs_dim + action_dim),)
        """
        # Take last window_size pairs
        recent = history[-window_size:] if len(history) > 0 else []

        # Encode each pair
        encoded_pairs = []
        for obs, action in recent:
            obs_vec = self.encode_observation(obs)
            action_vec = self.encode_action(action)
            pair_vec = np.concatenate([obs_vec, action_vec])
            encoded_pairs.append(pair_vec)

        # Zero-pad if needed
        pair_dim = self.obs_dim + self.action_dim
        while len(encoded_pairs) < window_size:
            encoded_pairs.insert(0, np.zeros(pair_dim, dtype=np.float32))

        # Concatenate to single vector
        return np.concatenate(encoded_pairs)

    def get_input_dim(self, window_size: int) -> int:
        """Get total input dimension for network.

        Parameters
        ----------
        window_size : int
            History window size

        Returns
        -------
        int
            Total dimension: window_size * (obs_dim + action_dim)
        """
        return window_size * (self.obs_dim + self.action_dim)

    def get_config(self) -> Dict[str, Any]:
        """Serialize encoder configuration for saving.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            'obs_vocab': self.obs_vocab,
            'action_vocab': self.action_vocab,
            'obs_mode': self.obs_mode,
            'action_mode': self.action_mode,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        observations: List[Any],
        actions: List[Any]
    ) -> 'ObservationActionEncoder':
        """Deserialize encoder from configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        observations : list
            Sample observations
        actions : list
            All possible actions

        Returns
        -------
        ObservationActionEncoder
            Reconstructed encoder
        """
        encoder = cls(
            observations=observations,
            actions=actions,
            obs_vocab=config['obs_vocab'],
            action_vocab=config['action_vocab']
        )

        # Override dimensions from config
        encoder.obs_dim = config['obs_dim']
        encoder.action_dim = config['action_dim']
        encoder.obs_mode = config['obs_mode']
        encoder.action_mode = config['action_mode']

        return encoder


# ============================================================
# Neural Network Architectures
# ============================================================

class QNetwork(nn.Module):
    """Deep Q-Network for action value estimation.

    Architecture:
        Input → Dense(128) + ReLU + Dropout
              → Dense(64) + ReLU + Dropout
              → Dense(32) + ReLU
              → Output(|Actions|)
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1
    ):
        """Initialize Q-Network.

        Parameters
        ----------
        input_dim : int
            Input dimension (encoded history size)
        num_actions : int
            Number of actions (output size)
        hidden_dims : tuple of int
            Hidden layer dimensions
        dropout : float
            Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Add dropout except for last hidden layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Q-values of shape (batch_size, num_actions)
        """
        return self.network(x)


# ============================================================
# Experience Replay
# ============================================================

class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Stores transitions (history, action, reward, next_history, done)
    and supports random sampling for training.
    """

    def __init__(self, capacity: int = 50000):
        """Initialize replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        history: Tuple,
        action: Any,
        reward: float,
        next_history: Tuple,
        done: bool
    ):
        """Store a transition.

        Parameters
        ----------
        history : tuple
            Previous observation-action history (as tuple for hashability)
        action : any
            Action taken
        reward : float
            Reward received
        next_history : tuple
            Updated history after action
        done : bool
            Whether episode terminated
        """
        self.buffer.append(Transition(history, action, reward, next_history, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        list of Transition
            Random batch
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


# ============================================================
# Neural Action Selector
# ============================================================

class NeuralActionSelector(RLActionSelector):
    """Deep Q-Network based action selector.

    Trains directly on IPOMDP dynamics without runtime shield dependency.
    Uses observation-action history as state representation.
    Optimizes for binary FAIL vs safe objective.
    """

    def __init__(
        self,
        actions: List[Any],
        observations: List[Any],
        history_window: int = 10,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.01,
        maximize_safety: bool = True,
        replay_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        architecture: str = 'feedforward',
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """Initialize neural action selector.

        Parameters
        ----------
        actions : list
            All possible actions
        observations : list
            Sample observations (for encoder initialization)
        history_window : int
            Number of recent (obs, action) pairs to use as state
        learning_rate : float
            Adam optimizer learning rate
        discount_factor : float
            Future reward discount (gamma)
        exploration_rate : float
            Initial epsilon for epsilon-greedy exploration
        exploration_decay : float
            Decay factor for epsilon per episode
        min_exploration : float
            Minimum epsilon value
        maximize_safety : bool
            If True, train to avoid FAIL (safe agent)
            If False, train to reach FAIL (adversarial agent)
        replay_capacity : int
            Experience replay buffer capacity
        batch_size : int
            Training batch size
        target_update_freq : int
            Steps between target network updates
        architecture : str
            Network architecture ('feedforward' or 'gru')
        hidden_dims : tuple of int
            Hidden layer dimensions
        dropout : float
            Dropout rate for regularization
        device : str
            Device for training ('cpu' or 'cuda')
        """
        self.actions = actions
        self.observations = observations
        self.history_window = history_window
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration = exploration_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.maximize_safety = maximize_safety
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.architecture = architecture
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = torch.device(device)

        # Initialize encoder
        self.encoder = ObservationActionEncoder(observations, actions)

        # Create networks
        input_dim = self.encoder.get_input_dim(history_window)
        num_actions = len(actions)

        self.q_network = QNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        self.target_network = QNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(replay_capacity)

        # Training state
        self.train_steps = 0
        self.episodes_trained = 0
        self.final_safe_rate = 0.0

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using epsilon-greedy policy.

        NOTE: allowed_actions is IGNORED during training (no shield dependency).
        During evaluation, it can be used to filter actions.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Observation-action history
        allowed_actions : list
            Actions permitted by shield (ignored during training)
        context : dict, optional
            Additional context (unused)

        Returns
        -------
        action
            Selected action
        """
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Encode history
        history_vec = self._encode_history(history)

        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(history_vec.unsqueeze(0)).squeeze(0)
        self.q_network.train()

        # Select action based on objective
        if self.maximize_safety:
            # Maximize Q-value (higher = safer)
            action_idx = q_values.argmax().item()
        else:
            # Minimize Q-value (lower = more likely to fail)
            action_idx = q_values.argmin().item()

        return self.actions[action_idx]

    def train(
        self,
        ipomdp: IPOMDP,
        perception: "PerceptionModel",
        num_episodes: int,
        episode_length: int,
        initial_generator: Optional["InitialStateGenerator"] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train DQN policy through simulation.

        NO SHIELD DEPENDENCY - trains directly on IPOMDP dynamics.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        perception : PerceptionModel
            Perception model for sampling observations
        num_episodes : int
            Number of training episodes
        episode_length : int
            Maximum steps per episode
        initial_generator : InitialStateGenerator, optional
            Strategy for initial states (unused - uses random)
        verbose : bool
            Print training progress

        Returns
        -------
        dict
            Training metrics including episode rewards and safety rates
        """
        # Import here to avoid circular imports
        from .initial_states import RandomInitialState

        if initial_generator is None:
            initial_generator = RandomInitialState()

        # Reset exploration rate for training
        self.exploration_rate = self.initial_exploration

        episode_rewards = []
        episode_outcomes = []

        for episode in range(num_episodes):
            # Generate initial state (simple random - no pp_shield needed)
            safe_states = [s for s in ipomdp.states if s != "FAIL"]
            state = random.choice(safe_states)
            action = random.choice(self.actions)

            self.reset()
            history = []
            total_reward = 0.0
            outcome = "safe"

            for step in range(episode_length):
                # Check for failure
                if state == "FAIL":
                    reward = self._compute_reward("fail")
                    self._store_transition(history, action, reward, history, done=True)
                    total_reward += reward
                    outcome = "fail"
                    break

                # Get observation (no shield in context)
                obs = perception.sample_observation(state, ipomdp, context={})
                history.append((obs, action))

                # Step reward
                step_reward = self._compute_reward("step")

                # Select next action (ignore shield constraints)
                next_action = self.select(history, self.actions)

                # Store transition
                prev_history = history[:-1] if len(history) > 1 else []
                self._store_transition(
                    prev_history=prev_history,
                    action=action,
                    reward=step_reward,
                    next_history=history,
                    done=False
                )
                total_reward += step_reward

                # Update action for next iteration
                action = next_action

                # Evolve state (direct IPOMDP dynamics, no shield)
                state = ipomdp.evolve(state, action)

                # Train network
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self._train_step()

            # Episode completion
            if outcome == "safe":
                reward = self._compute_reward("safe")
                self._store_transition(history, action, reward, history, done=True)
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_outcomes.append(outcome)
            self._decay_exploration()

            # Logging
            if verbose and (episode + 1) % 50 == 0:
                recent_outcomes = episode_outcomes[-50:]
                safe_rate = sum(1 for o in recent_outcomes if o == "safe") / len(recent_outcomes)
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Ep {episode+1:4d}: safe={safe_rate:.2%}, "
                      f"ε={self.exploration_rate:.3f}, "
                      f"reward={avg_reward:.2f}")

        # Update training metadata
        self.episodes_trained = num_episodes
        self.final_safe_rate = sum(1 for o in episode_outcomes if o == "safe") / num_episodes

        # Compute metrics
        safe_count = sum(1 for o in episode_outcomes if o == "safe")
        fail_count = sum(1 for o in episode_outcomes if o == "fail")

        return {
            'episode_rewards': episode_rewards,
            'episode_outcomes': episode_outcomes,
            'final_safe_rate': safe_count / num_episodes,
            'final_fail_rate': fail_count / num_episodes,
        }

    def _train_step(self) -> float:
        """Perform single DQN training step.

        Returns
        -------
        float
            Training loss
        """
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare tensors
        history_vecs = torch.stack([
            self._encode_history(list(t.history))
            for t in batch
        ]).to(self.device)

        action_indices = torch.tensor([
            self.actions.index(t.action)
            for t in batch
        ], dtype=torch.long).to(self.device)

        rewards = torch.tensor([
            t.reward
            for t in batch
        ], dtype=torch.float32).to(self.device)

        next_history_vecs = torch.stack([
            self._encode_history(list(t.next_history))
            for t in batch
        ]).to(self.device)

        dones = torch.tensor([
            float(t.done)
            for t in batch
        ], dtype=torch.float32).to(self.device)

        # Compute current Q-values
        current_q = self.q_network(history_vecs).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_history_vecs).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def _encode_history(self, history: List[Tuple]) -> torch.Tensor:
        """Encode history to tensor.

        Parameters
        ----------
        history : list of (observation, action) pairs
            History to encode

        Returns
        -------
        torch.Tensor
            Encoded history tensor
        """
        encoded = self.encoder.encode_history(history, self.history_window)
        return torch.tensor(encoded, dtype=torch.float32, device=self.device)

    def _compute_reward(self, outcome: str) -> float:
        """Compute reward based on outcome.

        Parameters
        ----------
        outcome : str
            Outcome type: "fail", "safe", or "step"

        Returns
        -------
        float
            Reward value
        """
        if outcome == "fail":
            return -10.0 if self.maximize_safety else +10.0
        elif outcome == "safe":
            return +10.0 if self.maximize_safety else -10.0
        elif outcome == "step":
            return +0.1 if self.maximize_safety else -0.1
        return 0.0

    def _store_transition(
        self,
        prev_history: List[Tuple],
        action: Any,
        reward: float,
        next_history: List[Tuple],
        done: bool
    ):
        """Store transition in replay buffer.

        Parameters
        ----------
        prev_history : list
            Previous history
        action : any
            Action taken
        reward : float
            Reward received
        next_history : list
            Updated history
        done : bool
            Whether episode terminated
        """
        # Convert to tuples to avoid mutation
        self.replay_buffer.push(
            tuple(prev_history),
            action,
            reward,
            tuple(next_history),
            done
        )

    def _decay_exploration(self):
        """Decay exploration rate after episode."""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def reset(self):
        """Reset episode-specific state."""
        # No episode-specific state for DQN
        pass

    def save(self, filepath: str):
        """Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to save file (.pt extension recommended)
        """
        torch.save({
            'network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'history_window': self.history_window,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'maximize_safety': self.maximize_safety,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'architecture': self.architecture,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'exploration_decay': self.exploration_decay,
                'min_exploration': self.min_exploration,
                'replay_capacity': self.replay_capacity,
            },
            'encoder_config': self.encoder.get_config(),
            'training_metadata': {
                'episodes_trained': self.episodes_trained,
                'final_safe_rate': self.final_safe_rate,
                'timestamp': datetime.now().isoformat(),
            }
        }, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        ipomdp: IPOMDP,
        device: str = 'cpu'
    ) -> 'NeuralActionSelector':
        """Load pre-trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model file
        ipomdp : IPOMDP
            IPOMDP model (for actions and observations)
        device : str
            Device for loading ('cpu' or 'cuda')

        Returns
        -------
        NeuralActionSelector
            Loaded model in evaluation mode
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Reconstruct selector
        selector = cls(
            actions=list(ipomdp.actions),
            observations=ipomdp.observations,
            device=device,
            **checkpoint['hyperparameters']
        )

        # Load encoder config
        selector.encoder = ObservationActionEncoder.from_config(
            checkpoint['encoder_config'],
            ipomdp.observations,
            list(ipomdp.actions)
        )

        # Recreate networks with correct dimensions
        input_dim = selector.encoder.get_input_dim(selector.history_window)
        num_actions = len(selector.actions)

        selector.q_network = QNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=selector.hidden_dims,
            dropout=selector.dropout
        ).to(selector.device)

        selector.target_network = QNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=selector.hidden_dims,
            dropout=selector.dropout
        ).to(selector.device)

        # Load weights
        selector.q_network.load_state_dict(checkpoint['network_state_dict'])
        selector.target_network.load_state_dict(checkpoint['target_network_state_dict'])

        # Recreate optimizer
        selector.optimizer = torch.optim.Adam(
            selector.q_network.parameters(),
            lr=selector.learning_rate
        )
        selector.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set to evaluation mode
        selector.exploration_rate = 0.0
        selector.q_network.eval()
        selector.target_network.eval()

        # Load metadata
        selector.episodes_trained = checkpoint['training_metadata']['episodes_trained']
        selector.final_safe_rate = checkpoint['training_metadata']['final_safe_rate']

        return selector
