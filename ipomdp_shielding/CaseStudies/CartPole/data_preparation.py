"""Data preparation pipeline for CartPole case study.

This module bridges between the existing cartpole training code and the
ipomdp_shielding framework structure. It generates:
1. Trained CartPoleStateNet model
2. Confusion matrices for each state dimension
3. Bin edges for state discretization
4. Empirical dynamics MDP from gymnasium rollouts

Supports configurable discretization with different bin counts per dimension.
"""

import sys
from pathlib import Path
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

# Discretization configuration type
# Can be either:
# - int: uniform bins across all dimensions
# - List[int]: per-dimension bin counts [n_x, n_xdot, n_theta, n_thetadot]
DiscretizationConfig = Union[int, List[int]]

# Add training directory to path to import existing modules
TRAINING_DIR = Path(__file__).parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from model import (
    collect_cartpole_data,
    make_dataloaders,
    CartPoleStateNet,
    train_state_net,
    collect_predictions,
    make_confusion_matrices,
)

# Import framework components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Models.mdp import MDP


def _parse_discretization_config(num_bins: DiscretizationConfig) -> List[int]:
    """Parse discretization config into per-dimension bin counts.

    Args:
        num_bins: Either int (uniform) or List[int] (per-dimension)

    Returns:
        List of 4 integers: [n_x, n_xdot, n_theta, n_thetadot]
    """
    if isinstance(num_bins, int):
        return [num_bins] * 4
    else:
        if len(num_bins) != 4:
            raise ValueError(f"num_bins list must have 4 elements, got {len(num_bins)}")
        return list(num_bins)


def make_confusion_matrices_configurable(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins_per_dim: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build confusion matrices with configurable bins per dimension.

    Args:
        y_true: True states array of shape (N, 4)
        y_pred: Predicted states array of shape (N, 4)
        bins_per_dim: Number of bins for each dimension [n_x, n_xdot, n_theta, n_thetadot]

    Returns:
        Tuple of (cms, edges_list) where:
        - cms: list of 4 confusion matrices, each of shape (bins[i], bins[i])
        - edges_list: list of 4 bin edge arrays, each of length (bins[i] + 1)
    """
    assert y_true.shape == y_pred.shape
    N, dims = y_true.shape
    assert dims == 4
    assert len(bins_per_dim) == 4

    cms = []
    edges_list = []

    for d in range(dims):
        k = bins_per_dim[d]

        # Compute bin edges using combined range (true + predicted)
        v_min = float(min(y_true[:, d].min(), y_pred[:, d].min()))
        v_max = float(max(y_true[:, d].max(), y_pred[:, d].max()))

        # Add small epsilon to avoid edge cases
        eps = (v_max - v_min) * 1e-6 + 1e-9
        edges = np.linspace(v_min - eps, v_max + eps, k + 1)

        # Discretize true and predicted values
        true_bins = np.digitize(y_true[:, d], edges) - 1
        pred_bins = np.digitize(y_pred[:, d], edges) - 1

        # Clip to valid range [0, k-1]
        true_bins = np.clip(true_bins, 0, k - 1)
        pred_bins = np.clip(pred_bins, 0, k - 1)

        # Build confusion matrix
        cm = np.zeros((k, k), dtype=int)
        for i in range(N):
            cm[true_bins[i], pred_bins[i]] += 1

        cms.append(cm)
        edges_list.append(edges)

    return cms, edges_list


def prepare_perception_data(
    num_episodes: int = 200,
    epochs: int = 20,
    num_bins: DiscretizationConfig = 7,
    seed: int = 0,
    device: str = "cuda",
    data_dir: Optional[Path] = None,
):
    """Generate and save perception model data.

    Steps:
    1. Collect frame pairs and states from CartPole-v1
    2. Train CartPoleStateNet on collected data
    3. Generate confusion matrices for each dimension
    4. Save confusion matrices, bin edges, and trained model

    Args:
        num_episodes: Number of episodes to collect for training
        epochs: Number of training epochs
        num_bins: Number of bins per dimension. Can be:
            - int: Same number of bins for all dimensions (e.g., 7)
            - List[int]: Per-dimension bins [n_x, n_xdot, n_theta, n_thetadot] (e.g., [5, 4, 5, 4])
        seed: Random seed for reproducibility
        device: Device for training ("cuda" or "cpu")
        data_dir: Directory to save data. If None, uses artifacts/ subdirectory
    """
    bins = _parse_discretization_config(num_bins)
    if data_dir is None:
        data_dir = Path(__file__).parent / "artifacts"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Collecting CartPole data...")
    print("=" * 60)
    frame_pairs, states = collect_cartpole_data(
        num_episodes=num_episodes,
        seed=seed
    )
    print(f"Collected {len(states)} state observations")

    print("\n" + "=" * 60)
    print("STEP 2: Creating data loaders...")
    print("=" * 60)
    train_loader, val_loader, test_loader = make_dataloaders(
        frame_pairs, states, batch_size=64
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    print("\n" + "=" * 60)
    print("STEP 3: Training CartPoleStateNet...")
    print("=" * 60)
    model = train_state_net(
        train_loader,
        val_loader,
        epochs=epochs,
        device=device
    )

    print("\n" + "=" * 60)
    print("STEP 4: Generating confusion matrices...")
    print("=" * 60)
    y_true, y_pred = collect_predictions(model, test_loader, device=device)
    cms, edges_list = make_confusion_matrices_configurable(y_true, y_pred, bins)

    print("\n" + "=" * 60)
    print("STEP 5: Saving data to artifacts/...")
    print("=" * 60)

    # Save bin edges
    edges_array = np.array(edges_list, dtype=object)
    np.save(data_dir / "bin_edges.npy", edges_array, allow_pickle=True)
    print(f"✓ Saved bin_edges.npy (bins per dimension: {bins})")

    # Save confusion matrices for each dimension
    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    for i, dim in enumerate(dim_names):
        np.save(data_dir / f"{dim}_confusion.npy", cms[i])
        print(f"✓ Saved {dim}_confusion.npy (shape: {cms[i].shape}, {bins[i]} bins)")

    # Save trained model
    torch.save(model.state_dict(), data_dir / "cartpole_state_net.pt")
    print(f"✓ Saved cartpole_state_net.pt")

    print("\n" + "=" * 60)
    print("Perception data preparation complete!")
    print(f"Discretization: {bins[0]}×{bins[1]}×{bins[2]}×{bins[3]} = {np.prod(bins)} total states")
    print("=" * 60)


def discretize_state(state: np.ndarray, bin_edges: np.ndarray) -> Tuple[int, int, int, int]:
    """Discretize a continuous state into bin indices.

    Args:
        state: Continuous state [x, x_dot, theta, theta_dot]
        bin_edges: Array of shape (4, k+1) containing bin edges for each dimension

    Returns:
        Tuple of 4 bin indices
    """
    bins = []
    for d in range(4):
        edges = bin_edges[d]
        k = len(edges) - 1
        bin_idx = np.clip(np.digitize(state[d], edges) - 1, 0, k - 1)
        bins.append(int(bin_idx))
    return tuple(bins)


def prepare_dynamics_data(
    num_episodes: int = 10000,
    max_steps: int = 200,
    num_bins: Optional[DiscretizationConfig] = None,
    seed: int = 0,
    data_dir: Optional[Path] = None,
):
    """Generate and save empirical dynamics MDP from gymnasium rollouts.

    Collects transition counts by running random policy in CartPole-v1,
    then normalizes to create an MDP.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        num_bins: Number of bins per dimension. If None, infers from loaded bin_edges.
            Can be:
            - int: Same number of bins for all dimensions (e.g., 7)
            - List[int]: Per-dimension bins [n_x, n_xdot, n_theta, n_thetadot] (e.g., [5, 4, 5, 4])
        seed: Random seed for reproducibility
        data_dir: Directory to save data. If None, uses artifacts/ subdirectory
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "artifacts"
    data_dir.mkdir(exist_ok=True)

    # Load bin edges
    bin_edges = np.load(data_dir / "bin_edges.npy", allow_pickle=True)

    # Infer bins from bin_edges if not provided
    if num_bins is None:
        bins = [len(edges) - 1 for edges in bin_edges]
    else:
        bins = _parse_discretization_config(num_bins)

    print("=" * 60)
    print("Collecting dynamics data from CartPole-v1...")
    print(f"Discretization: {bins[0]}×{bins[1]}×{bins[2]}×{bins[3]} = {np.prod(bins)} states")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)

    # Track transition counts
    transition_counts = defaultdict(lambda: defaultdict(int))
    FAIL = "FAIL"

    for ep in trange(num_episodes, desc="Episodes"):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = np.array(env.unwrapped.state, dtype=np.float32)
        s_discrete = discretize_state(state, bin_edges)

        for t in range(max_steps):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(env.unwrapped.state, dtype=np.float32)
            s_next_discrete = discretize_state(next_state, bin_edges)

            # If episode terminated, transition to FAIL state
            if terminated:
                s_next_discrete = FAIL

            transition_counts[(s_discrete, action)][s_next_discrete] += 1

            if terminated or truncated:
                break

            s_discrete = s_next_discrete
            state = next_state

    env.close()

    print("\n" + "=" * 60)
    print("Building MDP from transition counts...")
    print("=" * 60)

    # Build state space with configurable bins per dimension
    states = [
        (x_bin, xdot_bin, theta_bin, thetadot_bin)
        for x_bin in range(bins[0])
        for xdot_bin in range(bins[1])
        for theta_bin in range(bins[2])
        for thetadot_bin in range(bins[3])
    ]
    states.append(FAIL)

    # Build action dictionary
    actions_dict = {s: [0, 1] for s in states if s != FAIL}
    actions_dict[FAIL] = []  # No actions from FAIL state

    # Build transition probabilities
    P = {}

    # Add FAIL self-loop (absorbing state)
    # Note: FAIL has no actions, so no transitions needed in P

    # Normalize transition counts to probabilities
    for (s, a), next_counts in transition_counts.items():
        total = sum(next_counts.values())
        if total > 0:
            P[(s, a)] = {s_next: count / total for s_next, count in next_counts.items()}
        else:
            # Should not happen, but handle gracefully
            P[(s, a)] = {}

    # Fill in missing transitions with uniform distribution over visited states
    # (or could use single self-loop for unobserved state-action pairs)
    num_state_actions = 0
    num_observed = len(P)

    for s in states:
        if s == FAIL:
            continue
        for a in [0, 1]:
            num_state_actions += 1
            if (s, a) not in P:
                # Unobserved state-action: add self-loop (conservative)
                P[(s, a)] = {s: 1.0}

    mdp = MDP(states, actions_dict, P)

    print(f"States: {len(states)}")
    print(f"State-action pairs: {num_state_actions}")
    print(f"Observed transitions: {num_observed} ({100 * num_observed / num_state_actions:.1f}%)")
    print(f"Unobserved transitions: {num_state_actions - num_observed} (filled with self-loops)")

    print("\n" + "=" * 60)
    print("Saving dynamics MDP...")
    print("=" * 60)

    with open(data_dir / "dynamics_mdp.pkl", "wb") as f:
        pickle.dump(mdp, f)
    print(f"✓ Saved dynamics_mdp.pkl")

    print("\n" + "=" * 60)
    print("Dynamics data preparation complete!")
    print("=" * 60)


def prepare_all_data(
    perception_episodes: int = 200,
    dynamics_episodes: int = 10000,
    num_bins: DiscretizationConfig = 7,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cuda",
):
    """Run complete data preparation pipeline.

    Args:
        perception_episodes: Episodes for perception model training
        dynamics_episodes: Episodes for dynamics collection
        num_bins: Number of bins per dimension. Can be:
            - int: Same number of bins for all dimensions (e.g., 7)
            - List[int]: Per-dimension bins [n_x, n_xdot, n_theta, n_thetadot] (e.g., [5, 4, 5, 4])
        epochs: Training epochs for perception model
        seed: Random seed
        device: Device for training
    """
    print("\n" + "█" * 60)
    print("CartPole Data Preparation Pipeline")
    print("█" * 60 + "\n")

    # Step 1: Perception data
    prepare_perception_data(
        num_episodes=perception_episodes,
        epochs=epochs,
        num_bins=num_bins,
        seed=seed,
        device=device,
    )

    print("\n")

    # Step 2: Dynamics data (will infer bins from bin_edges created in step 1)
    prepare_dynamics_data(
        num_episodes=dynamics_episodes,
        seed=seed + 1,  # Different seed for diversity
    )

    print("\n" + "█" * 60)
    print("All data preparation complete!")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    # Run with default parameters
    prepare_all_data()
