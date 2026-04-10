"""CartPole case study: Vision-based perception shielding for continuous state space.

This module implements a CartPole IPOMDP where:
- State space: 4D continuous (position, velocity, pole angle, angular velocity) discretized with configurable bins per dimension
- Actions: {0=left, 1=right}
- Dynamics: Empirical MDP from gymnasium rollouts
- Perception: Factored IMDP from CNN estimates (product of 4 independent dimension IMDPs)
- Safety: Within CartPole-v1 termination bounds
"""

import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import Counter

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from ...Models.imdp import IMDP, product_imdp
from ...Models.ipomdp import IPOMDP
from ...Models.mdp import MDP
from ...Models.Confidence.confidence import ConfidenceInterval
from .data_loader import get_bin_edges, get_confusion_data

# Type aliases
State = Tuple[int, int, int, int]  # (x_bin, x_dot_bin, theta_bin, theta_dot_bin)
FAIL = "FAIL"  # Absorbing failure state

# Discretization configuration type
# Can be either:
# - int: uniform bins across all dimensions
# - List[int]: per-dimension bin counts [n_x, n_xdot, n_theta, n_thetadot]
DiscretizationConfig = Union[int, List[int]]


def cartpole_states(num_bins: DiscretizationConfig = 7, with_fail: bool = False) -> List:
    """Generate all discretized CartPole states.

    Args:
        num_bins: Number of bins per dimension. Can be:
            - int: Same number of bins for all dimensions (e.g., 7)
            - List[int]: Per-dimension bins [n_x, n_xdot, n_theta, n_thetadot] (e.g., [5, 4, 5, 4])
        with_fail: Whether to include FAIL absorbing state

    Returns:
        List of states. Each non-FAIL state is a tuple (x_bin, x_dot_bin, theta_bin, theta_dot_bin)
    """
    # Parse bin configuration
    if isinstance(num_bins, int):
        bins = [num_bins] * 4
    else:
        if len(num_bins) != 4:
            raise ValueError(f"num_bins list must have 4 elements, got {len(num_bins)}")
        bins = num_bins

    states = [
        (x_bin, xdot_bin, theta_bin, thetadot_bin)
        for x_bin in range(bins[0])
        for xdot_bin in range(bins[1])
        for theta_bin in range(bins[2])
        for thetadot_bin in range(bins[3])
    ]
    if with_fail:
        states.append(FAIL)
    return states


def cartpole_actions(num_bins: DiscretizationConfig = 7) -> Dict:
    """Generate action dictionary for CartPole.

    Args:
        num_bins: Number of bins per dimension (int or List[int])

    Returns:
        Dictionary mapping each state to list of available actions [0=left, 1=right]
    """
    states = cartpole_states(num_bins, with_fail=True)
    actions = {}
    for s in states:
        if s == FAIL:
            actions[s] = []  # No actions from FAIL state
        else:
            actions[s] = [0, 1]  # left, right
    return actions


def cartpole_dynamics() -> MDP:
    """Load empirical dynamics MDP from file.

    Returns:
        MDP with empirical transition probabilities collected from gymnasium
    """
    dynamics_file = Path(__file__).parent / "artifacts" / "dynamics_mdp.pkl"
    with open(dynamics_file, "rb") as f:
        return pickle.load(f)


def _fill_imdp_coverage(
    imdp: IMDP,
    states: List[int],
    data: List[Tuple[int, int]],
    action: str,
    alpha: float
) -> None:
    """Fill missing state pairs in an IMDP with conservative zero-count bounds.

    For each (true_state, est_state) pair not present in the IMDP, adds
    lower=0 and upper=CI_upper(0, n_true) where n_true is the total number
    of observations for that true_state. This provides coverage without
    biasing the intervals for observed pairs.

    Args:
        imdp: IMDP to modify in-place
        states: List of valid states
        data: List of (true_state, est_state) observation pairs
        action: Action name (typically "PERC")
        alpha: Significance level for confidence intervals
    """
    # Count total observations per true state
    totals = Counter(s_true for s_true, _ in data)

    for s_true in states:
        key = (s_true, action)
        n = totals.get(s_true, 0)
        if n == 0:
            # No observations at all for this true state; assign uniform bounds
            imdp.P_lower[key] = {s: 0.0 for s in states}
            imdp.P_upper[key] = {s: 1.0 for s in states}
            continue

        if key not in imdp.P_lower:
            imdp.P_lower[key] = {}
            imdp.P_upper[key] = {}

        # Compute zero-count upper bound for this sample size
        _, zero_upper = proportion_confint(0, n, alpha=alpha, method="beta")

        for s_est in states:
            if s_est not in imdp.P_lower[key]:
                imdp.P_lower[key][s_est] = 0.0
                imdp.P_upper[key][s_est] = zero_upper


def cartpole_perception(
    confidence_method: str,
    alpha: float,
    confusion_data: Dict[str, List[Tuple[int, int]]],
    num_bins: DiscretizationConfig = 7,
    smoothing: bool = True,
) -> IMDP:
    """Build factored perception IMDP from confusion matrix data.

    Constructs a 4D perception IMDP by taking the product of 4 independent
    dimension IMDPs (x, x_dot, theta, theta_dot). This assumes perception
    errors are independent across dimensions.

    Args:
        confidence_method: Confidence interval method (e.g., "Clopper_Pearson")
        alpha: Significance level for confidence intervals
        confusion_data: Dictionary mapping dimension names to (true_bin, est_bin) tuples
        num_bins: Number of bins per dimension (int or List[int])
        smoothing: Whether to fill unobserved state pairs with conservative bounds

    Returns:
        IMDP with 4D state space and single "PERC" action
    """
    # Parse bin configuration
    if isinstance(num_bins, int):
        bins = [num_bins] * 4
    else:
        if len(num_bins) != 4:
            raise ValueError(f"num_bins list must have 4 elements, got {len(num_bins)}")
        bins = num_bins

    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    dim_imdps = []

    perceive_action = "PERC"

    for i, dim in enumerate(dim_names):
        # Build confidence interval model for this dimension
        dim_CI = ConfidenceInterval(confusion_data[dim])

        # States for this dimension are bin indices 0..bins[i]-1
        dim_states = list(range(bins[i]))

        # Produce IMDP for this dimension
        dim_imdp = dim_CI.produce_imdp(dim_states, perceive_action, confidence_method, alpha)

        # Apply smoothing if requested
        if smoothing:
            _fill_imdp_coverage(dim_imdp, dim_states, confusion_data[dim], perceive_action, alpha)

        dim_imdps.append(dim_imdp)

    # Combine into 4D perception IMDP via nested products
    # product_imdp expects states to be tuples
    perception_imdp = product_imdp(
        product_imdp(dim_imdps[0], dim_imdps[1]),
        product_imdp(dim_imdps[2], dim_imdps[3])
    )

    # Add FAIL state self-loop (absorbing state)
    perception_imdp.actions[FAIL] = [perceive_action]
    perception_imdp.P_lower[(FAIL, perceive_action)] = {FAIL: 1.0}
    perception_imdp.P_upper[(FAIL, perceive_action)] = {FAIL: 1.0}

    return perception_imdp


def cartpole_safe(state, bin_edges: np.ndarray) -> bool:
    """Check if a state is safe (within CartPole-v1 termination bounds).

    Args:
        state: Discretized state tuple or FAIL
        bin_edges: Array of shape (4, k+1) with bin edges for each dimension

    Returns:
        True if state is safe, False otherwise
    """
    if state == FAIL:
        return False

    x_bin, x_dot_bin, theta_bin, theta_dot_bin = state

    # Map bin to continuous value (use bin center)
    x = (bin_edges[0][x_bin] + bin_edges[0][x_bin + 1]) / 2
    theta = (bin_edges[2][theta_bin] + bin_edges[2][theta_bin + 1]) / 2

    # CartPole-v1 thresholds
    x_threshold = 2.4
    theta_threshold = 0.209  # 12 degrees in radians

    return abs(x) <= x_threshold and abs(theta) <= theta_threshold


def cartpole_safe_action(state, action: int, dynamics: MDP, bin_edges: np.ndarray) -> bool:
    """Check if a state-action pair is safe (one-step lookahead).

    Args:
        state: Current discretized state
        action: Action to take (0=left, 1=right)
        dynamics: Dynamics MDP
        bin_edges: Bin edges for safety checking

    Returns:
        True if action is safe from this state, False otherwise
    """
    if not cartpole_safe(state, bin_edges):
        return False

    # Check all possible next states
    if (state, action) not in dynamics.P:
        # Unobserved state-action pair; conservatively mark unsafe
        return False

    next_state_dist = dynamics.P[(state, action)]
    return all(cartpole_safe(s_next, bin_edges) for s_next in next_state_dist.keys())


def build_cartpole_ipomdp(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    train_fraction: float = 0.8,
    num_bins: DiscretizationConfig = 7,
    smoothing: bool = True,
    seed: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[IPOMDP, Dict, Dict, None]:
    """Build complete CartPole IPOMDP with train/test split.

    This is the main entry point for constructing a CartPole IPOMDP. It:
    1. Loads perception data and splits into train/test
    2. Builds perception IMDP from training data
    3. Loads dynamics MDP
    4. Constructs complete IPOMDP
    5. Builds perfect-perception shield

    Args:
        confidence_method: Confidence interval method
        alpha: Significance level (default 0.05 = 95% confidence)
        train_fraction: Fraction of data to use for training (default 0.8)
        num_bins: Number of bins per dimension. Can be:
            - int: Same number of bins for all dimensions (e.g., 7)
            - List[int]: Per-dimension bins [n_x, n_xdot, n_theta, n_thetadot] (e.g., [5, 4, 5, 4])
        smoothing: Whether to apply coverage-only smoothing
        seed: Random seed for train/test split
        data_dir: Optional path to data directory for perception artifacts.
            If None, uses default artifacts/ subdirectory.

    Returns:
        Tuple of (ipomdp, pp_shield, test_data, None) where:
        - ipomdp: Complete IPOMDP model
        - pp_shield: Perfect-perception shield (dict mapping state -> set of safe actions)
        - test_data: Test set confusion data for validation
        - None: Placeholder for interface compatibility
    """
    if seed is not None:
        random.seed(seed)

    # Load bin edges
    bin_edges = get_bin_edges(data_dir)

    # Load and split perception data
    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    confusion_data_full = {
        dim: get_confusion_data(dim, data_dir)
        for dim in dim_names
    }

    # Train/test split
    confusion_train = {}
    confusion_test = {}
    for dim, data in confusion_data_full.items():
        data_copy = data.copy()
        random.shuffle(data_copy)
        split_idx = int(len(data_copy) * train_fraction)
        confusion_train[dim] = data_copy[:split_idx]
        confusion_test[dim] = data_copy[split_idx:]

    # Build perception IMDP from training data
    perc_imdp = cartpole_perception(
        confidence_method, alpha, confusion_train, num_bins, smoothing
    )

    # Map between nested tuple format (from product_imdp) and flat tuple format
    def flatten_state(s):
        """Convert ((x, xd), (th, thd)) -> (x, xd, th, thd)"""
        if s == FAIL:
            return FAIL
        return (s[0][0], s[0][1], s[1][0], s[1][1])

    def nest_state(s):
        """Convert (x, xd, th, thd) -> ((x, xd), (th, thd))"""
        if s == FAIL:
            return FAIL
        return ((s[0], s[1]), (s[2], s[3]))

    # Strip "PERC" action to match IPOMDP format (state->state transitions only)
    states = cartpole_states(num_bins, with_fail=True)
    perc_L = {}
    perc_U = {}

    for s in states:
        s_nested = nest_state(s)
        perc_L[s] = {}
        perc_U[s] = {}

        for s2_nested, prob_lower in perc_imdp.P_lower[(s_nested, "PERC")].items():
            s2 = flatten_state(s2_nested)
            perc_L[s][s2] = prob_lower

        for s2_nested, prob_upper in perc_imdp.P_upper[(s_nested, "PERC")].items():
            s2 = flatten_state(s2_nested)
            perc_U[s][s2] = prob_upper

    # Load dynamics MDP
    dyn_mdp = cartpole_dynamics()
    actions = [0, 1]

    # Filter dynamics to the requested state space.
    # The stored dynamics_mdp.pkl may have been generated with a different (e.g. coarser)
    # number of bins.  Any next-state that falls outside the requested state space is
    # treated as a transition to FAIL (absorbing failure state).
    states_set = set(states)
    filtered_P: Dict[Tuple, Dict] = {}
    for (s, a), dist in dyn_mdp.P.items():
        if s not in states_set:
            continue  # source state is out of range – skip
        fd: Dict = {}
        for sp, p in dist.items():
            if sp in states_set:
                fd[sp] = fd.get(sp, 0.0) + p
            else:
                fd[FAIL] = fd.get(FAIL, 0.0) + p
        filtered_P[(s, a)] = fd if fd else {FAIL: 1.0}
    # Ensure every (s, a) pair for in-range states has an entry
    for s in states:
        if s == FAIL:
            continue
        for a in actions:
            if (s, a) not in filtered_P:
                filtered_P[(s, a)] = {s: 1.0}  # conservative self-loop

    filtered_dyn = MDP(states, cartpole_actions(num_bins), filtered_P)

    # Filter perception bounds so that all observation keys are valid states.
    # Out-of-range observations are dropped from lower bounds (keeps sum ≤ 1) and
    # their upper-bound mass is transferred to FAIL (keeps sum ≥ 1).
    filtered_perc_L: Dict = {}
    filtered_perc_U: Dict = {}
    for s in states:
        lo: Dict = {}
        hi: Dict = {}
        for s2, p in perc_L[s].items():
            if s2 in states_set:
                lo[s2] = p
        for s2, p in perc_U[s].items():
            if s2 in states_set:
                hi[s2] = p
            else:
                hi[FAIL] = hi.get(FAIL, 0.0) + p
        filtered_perc_L[s] = lo
        filtered_perc_U[s] = hi

    # Construct IPOMDP
    cartpole_ipomdp = IPOMDP(states, states, actions, filtered_P, filtered_perc_L, filtered_perc_U)

    # Build perfect-perception shield (one-step lookahead)
    pp_shield = {
        s: {a for a in actions if cartpole_safe_action(s, a, filtered_dyn, bin_edges)}
        for s in states if s != FAIL
    }
    pp_shield[FAIL] = set()  # No safe actions from FAIL

    return cartpole_ipomdp, pp_shield, confusion_test, None


def cartpole_evaluation():
    """Run evaluation of CartPole IPOMDP with exact belief propagation.

    This is a placeholder for future evaluation code that would:
    1. Build the IPOMDP
    2. Create belief propagator
    3. Create runtime shield
    4. Run Monte Carlo trials
    5. Report metrics (fail rate, stuck rate, etc.)
    """
    from ...Propagators.ExactIHMMBelief import ExactIHMMBelief
    from ...Evaluation.runtime_shield import RuntimeImpShield

    print("=" * 60)
    print("CartPole IPOMDP Evaluation")
    print("=" * 60)

    # Build IPOMDP
    print("\nBuilding IPOMDP...")
    cartpole_ipomdp, pp_shield, test_data, _ = build_cartpole_ipomdp()

    print(f"States: {len(cartpole_ipomdp.states)}")
    print(f"Actions: {len(cartpole_ipomdp.actions)}")
    print(f"Shield entries: {len(pp_shield)}")

    # Create belief propagator
    print("\nCreating exact belief propagator...")
    exact_belief = ExactIHMMBelief(cartpole_ipomdp)

    # Create runtime shield
    print("Creating runtime shield...")
    exact_rtips = RuntimeImpShield(pp_shield, exact_belief, 0.5, 0)

    print("\n" + "=" * 60)
    print("Setup complete! Ready for Monte Carlo evaluation.")
    print("=" * 60)

    # TODO: Implement Monte Carlo evaluation
    # This would require a perceptor function that samples from test_data
    # and integration with the MonteCarlo evaluation framework

    return cartpole_ipomdp, pp_shield, test_data, exact_rtips
