"""Test CartPole perception model bounds against held-out test data."""

import random
from collections import Counter
from typing import Dict, List, Tuple

from .data_loader import get_confusion_data
from .cartpole import _fill_imdp_coverage
from ...Models.Confidence import ConfidenceInterval


def split_data(data: List[Tuple[int, int]], train_fraction: float, seed: int):
    """Split data into train/test with a fixed seed."""
    random.seed(seed)
    data = list(data)
    random.shuffle(data)
    split_idx = int(len(data) * train_fraction)
    return data[:split_idx], data[split_idx:]


def empirical_frequencies(
    test_data: List[Tuple[int, int]],
    states: List[int],
) -> Dict[int, Dict[int, float]]:
    """Compute empirical observation frequencies from test data.

    Returns:
        dict: {true_state: {est_state: frequency}} for each true_state with
        observations in the test set. States with zero total observations
        are omitted.
    """
    counts: Dict[int, Counter] = {s: Counter() for s in states}
    totals: Dict[int, int] = {s: 0 for s in states}

    for s_true, s_est in test_data:
        counts[s_true][s_est] += 1
        totals[s_true] += 1

    freqs = {}
    for s in states:
        if totals[s] == 0:
            continue
        freqs[s] = {s2: counts[s][s2] / totals[s] for s2 in states}
    return freqs


def check_bounds(
    freqs: Dict[int, Dict[int, float]],
    imdp,
    states: List[int],
    action: str = "PERC",
) -> List[Dict]:
    """Check if empirical frequencies satisfy the IMDP interval bounds.

    Returns a list of violation records, each containing:
        - true_state: the conditioning state
        - est_state: the observed state
        - frequency: empirical frequency from test data
        - lower: model lower bound
        - upper: model upper bound
        - violation: signed magnitude (negative = below lower, positive = above upper)
    """
    violations = []
    for s_true in freqs:
        if (s_true, action) not in imdp.P_lower:
            continue
        for s_est in states:
            freq = freqs[s_true].get(s_est, 0.0)
            lower = imdp.P_lower[(s_true, action)].get(s_est, 0.0)
            upper = imdp.P_upper[(s_true, action)].get(s_est, 1.0)

            if freq < lower:
                violations.append({
                    "true_state": s_true,
                    "est_state": s_est,
                    "frequency": freq,
                    "lower": lower,
                    "upper": upper,
                    "violation": lower - freq,
                })
            elif freq > upper:
                violations.append({
                    "true_state": s_true,
                    "est_state": s_est,
                    "frequency": freq,
                    "lower": lower,
                    "upper": upper,
                    "violation": freq - upper,
                })
    return violations


def test_perception_model(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    train_fraction: float = 0.8,
    num_bins: int = 7,
    smoothing: bool = True,
    seed: int = 42,
):
    """Test the perception model interval bounds against held-out test data.

    Builds the perception model from training data and checks whether
    the empirical frequencies in the test data fall within the model's
    probability interval bounds.

    Args:
        confidence_method: Confidence interval method (e.g., "Clopper_Pearson")
        alpha: Significance level (default 0.05 = 95% confidence)
        train_fraction: Fraction of data to use for training
        num_bins: Number of bins per dimension
        smoothing: Whether to apply coverage-only smoothing
        seed: Random seed for reproducibility
    """
    dim_names = ["x", "x_dot", "theta", "theta_dot"]

    # Load and split data for each dimension
    train_data = {}
    test_data = {}
    for i, dim in enumerate(dim_names):
        full_data = get_confusion_data(dim)
        train_data[dim], test_data[dim] = split_data(full_data, train_fraction, seed + i)

    states = list(range(num_bins))

    # Build per-dimension perception models
    dim_imdps = {}
    for dim in dim_names:
        # Build IMDP from training data
        dim_CI = ConfidenceInterval(train_data[dim])
        dim_imdps[dim] = dim_CI.produce_imdp(states, "PERC", confidence_method, alpha)

        # Apply smoothing if requested
        if smoothing:
            _fill_imdp_coverage(dim_imdps[dim], states, train_data[dim], "PERC", alpha)

    # Compute empirical frequencies from test data
    empirical_freqs = {
        dim: empirical_frequencies(test_data[dim], states)
        for dim in dim_names
    }

    # Check bounds for each dimension
    violations = {}
    total_checks = {}
    for dim in dim_names:
        violations[dim] = check_bounds(empirical_freqs[dim], dim_imdps[dim], states)
        total_checks[dim] = sum(len(states) for s in empirical_freqs[dim]
                                if (s, "PERC") in dim_imdps[dim].P_lower)

    # Report results
    print("=" * 70)
    print(f"CartPole Perception Model Bound Test")
    print(f"Method: {confidence_method}, Alpha: {alpha}, Smoothing: {smoothing}, Seed: {seed}")
    print("=" * 70)

    for dim in dim_names:
        print(f"\n--- {dim} dimension ---")
        print(f"Total bound checks: {total_checks[dim]}")
        print(f"Violations: {len(violations[dim])}")

        if violations[dim]:
            max_v = max(v["violation"] for v in violations[dim])
            avg_v = sum(v["violation"] for v in violations[dim]) / len(violations[dim])
            print(f"Max violation magnitude: {max_v:.6f}")
            print(f"Avg violation magnitude: {avg_v:.6f}")

            if len(violations[dim]) <= 10:  # Only print details if not too many
                print(f"Details:")
                for v in violations[dim]:
                    print(f"  true={v['true_state']} est={v['est_state']}: "
                          f"freq={v['frequency']:.4f} bounds=[{v['lower']:.4f}, {v['upper']:.4f}] "
                          f"violation={v['violation']:.6f}")

    # Summary
    total_violations_all = sum(len(violations[dim]) for dim in dim_names)
    total_checks_all = sum(total_checks[dim] for dim in dim_names)

    print(f"\n{'=' * 70}")
    print(f"--- Summary ---")
    print(f"Total checks across all dimensions: {total_checks_all}")
    print(f"Total violations: {total_violations_all} "
          f"({100 * total_violations_all / total_checks_all:.1f}% if total_checks_all > 0 else 0.0)")
    print(f"Expected violation rate (alpha): {100 * alpha:.1f}%")

    if total_violations_all / total_checks_all <= alpha:
        print("\n✓ Violation rate is within expected bounds!")
    else:
        print(f"\n⚠ Violation rate exceeds expected alpha level")

    print("=" * 70)

    return violations


if __name__ == "__main__":
    seed: int = 42
    alpha: float = 0.05

    print("\n>>> With smoothing (default):")
    test_perception_model(smoothing=True, alpha=alpha, seed=seed)

    print("\n\n>>> Without smoothing:")
    test_perception_model(smoothing=False, alpha=alpha, seed=seed)
