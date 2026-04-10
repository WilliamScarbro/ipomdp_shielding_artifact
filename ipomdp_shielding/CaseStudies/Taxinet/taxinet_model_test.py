"""Test TaxiNet perception model bounds against held-out test data."""
 
import random
from collections import Counter
from typing import Dict, List, Tuple

from .data_loader import get_cte_data, get_he_data
from .taxinet import (
    taxinet_he_states,
    taxinet_cte_states,
    _fill_imdp_coverage,
)
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
    """
    Compute empirical observation frequencies from test data.

    Returns
    -------
    dict
        {true_state: {est_state: frequency}} for each true_state with
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
    """
    Check if empirical frequencies satisfy the IMDP interval bounds.

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
    smoothing: bool = True,
    seed: int = 42,
):
    """
    Test the perception model interval bounds against held-out test data.

    Builds the perception model from training data and checks whether
    the empirical frequencies in the test data fall within the model's
    probability interval bounds.
    """
    cte_data = get_cte_data()
    he_data = get_he_data()

    cte_train, cte_test = split_data(cte_data, train_fraction, seed)
    he_train, he_test = split_data(he_data, train_fraction, seed + 1)

    he_states = taxinet_he_states()
    cte_states = taxinet_cte_states()

    # Build per-dimension perception models from raw (unsmoothed) data
    he_CI = ConfidenceInterval(he_train)
    cte_CI = ConfidenceInterval(cte_train)

    he_imdp = he_CI.produce_imdp(he_states, "PERC", confidence_method, alpha)
    cte_imdp = cte_CI.produce_imdp(cte_states, "PERC", confidence_method, alpha)

    # Apply coverage-only smoothing: fill missing pairs with [0, CI_upper(0, n)]
    if smoothing:
        _fill_imdp_coverage(he_imdp, he_states, he_train, "PERC", alpha)
        _fill_imdp_coverage(cte_imdp, cte_states, cte_train, "PERC", alpha)

    # Compute empirical frequencies from test data
    he_freqs = empirical_frequencies(he_test, he_states)
    cte_freqs = empirical_frequencies(cte_test, cte_states)

    # Check bounds
    he_violations = check_bounds(he_freqs, he_imdp, he_states)
    cte_violations = check_bounds(cte_freqs, cte_imdp, cte_states)

    # Report
    he_total_checks = sum(len(he_states) for s in he_freqs if (s, "PERC") in he_imdp.P_lower)
    cte_total_checks = sum(len(cte_states) for s in cte_freqs if (s, "PERC") in cte_imdp.P_lower)

    print("=" * 60)
    print(f"Perception Model Bound Test (method={confidence_method}, "
          f"alpha={alpha}, smoothing={smoothing}, seed={seed})")
    print("=" * 60)

    print(f"\n--- Heading Error (HE) ---")
    print(f"Total bound checks: {he_total_checks}")
    print(f"Violations: {len(he_violations)}")
    if he_violations:
        max_v = max(v["violation"] for v in he_violations)
        avg_v = sum(v["violation"] for v in he_violations) / len(he_violations)
        print(f"Max violation magnitude: {max_v:.6f}")
        print(f"Avg violation magnitude: {avg_v:.6f}")
        print(f"Details:")
        for v in he_violations:
            print(f"  true={v['true_state']:+d} est={v['est_state']:+d}: "
                  f"freq={v['frequency']:.4f} bounds=[{v['lower']:.4f}, {v['upper']:.4f}] "
                  f"violation={v['violation']:.6f}")

    print(f"\n--- Cross-Track Error (CTE) ---")
    print(f"Total bound checks: {cte_total_checks}")
    print(f"Violations: {len(cte_violations)}")
    if cte_violations:
        max_v = max(v["violation"] for v in cte_violations)
        avg_v = sum(v["violation"] for v in cte_violations) / len(cte_violations)
        print(f"Max violation magnitude: {max_v:.6f}")
        print(f"Avg violation magnitude: {avg_v:.6f}")
        print(f"Details:")
        for v in cte_violations:
            print(f"  true={v['true_state']:+d} est={v['est_state']:+d}: "
                  f"freq={v['frequency']:.4f} bounds=[{v['lower']:.4f}, {v['upper']:.4f}] "
                  f"violation={v['violation']:.6f}")

    total_violations = len(he_violations) + len(cte_violations)
    total_checks = he_total_checks + cte_total_checks
    print(f"\n--- Summary ---")
    print(f"Total checks: {total_checks}")
    print(f"Total violations: {total_violations} ({100*total_violations/total_checks:.1f}%)")

    return he_violations, cte_violations


if __name__ == "__main__":
    seed: int = 30
    alpha: float = 0.05
    print("\n>>> With smoothing (default):")
    test_perception_model(smoothing=True, alpha=alpha, seed=seed)

    print("\n\n>>> Without smoothing:")
    test_perception_model(smoothing=False, alpha=alpha, seed=seed)
