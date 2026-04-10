"""Sidak-Wilson hybrid confidence intervals."""

from collections import defaultdict, Counter
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Hashable, Dict, Tuple, Any, List

from .simplex_projection import project_intervals_to_simplex


def _wilson_interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion k/n at level 1 - alpha.
    """
    if n == 0:
        return 0.0, 1.0

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def estimate_sidak_wilson_hybrid(
    data: Iterable[Tuple[Hashable, Hashable]],
    alpha: float = 0.05,
) -> Dict[Hashable, Dict[Hashable, Dict[str, float]]]:
    """
    Estimate confidence intervals for P(s_est | s_true) using the
    Sidak-corrected Wilson + simplex hybrid method.

    For each true_state s_true:
      - Use Sidak to get a per-category marginal level that yields
        (approximate) simultaneous (1 - alpha) coverage across categories.
      - For each category, compute a Wilson score binomial CI at this level.
      - Project the resulting intervals onto the probability simplex.

    Parameters
    ----------
    data : iterable of (true_state, estimated_state)
    alpha : float
        Desired simultaneous confidence level (e.g. 0.05 for 95%).

    Returns
    -------
    result : dict
        Nested dict with count, total, p_hat, lower, upper for each state pair.
    """
    counts = defaultdict(Counter)
    for s_true, s_est in data:
        counts[s_true][s_est] += 1

    result: Dict[Any, Dict[Any, Dict[str, float]]] = {}

    for s_true, est_counter in counts.items():
        est_states = sorted(est_counter.keys())
        count_vec = [est_counter[e] for e in est_states]
        n_i = sum(count_vec)
        K = len(est_states)

        if K == 0 or n_i == 0:
            continue

        # Sidak correction
        alpha_marg = 1.0 - (1.0 - alpha) ** (1.0 / K)

        lowers: List[float] = []
        uppers: List[float] = []
        p_hats: List[float] = []
        for k_ij in count_vec:
            p_hat = k_ij / n_i
            p_hats.append(p_hat)
            lo, up = _wilson_interval(k_ij, n_i, alpha_marg)
            lowers.append(lo)
            uppers.append(up)

        lowers_adj, uppers_adj = project_intervals_to_simplex(lowers, uppers)

        result[s_true] = {}
        for est_state, k_ij, p_hat, lo, up in zip(
            est_states, count_vec, p_hats, lowers_adj, uppers_adj
        ):
            result[s_true][est_state] = {
                "count": float(k_ij),
                "total": float(n_i),
                "p_hat": float(p_hat),
                "lower": float(lo),
                "upper": float(up),
            }

    return result
