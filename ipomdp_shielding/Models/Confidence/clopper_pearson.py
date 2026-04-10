"""Clopper-Pearson exact confidence intervals with simplex projection."""

from collections import defaultdict, Counter
from typing import Iterable, Hashable, Dict, Any, Tuple, List

from statsmodels.stats.proportion import proportion_confint

from .simplex_projection import project_intervals_to_simplex


def estimate_clopper_pearson(
    data: Iterable[Tuple[Hashable, Hashable]],
    alpha: float = 0.05,
) -> Dict[Hashable, Dict[Hashable, Dict[str, float]]]:
    """
    Compute Clopper-Pearson (exact) confidence intervals for P(s_est | s_true)
    and then project the resulting per-state intervals onto the probability
    simplex so that there exists at least one valid probability vector
    consistent with all intervals simultaneously.

    Parameters
    ----------
    data : iterable of (true_state, estimated_state)
        Each pair is one observation from P(s_est | s_true).
    alpha : float
        Significance level for the confidence intervals (default 0.05 -> 95% CI).

    Returns
    -------
    result : dict
        Nested dict of the form:
        {
          true_state: {
            est_state: {
              "count": k_ij,
              "total": n_i,
              "p_hat": k_ij / n_i,
              "lower": lower_CI,
              "upper": upper_CI,
            },
            ...
          },
          ...
        }
    """
    counts = defaultdict(Counter)
    for s_true, s_est in data:
        counts[s_true][s_est] += 1

    result: Dict[Any, Dict[Any, Dict[str, float]]] = {}

    for s_true, est_counter in counts.items():
        est_states: List[Hashable] = sorted(est_counter.keys())
        n_i = sum(est_counter.values())
        if n_i == 0:
            continue

        lowers: List[float] = []
        uppers: List[float] = []
        p_hats: List[float] = []
        counts_vec: List[int] = []

        for s_est in est_states:
            k_ij = est_counter[s_est]
            counts_vec.append(k_ij)
            p_hat = k_ij / n_i
            p_hats.append(p_hat)

            lower, upper = proportion_confint(
                count=k_ij,
                nobs=n_i,
                alpha=alpha,
                method="beta",
            )
            lowers.append(lower)
            uppers.append(upper)

        lowers_adj, uppers_adj = project_intervals_to_simplex(lowers, uppers)

        result[s_true] = {}
        for s_est, k_ij, p_hat, lo, up in zip(
            est_states, counts_vec, p_hats, lowers_adj, uppers_adj
        ):
            result[s_true][s_est] = {
                "count": float(k_ij),
                "total": float(n_i),
                "p_hat": float(p_hat),
                "lower": float(lo),
                "upper": float(up),
            }

    return result
