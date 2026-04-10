"""Goodman's multinomial confidence intervals."""

from collections import defaultdict, Counter
from typing import Iterable, Hashable, Dict, Tuple, Any

from statsmodels.stats.proportion import multinomial_proportions_confint


def estimate_goodman(
    data: Iterable[Tuple[Hashable, Hashable]],
    alpha: float = 0.05,
) -> Dict[Hashable, Dict[Hashable, Dict[str, float]]]:
    """
    Compute simultaneous Goodman's confidence intervals for P(s_est | s_true)
    from (state, state_estimate) observations.

    Parameters
    ----------
    data : iterable of (true_state, estimated_state)
        Each pair is one observation from P(s_est | s_true).
    alpha : float
        Significance level for the simultaneous CIs (default 0.05 -> 95% CIs).

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
        est_states = sorted(est_counter.keys())
        count_vec = [est_counter[e] for e in est_states]
        n_i = sum(count_vec)

        confint = multinomial_proportions_confint(
            count_vec,
            alpha=alpha,
            method="goodman",
        )

        result[s_true] = {}
        for est_state, k_ij, (lower, upper) in zip(est_states, count_vec, confint):
            p_hat = k_ij / n_i if n_i > 0 else 0.0
            result[s_true][est_state] = {
                "count": float(k_ij),
                "total": float(n_i),
                "p_hat": p_hat,
                "lower": float(lower),
                "upper": float(upper),
            }

    return result
