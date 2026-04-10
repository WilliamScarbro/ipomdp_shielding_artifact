"""Simplex projection utility for confidence intervals."""

from typing import List, Tuple


def project_intervals_to_simplex(
    lowers: List[float],
    uppers: List[float],
    eps: float = 1e-12,
) -> Tuple[List[float], List[float]]:
    """
    Simplex-feasibility adjustment.

    Given component-wise intervals [l_i, u_i] for i=1..K, adjust them so that:

      - 0 <= l_i <= u_i <= 1
      - sum_i l_i <= 1 <= sum_i u_i

    This guarantees that there exists at least one probability vector
    p in the simplex such that l_i <= p_i <= u_i.
    """
    K = len(lowers)
    l = [max(0.0, min(1.0, li)) for li in lowers]
    u = [max(0.0, min(1.0, ui)) for ui in uppers]

    # Make sure l_i <= u_i
    for i in range(K):
        if l[i] > u[i]:
            mid = 0.5 * (l[i] + u[i])
            l[i] = u[i] = max(0.0, min(1.0, mid))

    sum_l = sum(l)
    sum_u = sum(u)

    # If lower bounds sum to more than 1, scale them down proportionally
    if sum_l > 1.0 + eps:
        factor = 1.0 / sum_l
        l = [li * factor for li in l]
        sum_l = 1.0

    # If upper bounds sum to less than 1, scale them up proportionally
    sum_u = sum(u)
    if sum_u < 1.0 - eps and sum_u > 0.0:
        factor = 1.0 / sum_u
        u = [min(1.0, ui * factor) for ui in u]
        sum_u = sum(u)

    # One more safety pass on ordering
    for i in range(K):
        if l[i] > u[i]:
            l[i] = u[i]

    return l, u
