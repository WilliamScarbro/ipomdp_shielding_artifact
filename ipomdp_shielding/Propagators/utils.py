"""Shared utilities for belief propagation."""

from typing import Dict, Tuple

State = type
Action = type
Observation = type


def tightened_likelihood_bounds(ipomdp, s, o_obs) -> Tuple[float, float]:
    """
    Tighten bounds for the observed symbol o_obs using the row-sum (simplex) constraint:
      sum_o Z(o|s) = 1,  Z(o|s) in [L_s(o), U_s(o)].

    Returns: (L_eff, U_eff) for Z(o_obs | s).
    """
    L_row = ipomdp.P_lower[s]
    U_row = ipomdp.P_upper[s]

    L = float(L_row.get(o_obs, 0.0))
    U = float(U_row.get(o_obs, 0.0))

    sum_U_others = 0.0
    sum_L_others = 0.0
    for o in ipomdp.observations:
        if o == o_obs:
            continue
        sum_U_others += float(U_row.get(o, 0.0))
        sum_L_others += float(L_row.get(o, 0.0))

    L_eff = max(L, 1.0 - sum_U_others)
    U_eff = min(U, 1.0 - sum_L_others)

    L_eff = max(0.0, min(1.0, L_eff))
    U_eff = max(0.0, min(1.0, U_eff))

    if L_eff - 1e-10 > U_eff:
        return 1.0, 0.0

    return L_eff, U_eff


def transition_update(ipomdp, b: Dict, a) -> Dict:
    """
    Exact belief prediction under known dynamics:
      b_pred(s') = sum_s T(s'|s,a) * b(s)
    """
    S = ipomdp.states
    b_pred = {s_next: 0.0 for s_next in S}
    for s in S:
        bs = float(b.get(s, 0.0))
        if bs == 0.0:
            continue
        trans = ipomdp.T.get((s, a), {})
        for s_next, p in trans.items():
            if p:
                b_pred[s_next] += bs * float(p)

    total = sum(b_pred.values())
    if total > 0.0:
        inv = 1.0 / total
        for s_next in S:
            b_pred[s_next] *= inv
    return b_pred
