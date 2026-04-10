"""
TaxiNet POMDP Adapter: Convert IPOMDP to POMDP for Carr shield comparison.

Converts TaxiNet IPOMDP (with interval observation probabilities) to a standard POMDP
by selecting specific observation probabilities within the intervals.
"""

from typing import Dict, Tuple, FrozenSet
import random
from ipomdp_shielding.Models.ipomdp import IPOMDP
from ipomdp_shielding.Models.pomdp import POMDP, State, Action, Observation


def convert_taxinet_to_pomdp(
    ipomdp: IPOMDP,
    realization: str = "midpoint",
    seed: int | None = None
) -> POMDP:
    """
    Convert TaxiNet IPOMDP to POMDP by selecting observation probabilities.

    Args:
        ipomdp: The TaxiNet IPOMDP model
        realization: Strategy for selecting probabilities from intervals:
            - "midpoint": P(o|s) = (P_lower(o|s) + P_upper(o|s)) / 2
            - "lower": P(o|s) = P_lower(o|s) (pessimistic)
            - "upper": P(o|s) = P_upper(o|s) (optimistic)
            - "random": Random point within intervals
        seed: Random seed for "random" realization

    Returns:
        POMDP with exact observation probabilities
    """
    if seed is not None:
        random.seed(seed)

    # Convert transition probabilities (already exact in TaxiNet)
    T: Dict[Tuple[State, Action], Dict[State, float]] = {}
    for s in ipomdp.states:
        for a in ipomdp.actions:
            key = (s, a)
            # TaxiNet has exact transition probabilities
            T[key] = dict(ipomdp.T.get(key, {s: 1.0}))

    # Convert observation probabilities based on realization strategy
    P: Dict[State, Dict[Observation, float]] = {}

    for s in ipomdp.states:
        P[s] = {}
        P_lower = ipomdp.P_lower.get(s, {})
        P_upper = ipomdp.P_upper.get(s, {})

        # Ensure we have entries for all observations
        all_obs = set(P_lower.keys()) | set(P_upper.keys())

        for o in all_obs:
            p_low = P_lower.get(o, 0.0)
            p_high = P_upper.get(o, 1.0)

            if realization == "midpoint":
                P[s][o] = (p_low + p_high) / 2
            elif realization == "lower":
                P[s][o] = p_low
            elif realization == "upper":
                P[s][o] = p_high
            elif realization == "random":
                P[s][o] = random.uniform(p_low, p_high)
            else:
                raise ValueError(f"Unknown realization strategy: {realization}")

        # Normalize to ensure valid probability distribution
        total = sum(P[s].values())
        if total > 0:
            P[s] = {o: p / total for o, p in P[s].items()}
        else:
            # Fallback to uniform if somehow all zeros
            n_obs = len(all_obs)
            P[s] = {o: 1.0 / n_obs for o in all_obs}

    return POMDP(
        states=list(ipomdp.states),
        observations=list(ipomdp.observations),
        actions=list(ipomdp.actions),
        T=T,
        P=P
    )


def get_taxinet_avoid_states() -> FrozenSet[State]:
    """
    Get avoid states for TaxiNet (FAIL state and boundary states).

    Returns:
        Frozenset of states to avoid
    """
    # FAIL state is the main unsafe state
    avoid = {"FAIL"}
    return frozenset(avoid)


def get_taxinet_safe_states() -> FrozenSet[State]:
    """
    Get safe states for TaxiNet (all non-FAIL states).

    Returns:
        Frozenset of safe states
    """
    from .taxinet import taxinet_states

    safe_states = set(taxinet_states(with_fail=False))
    return frozenset(safe_states)


def get_taxinet_initial_support(ipomdp: IPOMDP) -> FrozenSet[State]:
    """
    Get initial support for TaxiNet (all safe states).

    For TaxiNet, we typically start with uncertain initial state,
    so the initial support includes all safe states.

    Args:
        ipomdp: The TaxiNet IPOMDP model

    Returns:
        Frozenset of initial support states
    """
    # Start with uniform belief over all safe states
    avoid = get_taxinet_avoid_states()
    initial_support = frozenset(s for s in ipomdp.states if s not in avoid)
    return initial_support
