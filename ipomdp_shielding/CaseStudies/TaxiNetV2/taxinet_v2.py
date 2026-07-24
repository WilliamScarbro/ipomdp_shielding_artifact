"""TaxiNetV2: TaxiNet dynamics with cp-control TaxiNet perception artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from statsmodels.stats.proportion import proportion_confint

from ...Models import IPOMDP, IMDP, MDP, product_imdp
from ...Models.Confidence import ConfidenceInterval
from ..Taxinet.taxinet import (
    FAIL,
    taxinet_actions,
    taxinet_next_state,
    taxinet_safe,
    taxinet_states,
)
from .data_loader import (
    DEFAULT_ALPHA_LEVEL,
    ConformalObservation,
    SignedTaxiState,
    get_taxinet_v2_conformal_data,
    get_taxinet_v2_single_axis_data,
    get_taxinet_v2_single_test_models,
)


BENCHMARK_SPEC = (
    "TaxiNetV2 uses the cp-control TaxiNet perception model and stochastic-action "
    "dynamics on the signed TaxiNet state space. Point-estimate artifacts back "
    "IPOMDP shields; conformal-set artifacts are reserved for conformal shielding."
)


def _fill_observation_coverage(
    imdp: IMDP,
    states: List[SignedTaxiState],
    observations: Sequence[Any],
    data: Sequence[Tuple[Any, Any]],
    action: str,
    alpha: float,
) -> None:
    """Add conservative zero-count upper bounds for unseen state/observation pairs."""
    totals = Counter(true_state for true_state, _ in data)

    for true_state in states:
        key = (true_state, action)
        n = totals.get(true_state, 0)
        if n == 0:
            imdp.P_lower[key] = {obs: 0.0 for obs in observations}
            imdp.P_upper[key] = {obs: 1.0 for obs in observations}
            continue

        if key not in imdp.P_lower:
            imdp.P_lower[key] = {}
            imdp.P_upper[key] = {}

        _, zero_upper = proportion_confint(0, n, alpha=alpha, method="beta")
        for obs in observations:
            if obs not in imdp.P_lower[key]:
                imdp.P_lower[key][obs] = 0.0
                imdp.P_upper[key][obs] = zero_upper


def _fill_axis_observation_coverage(
    imdp: IMDP,
    states: List[int],
    data: Sequence[Tuple[int, int]],
    action: str,
    alpha: float,
) -> None:
    """Add conservative zero-count upper bounds for point-estimate axis data."""
    totals = Counter(true_state for true_state, _ in data)

    for true_state in states:
        key = (true_state, action)
        n = totals.get(true_state, 0)
        if n == 0:
            imdp.P_lower[key] = {obs: 0.0 for obs in states}
            imdp.P_upper[key] = {obs: 1.0 for obs in states}
            continue

        if key not in imdp.P_lower:
            imdp.P_lower[key] = {}
            imdp.P_upper[key] = {}

        _, zero_upper = proportion_confint(0, n, alpha=alpha, method="beta")
        for obs in states:
            if obs not in imdp.P_lower[key]:
                imdp.P_lower[key][obs] = 0.0
                imdp.P_upper[key][obs] = zero_upper


def taxinet_v2_cp_control_dynamics(action_success: float = 0.9) -> MDP:
    """Return cp-control stochastic-action TaxiNet dynamics.

    cp-control first perturbs the selected action: the intended action is used
    with probability 0.9 by default, and each other action with probability 0.05.
    The resulting action is then applied to the deterministic TaxiNet dynamics.
    """
    if not 0.0 <= action_success <= 1.0:
        raise ValueError(f"action_success must be in [0, 1], got {action_success}")

    states = taxinet_states(with_fail=False)
    actions_by_state = taxinet_actions()
    actions = [-1, 0, 1]
    other_prob = (1.0 - action_success) / (len(actions) - 1)

    P = {}
    for state in states:
        for requested_action in actions_by_state[state]:
            row = {s: 0.0 for s in states}
            row[FAIL] = 0.0
            for applied_action in actions:
                prob = action_success if applied_action == requested_action else other_prob
                next_state = taxinet_next_state(state, applied_action)
                if taxinet_safe(next_state):
                    row[next_state] = row.get(next_state, 0.0) + prob
                else:
                    row[FAIL] = row.get(FAIL, 0.0) + prob
            P[(state, requested_action)] = {s: p for s, p in row.items() if p > 0.0}

    actions_by_state[FAIL] = actions
    for action in actions:
        P[(FAIL, action)] = {FAIL: 1.0}

    states_with_fail = states + [FAIL]
    return MDP(states_with_fail, actions_by_state, P)


def taxinet_v2_conformal_perception(
    confidence_method: str,
    alpha: float,
    observation_data: List[Tuple[SignedTaxiState, ConformalObservation]],
    smoothing: bool = True,
) -> Tuple[IMDP, List[ConformalObservation]]:
    """Build an IMDP over TaxiNetV2 conformal-set observations."""
    states = taxinet_states(with_fail=True)
    observations = sorted({obs for _state, obs in observation_data})
    perceive_action = "PERC"

    obs_ci = ConfidenceInterval(observation_data)
    obs_imdp = obs_ci.produce_imdp(states[:-1], perceive_action, confidence_method, alpha)

    if smoothing:
        _fill_observation_coverage(
            obs_imdp,
            states[:-1],
            observations,
            observation_data,
            perceive_action,
            alpha,
        )

    obs_imdp.actions[FAIL] = [perceive_action]
    obs_imdp.P_lower[(FAIL, perceive_action)] = {FAIL: 1.0}
    obs_imdp.P_upper[(FAIL, perceive_action)] = {FAIL: 1.0}

    return obs_imdp, observations + [FAIL]


def taxinet_v2_single_estimate_perception(
    confidence_method: str,
    alpha: float,
    cte_data: List[Tuple[int, int]],
    he_data: List[Tuple[int, int]],
    smoothing: bool = True,
) -> IMDP:
    """Build an IMDP over TaxiNetV2 point-estimate observations."""
    cte_states = [-2, -1, 0, 1, 2]
    he_states = [-1, 0, 1]
    perceive_action = "PERC"

    cte_ci = ConfidenceInterval(cte_data)
    he_ci = ConfidenceInterval(he_data)
    cte_imdp = cte_ci.produce_imdp(cte_states, perceive_action, confidence_method, alpha)
    he_imdp = he_ci.produce_imdp(he_states, perceive_action, confidence_method, alpha)

    if smoothing:
        _fill_axis_observation_coverage(cte_imdp, cte_states, cte_data, perceive_action, alpha)
        _fill_axis_observation_coverage(he_imdp, he_states, he_data, perceive_action, alpha)

    perception_imdp = product_imdp(cte_imdp, he_imdp)
    perception_imdp.actions[FAIL] = [perceive_action]
    perception_imdp.P_lower[(FAIL, perceive_action)] = {FAIL: 1.0}
    perception_imdp.P_upper[(FAIL, perceive_action)] = {FAIL: 1.0}
    return perception_imdp


taxinet_v2_perception = taxinet_v2_conformal_perception


def _build_dyn_shield(states: Sequence[Hashable], actions: Sequence[int]) -> Dict[Hashable, set]:
    return {
        state: {action for action in actions if taxinet_safe(taxinet_next_state(state, action))}
        for state in states
    }


def build_taxinet_v2_single_estimate_ipomdp(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
    error: Optional[float] = None,
    action_success: float = 0.9,
    smoothing: bool = True,
    seed: Optional[int] = None,
) -> Tuple[IPOMDP, Dict[Hashable, set], Dict[int, List[int]], Dict[int, List[int]]]:
    """Build TaxiNetV2 from cp-control point-estimate observation samples.

    This is the fair-comparison builder for envelope, forward-sampling, and
    single-belief shielding. It does not consume conformal-set observations.
    """
    del confidence_level, error, seed

    cte_data, he_data = get_taxinet_v2_single_axis_data()
    perception_imdp = taxinet_v2_single_estimate_perception(
        confidence_method=confidence_method,
        alpha=alpha,
        cte_data=cte_data,
        he_data=he_data,
        smoothing=smoothing,
    )
    observations = [(cte_obs, he_obs) for cte_obs in [-2, -1, 0, 1, 2] for he_obs in [-1, 0, 1]]
    observations.append(FAIL)

    states = taxinet_states(with_fail=True)
    perc_lower = {
        state: dict(perception_imdp.P_lower[(state, "PERC")])
        for state in states
    }
    perc_upper = {
        state: dict(perception_imdp.P_upper[(state, "PERC")])
        for state in states
    }

    dyn_mdp = taxinet_v2_cp_control_dynamics(action_success=action_success)
    actions = [-1, 0, 1]
    ipomdp = IPOMDP(states, observations, actions, dyn_mdp.P, perc_lower, perc_upper)

    dyn_shield = _build_dyn_shield(states, actions)

    test_cte_model, test_he_model = get_taxinet_v2_single_test_models()
    return ipomdp, dyn_shield, test_cte_model, test_he_model


def build_taxinet_v2_conformal_ipomdp(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
    error: Optional[float] = None,
    action_success: float = 0.9,
    smoothing: bool = True,
    seed: Optional[int] = None,
) -> Tuple[IPOMDP, Dict[Hashable, set], Dict[int, List[int]], Dict[int, List[int]]]:
    """Build the point-estimate IPOMDP used by TaxiNetV2 conformal evaluation.

    Conformal TaxiNetV2 semantics are two-stage:
    1. an estimate is sampled from ``P(estimate | true_state)``
    2. a conformal set is sampled from empirical
       ``P(set | true_state, estimate)``

    The IPOMDP therefore remains the point-estimate observation model; the
    conformal sets are not encoded as direct IPOMDP observations.
    """
    del confidence_level
    return build_taxinet_v2_single_estimate_ipomdp(
        confidence_method=confidence_method,
        alpha=alpha,
        error=error,
        action_success=action_success,
        smoothing=smoothing,
        seed=seed,
    )


build_taxinet_v2_ipomdp = build_taxinet_v2_single_estimate_ipomdp
