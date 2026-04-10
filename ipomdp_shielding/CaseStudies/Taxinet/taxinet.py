"""TaxiNet case study: Ground Effect Estimation for autonomous taxiing."""

from typing import Dict, Tuple, List, Iterable, Hashable, Set, Optional
from collections import Counter
import random

from statsmodels.stats.proportion import proportion_confint

from ...Models import MDP, IMDP, IPOMDP, imdp_from_mdp, product_imdp, imdp_interval_width_dist
from ...Models.Confidence import ConfidenceInterval
from ...Propagators import IPOMDP_ApproxBelief, ExactIHMMBelief, MinMaxIHMMBelief
from ...Evaluation import RuntimeImpShield, evaluate_runtime_shield

from .data_loader import get_cte_data, get_he_data

State = Hashable
FAIL = "FAIL"


# State space bounds
HE_HIGH = 1   # Heading error range: -1 to 1
CTE_HIGH = 2  # Cross-track error range: -2 to 2


def taxinet_he_states() -> List[int]:
    """Return heading error state values."""
    return list(range(-HE_HIGH, HE_HIGH + 1))


def taxinet_cte_states() -> List[int]:
    """Return cross-track error state values."""
    return list(range(-CTE_HIGH, CTE_HIGH + 1))


def taxinet_states(with_fail: bool = False) -> List[State]:
    """Return all TaxiNet states as (cte, he) tuples."""
    states = [
        (cte, he)
        for cte in taxinet_cte_states()
        for he in taxinet_he_states()
    ]
    if with_fail:
        states.append(FAIL)
    return states


def taxinet_actions() -> Dict[State, List[int]]:
    """Return action mapping: state -> list of enabled actions."""
    states = taxinet_states()
    return {s: list(range(-1, 2)) for s in states}


def taxinet_next_state(state: State, action: int) -> Tuple[int, int]:
    """Compute next state from current state and action."""
    if state == FAIL:
        return state
    cte, he = state
    return cte + he + action, he + action


def taxinet_next_safe(state: State, action: int) -> State:
    """Compute next state, returning FAIL if unsafe."""
    new_state = taxinet_next_state(state, action)
    if not taxinet_safe(new_state):
        return FAIL
    return new_state


def taxinet_safe(state: State) -> bool:
    """Check if state is safe (within bounds)."""
    if state == FAIL:
        return False
    cte, he = state
    return abs(he) <= HE_HIGH and abs(cte) <= CTE_HIGH


def taxinet_dynamics_prob(error: float = 0.1) -> MDP:
    """
    Create MDP with stochastic dynamics.

    Parameters
    ----------
    error : float
        Probability of deviation from expected next state
    """
    states = taxinet_states(with_fail=False)
    actions = taxinet_actions()

    P = {}
    for cte, he in states:
        for a in actions[(cte, he)]:
            new_state = taxinet_next_state((cte, he), a)
            P[((cte, he), a)] = {s: 0 for s in states}
            P[((cte, he), a)][FAIL] = 0

            for i in [-1, 0, 1]:
                ns = (new_state[0] + i, new_state[1])
                prob = abs(i) * error + (1 - abs(i)) * (1 - 2 * error)
                if not taxinet_safe(ns):
                    P[((cte, he), a)][FAIL] = P[((cte, he), a)].get(FAIL, 0) + prob
                else:
                    P[((cte, he), a)][ns] = prob

    actions[FAIL] = [-1, 0, 1]
    P[(FAIL, -1)] = {FAIL: 1}
    P[(FAIL, 0)] = {FAIL: 1}
    P[(FAIL, 1)] = {FAIL: 1}

    states.append(FAIL)
    return MDP(states, actions, P)


def taxinet_dynamics() -> MDP:
    """Create MDP with deterministic dynamics."""
    states = taxinet_states(with_fail=False)
    actions = taxinet_actions()

    P = {}
    for cte, he in states:
        for a in actions[(cte, he)]:
            new_state = taxinet_next_state((cte, he), a)
            if not taxinet_safe(new_state):
                P[((cte, he), a)] = {FAIL: 1}
            else:
                P[((cte, he), a)] = {new_state: 1}

    actions[FAIL] = [-1, 0, 1]
    P[(FAIL, -1)] = {FAIL: 1}
    P[(FAIL, 0)] = {FAIL: 1}
    P[(FAIL, 1)] = {FAIL: 1}

    states.append(FAIL)
    return MDP(states, actions, P)


def _fill_imdp_coverage(
    imdp: IMDP,
    states: List[int],
    data: List[Tuple[int, int]],
    action: str,
    alpha: float
) -> None:
    """
    Fill missing state pairs in an IMDP with conservative zero-count bounds.

    For each (true_state, est_state) pair not present in the IMDP, adds
    lower=0 and upper=CI_upper(0, n_true) where n_true is the total number
    of observations for that true_state. This provides coverage without
    biasing the intervals for observed pairs.
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


def taxinet_perception(
    confidence_method: str,
    alpha: float,
    he_data: List[Tuple[int, int]],
    cte_data: List[Tuple[int, int]],
    smoothing: bool = True,
) -> IMDP:
    """
    Build IMDP perception model from observation data.

    Parameters
    ----------
    confidence_method : str
        Confidence interval method to use
    alpha : float
        Significance level
    he_data : list
        Heading error observation data
    cte_data : list
        Cross-track error observation data
    smoothing : bool
        Whether to apply coverage-only smoothing (default True). When enabled,
        computes CIs on raw data then fills in unobserved state pairs with
        conservative [0, CI_upper(0, n)] bounds.

    Returns
    -------
    IMDP
        Interval perception model
    """
    he_states = taxinet_he_states()
    cte_states = taxinet_cte_states()

    he_CI = ConfidenceInterval(he_data)
    cte_CI = ConfidenceInterval(cte_data)

    states = taxinet_states(with_fail=True)

    percieve_action = "PERC"
    he_imdp = he_CI.produce_imdp(he_states, percieve_action, confidence_method, alpha)
    cte_imdp = cte_CI.produce_imdp(cte_states, percieve_action, confidence_method, alpha)

    if smoothing:
        _fill_imdp_coverage(he_imdp, he_states, he_data, percieve_action, alpha)
        _fill_imdp_coverage(cte_imdp, cte_states, cte_data, percieve_action, alpha)

    perception_imdp = product_imdp(cte_imdp, he_imdp)

    perception_imdp.actions[FAIL] = [percieve_action]
    perception_imdp.P_lower[(FAIL, percieve_action)] = {FAIL: 1}
    perception_imdp.P_upper[(FAIL, percieve_action)] = {FAIL: 1}

    return perception_imdp


def build_taxinet_ipomdp(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    train_fraction: float = 0.8,
    error: float = 0.1,
    smoothing: bool = True,
    seed: Optional[int] = None
) -> Tuple[IPOMDP, Dict, List, List]:
    """
    Build complete TaxiNet IPOMDP model.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible train/test splits.

    Returns
    -------
    tuple
        (ipomdp, dynamic_shield, test_cte_model, test_he_model)
    """
    if seed is not None:
        random.seed(seed)

    cte_data = get_cte_data()
    he_data = get_he_data()

    cte_train_index = int(len(cte_data) * train_fraction)
    random.shuffle(cte_data)
    cte_train, cte_test = cte_data[:cte_train_index], cte_data[cte_train_index:]

    he_train_index = int(len(he_data) * train_fraction)
    random.shuffle(he_data)
    he_train, he_test = he_data[:he_train_index], he_data[he_train_index:]

    perc_imdp = taxinet_perception(confidence_method, alpha, he_train, cte_train,
                                    smoothing=smoothing)

    # Strip "PERC" action from perception IMDP to match IPOMDP format
    states = taxinet_states(with_fail=True)
    perc_L = {
        s: {s2: perc_imdp.P_lower[(s, "PERC")][s2] for s2 in perc_imdp.P_lower[(s, "PERC")]}
        for s in states
    }
    perc_U = {
        s: {s2: perc_imdp.P_upper[(s, "PERC")][s2] for s2 in perc_imdp.P_upper[(s, "PERC")]}
        for s in states
    }

    dyn_mdp = taxinet_dynamics_prob(error=error)
    actions = {-1, 0, 1}

    taxinet_ipomdp = IPOMDP(states, states, actions, dyn_mdp.P, perc_L, perc_U)

    # Build dynamic shield: state -> set of safe actions
    dyn_shield = {
        s: {a for a in actions if taxinet_safe(taxinet_next_state(s, a))}
        for s in states
    }

    # Build test models
    test_cte_model = {s: [s_est for s2, s_est in cte_test if s2 == s] for s in taxinet_cte_states()}
    test_he_model = {s: [s_est for s2, s_est in he_test if s2 == s] for s in taxinet_he_states()}

    return taxinet_ipomdp, dyn_shield, test_cte_model, test_he_model


def taxinet_evaluation():
    """Run evaluation of TaxiNet shielding with ExactIHMMBelief."""
    taxinet_ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    exact_belief = ExactIHMMBelief(taxinet_ipomdp)
    exact_rtips = RuntimeImpShield(dyn_shield, exact_belief, 0.5, 0)

    def perceptor(state):
        return (
            random.choice(test_cte_model[state[0]]),
            random.choice(test_he_model[state[1]])
        )

    print("Testing with ExactIHMMBelief")
    evaluate_runtime_shield(
        taxinet_ipomdp,
        perceptor,
        exact_rtips,
        trials=100,
        trial_length=20,
        start_state_action=((0, 0), 0)
    )


if __name__ == "__main__":
    taxinet_evaluation()
