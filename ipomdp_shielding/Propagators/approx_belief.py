"""Approximate belief propagation using split numerator/denominator."""

from typing import Dict, Tuple, Iterable, Set

from .belief_base import IPOMDP_Belief

State = type
Action = type
Observation = type


class IPOMDP_ApproxBelief(IPOMDP_Belief):
    """
    Split numerator/denominator approximation for IPOMDP belief tracking.

    Implements a pessimistic/optimistic approximation:
      lower_bound ~ N_min / D_max

    where:
    - N_min is a pessimistic (minimized) unnormalized mass in 'allowed'
    - D_max is an optimistic (maximized) total evidence mass
    """

    def __init__(self, ipomdp):
        super().__init__(ipomdp)
        self.restart()

    def restart(self):
        """Reset to uniform prior."""
        n_states = len(self.ipomdp.states)
        prior = {s: 1.0 / n_states for s in self.ipomdp.states}
        self.belief = (prior.copy(), prior.copy())

    def propogate(self, evidence: Tuple[Observation, Action]):
        """
        Propagate belief forward by evidence (observation, action).

        Uses pessimistic numerator and optimistic denominator for sound bounds.
        """
        o_t, a_t = evidence

        # Prediction step (exact dynamics)
        next_num = {s_next: 0.0 for s_next in self.ipomdp.states}
        next_den = {s_next: 0.0 for s_next in self.ipomdp.states}

        for s in self.ipomdp.states:
            trans = self.ipomdp.T[(s, a_t)]
            prob_s_num = self.belief[0][s]
            prob_s_den = self.belief[1][s]

            if prob_s_num == 0.0 and prob_s_den == 0.0:
                continue

            for s_next, p_trans in trans.items():
                if p_trans == 0.0:
                    continue
                next_num[s_next] += p_trans * prob_s_num
                next_den[s_next] += p_trans * prob_s_den

        # Observation update with interval likelihoods
        alpha_num = {}
        alpha_den = {}

        for s_next in self.ipomdp.states:
            z_low = self.ipomdp.P_lower[s_next].get(o_t, 0.0)
            z_up = self.ipomdp.P_upper[s_next].get(o_t, 0.0)

            # Pessimistic: use minimum likelihood for numerator
            alpha_num[s_next] = next_num[s_next] * z_low
            # Optimistic: use maximum likelihood for denominator
            alpha_den[s_next] = next_den[s_next] * z_up

        self.belief = (alpha_num, alpha_den)

    def allowed_probability(self, allowed: Iterable) -> float:
        """
        Return lower bound on probability mass in allowed states.
        """
        allowed_set: Set = set(allowed)
        alpha_num, alpha_den = self.belief

        N_min = sum(alpha_num[s] for s in allowed_set)
        D_max = sum(alpha_den[s] for s in self.ipomdp.states)

        if D_max <= 0.0:
            return 0.0

        return N_min / D_max
