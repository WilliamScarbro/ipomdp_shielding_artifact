"""MinMax hypercube belief propagation for Interval HMM/IPOMDP."""

from typing import Dict, Tuple, Set, Iterable

from .belief_base import IPOMDP_Belief
from .utils import tightened_likelihood_bounds, transition_update

State = type
Action = type
Observation = type


class MinMaxIHMMBelief(IPOMDP_Belief):
    """
    Maintains hypercube of belief space.
    Max points: 2^|S|
    """

    def __init__(self, ipomdp):
        super().__init__(ipomdp)
        self.lower: Dict = {}
        self.upper: Dict = {}
        self.initialize_uniform()

    def initialize_uniform(self):
        """Initialize to uniform prior."""
        states = self.ipomdp.states
        for s in states:
            self.lower[s] = 1 / len(states)
            self.upper[s] = 1 / len(states)

    def restart(self):
        """Reset to uniform prior."""
        self.initialize_uniform()

    def enumerate_hypercube(self):
        """
        Returns points of hypercube from lower/upper.
        Collapses points when lower[s] ~= upper[s].
        """
        S = self.ipomdp.states
        epsilon = 1e-10
        disagree = []
        agree = []

        for s in S:
            if self.lower[s] + epsilon > self.upper[s]:
                agree.append(s)
            else:
                disagree.append(s)

        points = []

        for mask in range(1 << len(disagree)):
            point = {}
            for i in range(len(disagree)):
                s = disagree[i]
                p = self.lower[s] if (mask >> i) & 1 else self.upper[s]
                point[s] = p
            for s in agree:
                point[s] = self.lower[s]
            points.append(point)

        return points

    def _observation_extreme_points(
        self,
        b_pred: Dict,
        o_obs,
        lower: Dict,
        upper: Dict,
    ) -> Tuple[Dict, Dict]:
        """Compute observation update extremes and update lower/upper bounds."""
        S = self.ipomdp.states

        bounds = []
        for s in S:
            L_eff, U_eff = tightened_likelihood_bounds(self.ipomdp, s, o_obs)
            bounds.append((s, L_eff, U_eff))

        n = len(bounds)

        for mask in range(1 << n):
            unnorm = {}
            denom = 0.0
            for i, (s, L_eff, U_eff) in enumerate(bounds):
                z = U_eff if ((mask >> i) & 1) else L_eff
                val = float(b_pred.get(s, 0.0)) * z
                unnorm[s] = val
                denom += val

            if denom <= 0.0:
                continue

            inv = 1.0 / denom
            b_post = {s: unnorm[s] * inv for s in S}

            for s in S:
                if b_post[s] < lower[s]:
                    lower[s] = b_post[s]
                if b_post[s] > upper[s]:
                    upper[s] = b_post[s]

        return lower, upper

    def propogate(self, evidence: Tuple[Observation, Action]) -> None:
        """Update belief hypercube with evidence."""
        o_t, a_t = evidence

        S = self.ipomdp.states

        points = self.enumerate_hypercube()
        print("num points: ", len(points))

        new_lower = {s: 1 for s in S}
        new_upper = {s: 0 for s in S}

        for b in points:
            b_pred = transition_update(self.ipomdp, b, a_t)
            new_lower, new_upper = self._observation_extreme_points(
                b_pred, o_t, new_lower, new_upper
            )

        self.lower = new_lower
        self.upper = new_upper

    def allowed_probability(self, allowed: Iterable) -> float:
        """Return lower bound on probability in allowed states."""
        allowed_set: Set = set(allowed)
        return sum(float(self.lower.get(s, 0.0)) for s in allowed_set)
