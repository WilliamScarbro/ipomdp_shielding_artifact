"""Exact (extreme-point) belief propagation for Interval HMM/IPOMDP."""

from typing import Dict, List, Tuple, Set, Optional, Iterable

from .belief_base import IPOMDP_Belief
from .utils import tightened_likelihood_bounds, transition_update

State = type
Action = type
Observation = type


class ExactIHMMBelief(IPOMDP_Belief):
    """
    Exact (extreme-point) forward inference for an iHMM / IPOMDP with:
      - exact dynamics T
      - interval perception Z(o | s) in [P_lower, P_upper]

    Representation:
      self.points is a list of belief distributions b over states (dict s->p),
      representing the set of feasible posteriors as the convex hull of extreme points.

    WARNING:
      The number of points can grow exponentially in |S| per step.
      Use max_points / rounding to keep it manageable online.
    """

    def __init__(
        self,
        ipomdp,
        *,
        max_points: Optional[int] = 5000,
        round_decimals: int = 10,
        auto_prune: bool = False,
    ):
        super().__init__(ipomdp)
        self.max_points = max_points
        self.round_decimals = round_decimals
        self.auto_prune = auto_prune
        self.points: List[Dict] = []
        self.initialize_uniform()

    def initialize_uniform(self) -> None:
        """Initialize credal set to a single uniform prior."""
        S = self.ipomdp.states
        if not S:
            self.points = []
            return
        p = 1.0 / len(S)
        self.points = [{s: p for s in S}]

    def restart(self):
        """Reset to uniform prior."""
        self.initialize_uniform()

    def _observation_extreme_points(
        self,
        b_pred: Dict,
        o_obs,
    ) -> List[Dict]:
        """
        Given predicted belief b_pred and observed symbol o_obs, produce all extreme-point
        posteriors for:
          b_post(s) proportional to b_pred(s) * z_s
        where z_s in [L_eff(s), U_eff(s)] independently per state s.
        """
        S = self.ipomdp.states

        if self.auto_prune:
            mins = {s: 1 for s in S}
            maxes = {s: 0 for s in S}

        bounds = []
        for s in self.ipomdp.states:
            L_eff, U_eff = tightened_likelihood_bounds(self.ipomdp, s, o_obs)
            bounds.append((s, L_eff, U_eff))

        n = len(bounds)
        out: List[Dict] = []

        # Enumerate all corners of the box z_s in {L_eff, U_eff}
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

            if self.auto_prune:
                add_flag = False
                for s in S:
                    if b_post[s] < mins[s]:
                        add_flag = True
                        mins[s] = b_post[s]
                    if b_post[s] > maxes[s]:
                        add_flag = True
                        maxes[s] = b_post[s]
                if add_flag:
                    out.append(b_post)
            else:
                out.append(b_post)

        return out

    def _dedup(self, points: List[Dict]) -> List[Dict]:
        """Deduplicate belief points by rounding."""
        S = self.ipomdp.states
        seen = {}
        for b in points:
            key = tuple(round(float(b.get(s, 0.0)), self.round_decimals) for s in S)
            seen[key] = b
        return list(seen.values())

    def _prune_if_needed(self, points: List[Dict]) -> List[Dict]:
        """
        Prune points if max_points exceeded.
        Keeps points that are extreme w.r.t. each state coordinate.
        """
        if self.max_points is None or len(points) <= self.max_points:
            return points

        S = self.ipomdp.states
        keep: List[Dict] = []

        for s in S:
            b_min = min(points, key=lambda b: float(b.get(s, 0.0)))
            b_max = max(points, key=lambda b: float(b.get(s, 0.0)))
            keep.append(b_min)
            keep.append(b_max)

        keep = self._dedup(keep)

        if self.max_points < len(keep):
            print("Cannot prune to within max points without loss of validity")

        return keep

    def propogate(self, evidence: Tuple[Observation, Action]) -> None:
        """
        Update the belief envelope by one (observation, action) pair.
        """
        o_t, a_t = evidence
        if not self.points:
            return

        new_points: List[Dict] = []

        for b in self.points:
            b_pred = transition_update(self.ipomdp, b, a_t)
            exts = self._observation_extreme_points(b_pred, o_t)
            new_points.extend(exts)

        print("new points before dedup/pruning: ", len(new_points))
        new_points = self._dedup(new_points)
        print("new points after dedup: ", len(new_points))
        self.points = self._prune_if_needed(new_points)
        print("points after pruning", len(self.points))

        # Debug: show probable states
        probable = []
        for s in self.ipomdp.states:
            bmax = max(self.points, key=lambda b: float(b.get(s, 0.0)))
            mprob = bmax[s]
            if mprob > 0.1:
                probable.append(s)
        print("probable", probable)

    def allowed_probability(self, allowed: Iterable) -> float:
        """
        Return the exact lower probability mass in `allowed` among all beliefs.
        """
        allowed_set: Set = set(allowed)
        if not self.points:
            return 0.0

        def mass(b: Dict) -> float:
            return sum(float(b.get(s, 0.0)) for s in allowed_set)

        return min(mass(b) for b in self.points)
