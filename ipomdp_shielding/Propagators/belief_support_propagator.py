"""
Belief Support Propagator for Carr et al.'s support-based shielding.

Tracks the support of the belief (set of states with nonzero probability)
using probability-free graph reachability.
"""

from typing import FrozenSet, Set, Tuple, Iterable
from ipomdp_shielding.Propagators.belief_base import IPOMDP_Belief, State, Action, Observation
from ipomdp_shielding.Models.pomdp import POMDP


class BeliefSupportPropagator(IPOMDP_Belief):
    """
    Propagates belief support (set of states with nonzero probability).

    Uses graph reachability instead of probability calculations:
    - Prediction: support' = {s' : ∃s ∈ support with T(s'|s,a) > 0}
    - Observation: support'' = {s' ∈ support' : P(obs|s') > 0}

    This is more conservative than exact Bayesian belief tracking because
    it ignores probability magnitudes and only tracks possibility.
    """

    def __init__(self, pomdp: POMDP, initial_support: FrozenSet[State]):
        """
        Initialize support propagator.

        Args:
            pomdp: The POMDP model
            initial_support: Initial belief support (states with nonzero probability)
        """
        super().__init__(pomdp)  # Call parent constructor
        self.pomdp = pomdp
        self.support = frozenset(initial_support)
        self.initial_support = frozenset(initial_support)

    def restart(self) -> None:
        """Reset support to initial support."""
        self.support = self.initial_support

    def propogate(self, evidence: Tuple[Observation, Action]) -> None:
        """
        Update support through (observation, action) pair.

        Args:
            evidence: Tuple of (observation, action)
        """
        obs, action = evidence

        # Prediction step: find all states reachable from current support
        predicted_support: Set[State] = set()
        for s in self.support:
            trans = self.pomdp.T.get((s, action), {})
            for s_next, prob in trans.items():
                if prob > 0:
                    predicted_support.add(s_next)

        # Observation update: keep only states consistent with observation
        updated_support: Set[State] = set()
        for s in predicted_support:
            obs_probs = self.pomdp.P.get(s, {})
            if obs_probs.get(obs, 0.0) > 0:
                updated_support.add(s)

        self.support = frozenset(updated_support)

    def get_support(self) -> FrozenSet[State]:
        """Return current support as frozenset."""
        return self.support

    def support_size(self) -> int:
        """Return size of current support."""
        return len(self.support)

    def minimum_allowed_probability(self, allowed: Iterable[State]) -> float:
        """
        Return 1.0 if support ⊆ allowed, else 0.0.

        Args:
            allowed: Set of allowed states

        Returns:
            1.0 if all states in support are allowed, 0.0 otherwise
        """
        allowed_set = set(allowed)
        return 1.0 if self.support.issubset(allowed_set) else 0.0

    def maximum_disallowed_probability(self, disallowed: Iterable[State]) -> float:
        """
        Return 0.0 if support ∩ disallowed = ∅, else 1.0.

        Args:
            disallowed: Set of disallowed states

        Returns:
            0.0 if no states in support are disallowed, 1.0 otherwise
        """
        disallowed_set = set(disallowed)
        return 0.0 if self.support.isdisjoint(disallowed_set) else 1.0

    def clone(self) -> 'BeliefSupportPropagator':
        """Create a copy of this propagator with the same current support."""
        clone = BeliefSupportPropagator(self.pomdp, self.initial_support)
        clone.support = self.support
        return clone
