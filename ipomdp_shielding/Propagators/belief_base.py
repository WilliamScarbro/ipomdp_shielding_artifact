"""Base class for IPOMDP belief propagators."""

from typing import Tuple, Iterable, Hashable

State = Hashable
Action = Hashable
Observation = Hashable


class IPOMDP_Belief:
    """
    Abstract base class for belief propagators in Interval POMDPs.

    Subclasses must implement:
    - restart(): Reset belief to initial state
    - propogate(evidence): Update belief with (observation, action) pair
    - allowed_probability(allowed): Compute probability mass in allowed states
    """

    def __init__(self, ipomdp):
        self.ipomdp = ipomdp

    def restart(self):
        """Reset belief to initial distribution."""
        pass

    def propogate(self, evidence: Tuple[Observation, Action]):
        """Update belief with observed evidence."""
        pass

    def minimum_allowed_probability(self, allowed: Iterable[State]) -> float:
        """Return (lower bound on) probability that state is in allowed set."""
        pass

    def maximum_disallowed_probability(self, disallowed: Iterable[State]) -> float:
        """Return (lower bound on) probability that state is in allowed set."""
        pass
