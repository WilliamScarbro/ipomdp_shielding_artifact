"""
Carr Shield: Support-based shielding for POMDPs.

Implements Carr et al.'s approach using the winning region of the support-MDP.
"""

from typing import List, Set, Tuple, Dict, FrozenSet
from ipomdp_shielding.Models.pomdp import POMDP, State, Action, Observation
from ipomdp_shielding.Propagators.belief_support_propagator import BeliefSupportPropagator
from ipomdp_shielding.Evaluation.support_mdp_builder import SupportMDPBuilder


class CarrShield:
    """
    Carr et al.'s support-based shield for POMDPs.

    Uses the winning region of the support-MDP to filter actions.
    An action is allowed if all possible next supports remain in the winning region.
    """

    def __init__(
        self,
        pomdp: POMDP,
        avoid_states: FrozenSet[State],
        initial_support: FrozenSet[State]
    ):
        """
        Initialize Carr shield.

        Args:
            pomdp: The POMDP model
            avoid_states: States to avoid (unsafe states)
            initial_support: Initial belief support
        """
        self.pomdp = pomdp
        self.avoid_states = frozenset(avoid_states)
        self.initial_support = frozenset(initial_support)

        # Build support propagator
        self.propagator = BeliefSupportPropagator(pomdp, initial_support)

        # Build support-MDP and compute winning region (offline)
        self.mdp_builder = SupportMDPBuilder(pomdp, avoid_states)
        self.mdp_builder.build_support_mdp(initial_support)
        self.mdp_builder.compute_winning_region()

        # Metrics
        self.stuck_count = 0
        self.empty_action_timesteps: List[int] = []
        self.timestep = 0

    def restart(self) -> None:
        """Reset shield to initial state."""
        self.propagator.restart()
        self.timestep = 0

    def initialize(self) -> None:
        """Initialize shield (for compatibility with RuntimeImpShield interface)."""
        self.restart()

    def next_actions(self, evidence: Tuple[Observation, Action]) -> List[Action]:
        """
        Filter actions to those keeping support in winning region.

        Args:
            evidence: Tuple of (observation, action) from previous timestep

        Returns:
            List of allowed actions (may be empty if stuck)
        """
        # Update support via propagator
        if evidence is not None:
            self.propagator.propogate(evidence)

        self.timestep += 1

        # Get current support
        current_support = self.propagator.get_support()

        # Get safe actions from support-MDP builder
        safe_actions = self.mdp_builder.get_safe_actions(current_support)

        # Track if stuck
        if not safe_actions:
            self.stuck_count += 1
            self.empty_action_timesteps.append(self.timestep)

        return list(safe_actions)

    def get_metrics(self) -> Dict[str, object]:
        """
        Get shield metrics.

        Returns:
            Dictionary with stuck count and empty action timesteps
        """
        return {
            "stuck_count": self.stuck_count,
            "empty_action_timesteps": self.empty_action_timesteps,
            "total_timesteps": self.timestep,
            "stuck_rate": self.stuck_count / self.timestep if self.timestep > 0 else 0.0,
        }

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about support-MDP and winning region.

        Returns:
            Dictionary with statistics
        """
        return self.mdp_builder.get_statistics()
