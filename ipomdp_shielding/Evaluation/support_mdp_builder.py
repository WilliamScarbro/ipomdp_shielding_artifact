"""
Support-MDP Builder for Carr et al.'s support-based shielding.

Constructs subset-construction MDP where states are support sets,
and computes the winning region (supports from which safety can be maintained).
"""

from typing import Dict, FrozenSet, Set, Tuple, Hashable
from collections import deque
from ipomdp_shielding.Models.pomdp import POMDP, State, Action, Observation


class SupportMDPBuilder:
    """
    Builds MDP over support sets and computes winning region.

    The support-MDP has:
    - States: Support sets (frozensets of POMDP states)
    - Actions: Same as POMDP actions
    - Transitions: B --a--> B' if ∃o such that B' = update(B, a, o)
    """

    def __init__(self, pomdp: POMDP, avoid_states: FrozenSet[State]):
        """
        Initialize support-MDP builder.

        Args:
            pomdp: The POMDP model
            avoid_states: States to avoid (unsafe states)
        """
        self.pomdp = pomdp
        self.avoid_states = frozenset(avoid_states)
        self.support_mdp: Dict[FrozenSet[State], Dict[Action, Set[FrozenSet[State]]]] = {}
        self.winning_region: Set[FrozenSet[State]] = set()

    def _compute_next_support(self, support: FrozenSet[State], action: Action, obs: Observation) -> FrozenSet[State]:
        """
        Compute next support after taking action and observing obs.

        Args:
            support: Current support
            action: Action taken
            obs: Observation received

        Returns:
            Updated support
        """
        # Prediction: states reachable from current support
        predicted: Set[State] = set()
        for s in support:
            trans = self.pomdp.T.get((s, action), {})
            for s_next, prob in trans.items():
                if prob > 0:
                    predicted.add(s_next)

        # Observation update: states consistent with observation
        updated: Set[State] = set()
        for s in predicted:
            obs_probs = self.pomdp.P.get(s, {})
            if obs_probs.get(obs, 0.0) > 0:
                updated.add(s)

        return frozenset(updated)

    def build_support_mdp(self, initial_support: FrozenSet[State]) -> None:
        """
        Build support-MDP via BFS from initial support.

        Explores all reachable support sets from initial support.
        For each (support, action), finds all possible next supports
        via all possible observations.

        Args:
            initial_support: Initial belief support
        """
        self.support_mdp = {}
        visited: Set[FrozenSet[State]] = set()
        queue: deque[FrozenSet[State]] = deque([initial_support])
        visited.add(initial_support)

        while queue:
            support = queue.popleft()
            self.support_mdp[support] = {}

            # For each action, compute all possible next supports
            for action in self.pomdp.actions:
                next_supports: Set[FrozenSet[State]] = set()

                # Try all possible observations
                for obs in self.pomdp.observations:
                    next_support = self._compute_next_support(support, action, obs)

                    # Only add non-empty supports
                    if next_support:
                        next_supports.add(next_support)

                        # Add to BFS queue if not visited
                        if next_support not in visited:
                            visited.add(next_support)
                            queue.append(next_support)

                self.support_mdp[support][action] = next_supports

    def compute_winning_region(self) -> None:
        """
        Compute winning region via fixed-point iteration.

        A support is winning if:
        1. It contains no avoid states, AND
        2. There exists an action where all possible next supports are winning

        Algorithm:
        - Initialize: winning = {supports with no avoid states}
        - Fixed-point: remove supports where no action keeps you in winning
        """
        # Initialize: supports with no avoid states
        self.winning_region = {
            support for support in self.support_mdp.keys()
            if support.isdisjoint(self.avoid_states)
        }

        # Fixed-point iteration
        changed = True
        while changed:
            changed = False
            to_remove: Set[FrozenSet[int]] = set()

            for support in self.winning_region:
                # Check if there exists a safe action
                has_safe_action = False
                for action in self.pomdp.actions:
                    next_supports = self.support_mdp[support][action]

                    # Action is safe if all next supports are winning
                    if next_supports and all(ns in self.winning_region for ns in next_supports):
                        has_safe_action = True
                        break

                # If no safe action exists, remove from winning region
                if not has_safe_action:
                    to_remove.add(support)
                    changed = True

            self.winning_region -= to_remove

    def is_winning_support(self, support: FrozenSet[State]) -> bool:
        """
        Check if support is in winning region.

        Args:
            support: Support set to check

        Returns:
            True if support is winning, False otherwise
        """
        return support in self.winning_region

    def get_safe_actions(self, support: FrozenSet[State]) -> Set[Action]:
        """
        Get actions that keep support in winning region.

        Args:
            support: Current support

        Returns:
            Set of safe actions (may be empty if stuck)
        """
        if support not in self.winning_region:
            return set()

        safe_actions: Set[Action] = set()
        for action in self.pomdp.actions:
            next_supports = self.support_mdp.get(support, {}).get(action, set())

            # Action is safe if all next supports are winning
            if next_supports and all(ns in self.winning_region for ns in next_supports):
                safe_actions.add(action)

        return safe_actions

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about support-MDP and winning region.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_supports": len(self.support_mdp),
            "winning_supports": len(self.winning_region),
            "losing_supports": len(self.support_mdp) - len(self.winning_region),
            "pomdp_states": len(self.pomdp.states),
        }
