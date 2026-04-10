"""Initial state sampling strategies for Monte Carlo safety evaluation.

This module provides strategies for sampling initial states to evaluate
best-case, worst-case, and average-case safety scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Tuple
import random

from ..Models.ipomdp import IPOMDP


class InitialStateGenerator(ABC):
    """Base class for initial state sampling strategies."""

    @abstractmethod
    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Generate initial (state, action) pair.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions

        Returns
        -------
        tuple
            (initial_state, initial_action)
        """
        pass


class RandomInitialState(InitialStateGenerator):
    """Average case: uniform random sampling from all states."""

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample uniformly from all states and actions."""
        state = random.choice(ipomdp.states)
        action = random.choice(list(ipomdp.actions))
        return state, action


class SafeInitialState(InitialStateGenerator):
    """Best case: sample from safe interior regions.

    Selects states with maximum number of safe actions (most flexibility).
    """

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample from states with most safe actions."""
        # Count safe actions per state
        state_safety = {}
        for state in ipomdp.states:
            if state == "FAIL":
                continue
            safe_actions = pp_shield.get(state, set())
            state_safety[state] = len(safe_actions)

        if not state_safety:
            # Fallback to random if no safe states
            return RandomInitialState().generate(ipomdp, pp_shield)

        # Find states with maximum safe actions
        max_safety = max(state_safety.values())
        safest_states = [s for s, count in state_safety.items() if count == max_safety]

        state = random.choice(safest_states)
        safe_actions = list(pp_shield.get(state, ipomdp.actions))
        action = random.choice(safe_actions) if safe_actions else random.choice(list(ipomdp.actions))

        return state, action


class BoundaryInitialState(InitialStateGenerator):
    """Worst case: sample near safety boundary.

    Selects states with minimum number of safe actions (most constrained).
    """

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample from states with fewest safe actions."""
        # Count safe actions per state
        state_safety = {}
        for state in ipomdp.states:
            if state == "FAIL":
                continue
            safe_actions = pp_shield.get(state, set())
            state_safety[state] = len(safe_actions)

        if not state_safety:
            # Fallback to random if no safe states
            return RandomInitialState().generate(ipomdp, pp_shield)

        # Find states with minimum safe actions (but > 0)
        min_safety = min(state_safety.values())
        boundary_states = [s for s, count in state_safety.items() if count == min_safety]

        state = random.choice(boundary_states)
        safe_actions = list(pp_shield.get(state, ipomdp.actions))
        action = random.choice(safe_actions) if safe_actions else random.choice(list(ipomdp.actions))

        return state, action
