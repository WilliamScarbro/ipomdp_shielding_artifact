"""Action selector strategies for Monte Carlo safety evaluation.

This module provides various strategies for selecting actions from the shield's
allowed action set. Strategies range from simple random selection to RL-trained
policies that maximize or minimize safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random

from ..Models.pomdp import POMDP, POMDP_Belief


# ============================================================
# Action Selector Base Classes
# ============================================================

class ActionSelector(ABC):
    """Base class for action selection strategies.

    Type signature: Callable[[List[Tuple[obs, action]], List[action]], action]

    The history parameter provides full observation-action sequence for RL policies.
    The allowed_actions ensure safety (pre-filtered by shield).
    """

    @abstractmethod
    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select an action from allowed actions given history.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence up to current step
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Additional context for selection. May contain:
            - "rt_shield": RuntimeImpShield instance
            - "history": Reference to history list

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        pass


# ============================================================
# Simple Action Selectors
# ============================================================

class RandomActionSelector(ActionSelector):
    """Baseline random selection from allowed actions."""

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Randomly select from allowed actions."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")
        return random.choice(allowed_actions)


class UniformFallbackSelector(ActionSelector):
    """Random selection with fallback to all actions if stuck.

    If no allowed actions are provided (shield stuck), falls back to
    selecting uniformly from all available actions.
    """

    def __init__(self, all_actions: List[Any]):
        """Initialize with complete action set for fallback.

        Parameters
        ----------
        all_actions : list
            Complete set of actions in the IPOMDP
        """
        self.all_actions = all_actions

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select from allowed actions, or all actions if empty."""
        if allowed_actions:
            return random.choice(allowed_actions)
        else:
            return random.choice(self.all_actions)


class BeliefSelector(ActionSelector):
    """Selects actions based on runtime shield belief probabilities.

    Uses RuntimeImpShield.get_action_probs() to access the current belief state
    and select actions based on their safety probability.

    Modes:
    - "best": Select action with highest allowed probability (safety-maximizing)
    - "worst": Select action with lowest allowed probability (stress testing)
    """

    def __init__(self, mode: str = "best", exploration_rate: float = 0.0):
        """Initialize belief-based selector.

        Parameters
        ----------
        mode : str
            Selection mode: "best" or "worst"
        exploration_rate : float
            Probability of random selection (0.0 to 1.0)
        """
        if mode not in ["best", "worst"]:
            raise ValueError(f"mode must be 'best' or 'worst', got {mode}")

        self.mode = mode
        self.exploration_rate = exploration_rate

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using shield belief probabilities.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Must contain "rt_shield": RuntimeImpShield instance

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Get shield from context
        if context is None or "rt_shield" not in context:
            # Fallback to random if no shield available
            return random.choice(allowed_actions)

        rt_shield = context["rt_shield"]

        # Get action probabilities from shield
        action_probs = rt_shield.get_action_probs()

        # Build map: action -> (allowed_prob, disallowed_prob)
        prob_map = {
            action: (allowed_prob, disallowed_prob)
            for action, allowed_prob, disallowed_prob in action_probs
        }

        # Filter to only allowed actions and score them
        action_scores = []
        for action in allowed_actions:
            if action not in prob_map:
                continue

            allowed_prob, _ = prob_map[action]
            action_scores.append((action, allowed_prob))

        if not action_scores:
            return random.choice(allowed_actions)

        # Select based on mode
        if self.mode == "best":
            # Highest allowed probability
            action_scores.sort(key=lambda x: x[1], reverse=True)
        else:  # mode == "worst"
            # Lowest allowed probability
            action_scores.sort(key=lambda x: x[1])

        return action_scores[0][0]


class SingleBeliefSelector(ActionSelector):
    """Selects actions based on a single POMDP belief estimate of shield allowance.

    Unlike BeliefSelector, which queries the full interval-POMDP belief envelope
    via rt_shield.get_action_probs(), this selector maintains a single point
    belief over states using a standard POMDP and estimates how likely each
    action is to be allowed by a perfect perception shield.

    For each candidate action a, computes:
        P(a allowed) = sum_{s : a in pp_shield[s]} belief(s)

    Modes:
    - "best":  Select the allowed action with the highest allowance probability
    - "worst": Select the allowed action with the lowest allowance probability
    """

    def __init__(
        self,
        pomdp: POMDP,
        pp_shield: Dict[Any, Set[Any]],
        mode: str = "best",
        exploration_rate: float = 0.0,
    ):
        """Initialize single-belief selector.

        Parameters
        ----------
        pomdp : POMDP
            Standard POMDP used for belief propagation
        pp_shield : dict mapping state -> set of allowed actions
            Pre-computed perfect-perception shield
        mode : str
            "best" (safety-maximizing) or "worst" (stress testing)
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        if mode not in ("best", "worst"):
            raise ValueError(f"mode must be 'best' or 'worst', got {mode}")

        self.pomdp = pomdp
        self.pp_shield = pp_shield
        self.mode = mode
        self.exploration_rate = exploration_rate

        self._belief = POMDP_Belief(pomdp)
        self._history_len = 0  # how far into the history we have propagated

    def _allowance_probability(self, action: Any) -> float:
        """Estimate probability that *action* is allowed under current belief.

        Returns sum of belief mass over states where pp_shield allows *action*.
        """
        return sum(
            self._belief.belief.get(s, 0.0)
            for s, allowed_set in self.pp_shield.items()
            if action in allowed_set
        )

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Select action using single-belief allowance estimates.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence up to current step
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Not used (included for interface compatibility)

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Propagate belief for any new history entries
        while self._history_len < len(history):
            self._belief.propogate(history[self._history_len])
            self._history_len += 1

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Score each allowed action by P(action allowed | belief)
        action_scores = [
            (action, self._allowance_probability(action))
            for action in allowed_actions
        ]

        if self.mode == "best":
            action_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            action_scores.sort(key=lambda x: x[1])

        return action_scores[0][0]

    def reset(self):
        """Reset belief to uniform prior and clear history tracking."""
        self._belief.restart()
        self._history_len = 0


class ShieldedActionSelector(ActionSelector):
    """Combines a primary action selector with shield-based safety fallback.

    Selection logic:
    1. Get the primary selector's preferred action (over all actions)
    2. If that action is in the shield's allowed set, use it
    3. If not, fall back to _select_fallback (subclass-defined)
    4. If the shield returns no allowed actions, use the primary choice unrestricted

    Subclasses must implement _select_fallback to define how a safe fallback
    action is chosen when the primary selector's choice is disallowed.
    """

    def __init__(
        self,
        selector: ActionSelector,
        all_actions: List[Any],
        exploration_rate: float = 0.0
    ):
        """Initialize shielded action selector.

        Parameters
        ----------
        selector : ActionSelector
            Primary action selector (e.g. RL policy, heuristic)
        all_actions : list
            Complete set of actions in the IPOMDP
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        self.selector = selector
        self.all_actions = all_actions
        self.exploration_rate = exploration_rate

        # Selection statistics
        self.primary_selections = 0
        self.fallback_selections = 0
        self.unshielded_selections = 0

    @abstractmethod
    def _select_fallback(
        self,
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Select a fallback action when the primary choice is disallowed.

        Parameters
        ----------
        allowed_actions : list
            Non-empty list of actions permitted by the shield
        context : dict, optional
            Additional context (e.g. runtime shield instance)

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        pass

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using primary selector with shield-based fallback.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Additional context for selection

        Returns
        -------
        action
            Selected action
        """
        if not allowed_actions:
            self.unshielded_selections += 1
            return self.selector.select(history, self.all_actions, context)

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Get primary selector's preferred action over all actions
        preferred = self.selector.select(history, self.all_actions, context)

        if preferred in allowed_actions:
            self.primary_selections += 1
            return preferred

        # Primary action not allowed — delegate to subclass fallback
        self.fallback_selections += 1
        return self._select_fallback(allowed_actions, context)

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics on action selection behavior."""
        total = (self.primary_selections + self.fallback_selections
                 + self.unshielded_selections)
        shielded = self.primary_selections + self.fallback_selections
        acceptance_rate = (self.primary_selections / shielded
                         if shielded > 0 else 0.0)

        return {
            'primary_selections': self.primary_selections,
            'fallback_selections': self.fallback_selections,
            'unshielded_selections': self.unshielded_selections,
            'total_selections': total,
            'primary_acceptance_rate': acceptance_rate,
        }

    def reset_stats(self):
        """Reset selection statistics."""
        self.primary_selections = 0
        self.fallback_selections = 0
        self.unshielded_selections = 0


class BeliefShieldedActionSelector(ShieldedActionSelector):
    """Combines an RL policy with belief-envelope safety from the runtime shield.

    Selection logic (inherited from ShieldedActionSelector):
    1. Get the RL selector's preferred action (over all actions)
    2. If that action is in the shield's allowed set, use it
    3. If not, fall back to the allowed action with the highest allowed
       probability from the belief envelope (same metric as BeliefSelector)
    4. If the shield returns no allowed actions, use the RL choice unrestricted
    """

    def __init__(
        self,
        rl_selector: ActionSelector,
        all_actions: List[Any],
        exploration_rate: float = 0.0
    ):
        """Initialize belief-shielded action selector.

        Parameters
        ----------
        rl_selector : ActionSelector
            Trained RL-based action selector (e.g. NeuralActionSelector)
        all_actions : list
            Complete set of actions in the IPOMDP
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        super().__init__(rl_selector, all_actions, exploration_rate)

    @property
    def rl_selector(self) -> ActionSelector:
        """Alias for the primary selector (backwards compatibility)."""
        return self.selector

    @property
    def rl_selections(self) -> int:
        """Alias for primary_selections (backwards compatibility)."""
        return self.primary_selections

    def _select_fallback(
        self,
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Select allowed action with highest belief-envelope allowed probability.

        Uses rt_shield.get_action_probs() — the same safety metric as
        BeliefSelector in "best" mode.
        """
        if context is None or "rt_shield" not in context:
            return random.choice(allowed_actions)

        rt_shield = context["rt_shield"]
        action_probs = rt_shield.get_action_probs()

        prob_map = {
            action: allowed_prob
            for action, allowed_prob, _ in action_probs
        }

        action_scores = []
        for action in allowed_actions:
            if action in prob_map:
                action_scores.append((action, prob_map[action]))

        if not action_scores:
            return random.choice(allowed_actions)

        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics on action selection behavior.

        Returns stats with both generic and RL-specific key names
        for backwards compatibility.
        """
        stats = super().get_selection_stats()
        stats['rl_selections'] = stats['primary_selections']
        stats['rl_acceptance_rate'] = stats['primary_acceptance_rate']
        return stats


class ObservationShieldedActionSelector(ShieldedActionSelector):
    """Combines a primary selector with direct perfect-perception shield lookup.

    When the observation space equals the state space (observations are state
    estimates), the perfect perception shield can be applied directly:
    pp_shield[obs] gives the safe actions for the current observed state.

    Selection logic:
    1. Narrow allowed_actions to those also in pp_shield[observation]
    2. Get primary selector's preferred action (over all actions)
    3. If it's in the narrowed set, use it
    4. Otherwise, pick randomly from the narrowed set
    5. If the narrowed set is empty, fall back to the original allowed_actions
    6. If allowed_actions is empty, use the primary choice unrestricted
    """

    def __init__(
        self,
        selector: ActionSelector,
        all_actions: List[Any],
        pp_shield: Dict[Any, Set[Any]],
        exploration_rate: float = 0.0,
    ):
        """Initialize observation-shielded action selector.

        Parameters
        ----------
        selector : ActionSelector
            Primary action selector (e.g. RL policy, heuristic)
        all_actions : list
            Complete set of actions in the IPOMDP
        pp_shield : dict mapping state -> set of allowed actions
            Pre-computed perfect-perception shield
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        super().__init__(selector, all_actions, exploration_rate)
        self.pp_shield = pp_shield

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Narrow allowed actions via pp_shield[obs], then delegate to parent."""
        if history:
            obs = history[-1][0]
            pp_allowed = self.pp_shield.get(obs, set())
            obs_filtered = [a for a in allowed_actions if a in pp_allowed]
        else:
            obs_filtered = list(allowed_actions)

        effective_allowed = obs_filtered if obs_filtered else allowed_actions
        return super().select(history, effective_allowed, context)

    def _select_fallback(
        self,
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Select randomly from (already narrowed) allowed actions."""
        return random.choice(allowed_actions)


class SingleBeliefShieldedActionSelector(ShieldedActionSelector):
    """Combines a primary selector with single-POMDP-belief safety fallback.

    When the primary selector's choice is disallowed by the shield, falls back
    to the allowed action with the highest estimated allowance probability
    under a single point belief (same metric as SingleBeliefSelector).

    Unlike BeliefShieldedActionSelector, which queries the runtime shield's
    interval-POMDP belief envelope, this class maintains its own standard
    POMDP belief and scores actions by:
        P(a allowed) = sum_{s : a in pp_shield[s]} belief(s)
    """

    def __init__(
        self,
        selector: ActionSelector,
        all_actions: List[Any],
        pomdp: POMDP,
        pp_shield: Dict[Any, Set[Any]],
        exploration_rate: float = 0.0,
    ):
        """Initialize single-belief shielded action selector.

        Parameters
        ----------
        selector : ActionSelector
            Primary action selector (e.g. RL policy, heuristic)
        all_actions : list
            Complete set of actions in the IPOMDP
        pomdp : POMDP
            Standard POMDP used for belief propagation
        pp_shield : dict mapping state -> set of allowed actions
            Pre-computed perfect-perception shield
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        super().__init__(selector, all_actions, exploration_rate)
        self.pomdp = pomdp
        self.pp_shield = pp_shield
        self._belief = POMDP_Belief(pomdp)
        self._history_len = 0

    def _allowance_probability(self, action: Any) -> float:
        """Estimate probability that *action* is allowed under current belief.

        Returns sum of belief mass over states where pp_shield allows *action*.
        """
        return sum(
            self._belief.belief.get(s, 0.0)
            for s, allowed_set in self.pp_shield.items()
            if action in allowed_set
        )

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Propagate belief then delegate to ShieldedActionSelector.select."""
        while self._history_len < len(history):
            self._belief.propogate(history[self._history_len])
            self._history_len += 1
        return super().select(history, allowed_actions, context)

    def _select_fallback(
        self,
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Select allowed action with highest single-belief allowance probability."""
        action_scores = [
            (action, self._allowance_probability(action))
            for action in allowed_actions
        ]
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def reset(self):
        """Reset belief to uniform prior and clear history tracking."""
        self._belief.restart()
        self._history_len = 0


# ============================================================
# RL-Based Action Selector Base Class
# ============================================================

class RLActionSelector(ActionSelector):
    """Base class for RL-based action selectors.

    Provides interface for training and using learned policies.
    """

    @abstractmethod
    def train(
        self,
        ipomdp: "IPOMDP",
        pp_shield: Dict[Any, Set[Any]],
        rt_shield_factory: Callable,
        perception: "PerceptionModel",
        num_episodes: int,
        episode_length: int
    ) -> Dict[str, List[float]]:
        """Train the RL policy."""
        ...

    @abstractmethod
    def reset(self):
        """Reset any episode-specific state."""
        pass

