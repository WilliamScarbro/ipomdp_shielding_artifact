"""Perception model abstractions for Monte Carlo safety evaluation.

This module implements the 2-player game framework where Nature (Player 2)
chooses perception probabilities within IPOMDP intervals.

Nature can be cooperative (random) or adversarial (maximizing failure probability).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set
import random
import numpy as np
from scipy.optimize import linprog

from ..Models.ipomdp import IPOMDP


class PerceptionModel(ABC):
    """Base class for perception models.

    A perception model defines how observations are generated given the true state.
    It selects probabilities P(obs|state) within the IPOMDP intervals and samples
    an observation from this distribution.

    This abstraction allows for:
    - Random/uniform perception (cooperative nature)
    - Adversarial perception (nature maximizes failure probability)
    """

    @abstractmethod
    def sample_observation(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sample an observation given the true state.

        Parameters
        ----------
        state : any
            The true state
        ipomdp : IPOMDP
            The interval POMDP model (provides P_lower, P_upper)
        context : dict, optional
            Additional context (e.g., shield state, belief polytope)

        Returns
        -------
        observation
            Sampled observation within the interval constraints
        """
        pass

    def sample_distribution(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[Any, float]:
        """Return the full distribution P(obs|state) chosen by this model.

        Parameters
        ----------
        state : any
            The true state
        ipomdp : IPOMDP
            The interval POMDP model
        context : dict, optional
            Additional context

        Returns
        -------
        dict
            Mapping from observation to probability
        """
        # Default: return uniform within intervals (subclasses can override)
        return self._uniform_distribution(state, ipomdp)

    def _uniform_distribution(
        self,
        state: Any,
        ipomdp: IPOMDP
    ) -> Dict[Any, float]:
        """Compute a uniform-like distribution within intervals.

        Uses midpoints of intervals, then normalizes to sum to 1.
        """
        if state == "FAIL":
            return {"FAIL": 1.0}

        obs_probs = {}
        total = 0.0
        for obs in ipomdp.observations:
            lo = ipomdp.P_lower[state].get(obs, 0.0)
            hi = ipomdp.P_upper[state].get(obs, 0.0)
            # Use midpoint as initial weight
            mid = (lo + hi) / 2.0
            obs_probs[obs] = mid
            total += mid

        # Normalize
        if total > 0:
            for obs in obs_probs:
                obs_probs[obs] /= total
        else:
            # Fallback to uniform if all intervals are zero
            n = len(ipomdp.observations)
            for obs in ipomdp.observations:
                obs_probs[obs] = 1.0 / n

        return obs_probs


class UniformPerceptionModel(PerceptionModel):
    """Random perception within intervals (cooperative nature).

    Samples a valid probability distribution within the IPOMDP intervals,
    then samples an observation from that distribution.
    """

    def sample_observation(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sample uniformly from a valid distribution within intervals."""
        if state == "FAIL":
            return "FAIL"

        dist = self._sample_valid_distribution(state, ipomdp)
        observations = list(dist.keys())
        probs = [dist[o] for o in observations]
        return random.choices(observations, weights=probs, k=1)[0]

    def _sample_valid_distribution(
        self,
        state: Any,
        ipomdp: IPOMDP
    ) -> Dict[Any, float]:
        """Sample a valid distribution within intervals.

        Strategy: Sample each probability uniformly within its interval,
        then project onto the simplex (sum = 1) while respecting bounds.
        """
        observations = ipomdp.observations
        n = len(observations)

        # Get bounds
        lo = np.array([ipomdp.P_lower[state].get(o, 0.0) for o in observations])
        hi = np.array([ipomdp.P_upper[state].get(o, 0.0) for o in observations])

        # Sample uniformly in each interval
        sampled = np.array([random.uniform(lo[i], hi[i]) for i in range(n)])

        # Project onto simplex while respecting bounds
        probs = self._project_to_simplex_with_bounds(sampled, lo, hi)

        return {observations[i]: probs[i] for i in range(n)}

    def _project_to_simplex_with_bounds(
        self,
        x: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray
    ) -> np.ndarray:
        """Project x onto the simplex while respecting [lo, hi] bounds.

        Uses iterative clipping and normalization.
        """
        n = len(x)
        p = x.copy()

        for _ in range(100):  # Max iterations
            # Normalize to sum to 1
            total = p.sum()
            if total > 0:
                p = p / total
            else:
                p = np.ones(n) / n

            # Clip to bounds
            p = np.clip(p, lo, hi)

            # Check if valid (sum close to 1)
            if abs(p.sum() - 1.0) < 1e-6:
                break

        # Final normalization (may slightly violate bounds)
        total = p.sum()
        if total > 0:
            p = p / total

        return p


class AdversarialPerceptionModel(PerceptionModel):
    """Adversarial perception that maximizes failure probability.

    Nature chooses observation probabilities within IPOMDP intervals to
    maximize the likelihood that the agent fails. This can happen by:
    1. Misleading the shield's belief state to allow unsafe actions
    2. Causing the shield to get stuck (no allowed actions)

    The adversarial strategy scores each observation by its "badness"
    (how likely it leads to failure) and puts maximum probability on
    the worst observations within the interval constraints.
    """

    def __init__(
        self,
        pp_shield: Dict[Any, Set[Any]],
        strategy: str = "greedy",
        stuck_penalty: float = 0.5
    ):
        """Initialize adversarial perception model.

        Parameters
        ----------
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        strategy : str
            Adversarial strategy:
            - "greedy": Maximize weight on worst observation
            - "mixed": Distribute weight among bad observations
        stuck_penalty : float
            Relative badness of getting stuck vs failing (0 to 1)
            1.0 = stuck is as bad as failing
            0.0 = only direct failures count
        """
        self.pp_shield = pp_shield
        self.strategy = strategy
        self.stuck_penalty = stuck_penalty

    def sample_observation(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sample from adversarially-chosen distribution."""
        if state == "FAIL":
            return "FAIL"

        dist = self.sample_distribution(state, ipomdp, context)
        observations = list(dist.keys())
        probs = [dist[o] for o in observations]
        return random.choices(observations, weights=probs, k=1)[0]

    def sample_distribution(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[Any, float]:
        """Compute adversarial distribution maximizing failure probability.

        Parameters
        ----------
        state : any
            Current true state
        ipomdp : IPOMDP
            The interval POMDP model
        context : dict, optional
            May contain:
            - "rt_shield": RuntimeImpShield for belief-aware adversary
            - "history": observation-action history

        Returns
        -------
        dict
            Adversarial distribution over observations
        """
        if state == "FAIL":
            return {"FAIL": 1.0}

        observations = ipomdp.observations
        n = len(observations)

        # Get interval bounds
        lo = np.array([ipomdp.P_lower[state].get(o, 0.0) for o in observations])
        hi = np.array([ipomdp.P_upper[state].get(o, 0.0) for o in observations])

        # Score each observation by "badness"
        scores = self._score_observations(state, observations, ipomdp, context)

        # Solve LP to maximize expected badness within interval constraints
        probs = self._solve_adversarial_lp(scores, lo, hi)

        return {observations[i]: probs[i] for i in range(n)}

    def _score_observations(
        self,
        state: Any,
        observations: List[Any],
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Score observations by how bad they are for the agent.

        Higher score = worse for agent = better for adversary.

        The score combines:
        1. How much the observation misleads about true state
        2. How likely unsafe actions become allowed
        """
        n = len(observations)
        scores = np.zeros(n)

        # Get safe actions for current state
        safe_actions = self.pp_shield.get(state, set())

        for i, obs in enumerate(observations):
            # Score 1: Observation confusion
            # High score if obs suggests different state than true state
            confusion_score = self._observation_confusion(state, obs, ipomdp)

            # Score 2: Check which states this observation is likely from
            # and whether those states have different safe actions
            action_confusion = self._action_confusion(state, obs, ipomdp)

            scores[i] = confusion_score + action_confusion

        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _observation_confusion(
        self,
        true_state: Any,
        obs: Any,
        ipomdp: IPOMDP
    ) -> float:
        """Score how confusing an observation is.

        An observation is confusing if it's more likely from other states
        than from the true state (relative to intervals).
        """
        # How plausible is this observation from true state?
        true_lo = ipomdp.P_lower[true_state].get(obs, 0.0)
        true_hi = ipomdp.P_upper[true_state].get(obs, 0.0)
        true_mid = (true_lo + true_hi) / 2.0

        # How plausible from other states?
        other_total = 0.0
        count = 0
        for s in ipomdp.states:
            if s == true_state or s == "FAIL":
                continue
            lo = ipomdp.P_lower[s].get(obs, 0.0)
            hi = ipomdp.P_upper[s].get(obs, 0.0)
            other_total += (lo + hi) / 2.0
            count += 1

        other_avg = other_total / count if count > 0 else 0.0

        # Confusion: observation seems more likely from other states
        return max(0.0, other_avg - true_mid)

    def _action_confusion(
        self,
        true_state: Any,
        obs: Any,
        ipomdp: IPOMDP
    ) -> float:
        """Score how much an observation could lead to unsafe action selection.

        High score if observation suggests states where unsafe actions
        (for true state) are actually safe.
        """
        true_safe = self.pp_shield.get(true_state, set())
        all_actions = set(ipomdp.actions)
        true_unsafe = all_actions - true_safe

        if not true_unsafe:
            return 0.0  # All actions are safe, can't cause failure

        danger_score = 0.0

        for s in ipomdp.states:
            if s == "FAIL":
                continue

            # How plausible does obs make state s seem?
            lo = ipomdp.P_lower[s].get(obs, 0.0)
            hi = ipomdp.P_upper[s].get(obs, 0.0)
            plausibility = (lo + hi) / 2.0

            # How many of true_state's unsafe actions are safe in s?
            s_safe = self.pp_shield.get(s, set())
            overlap = len(true_unsafe & s_safe)

            # Weight by plausibility
            danger_score += plausibility * overlap

        return danger_score

    def _solve_adversarial_lp(
        self,
        scores: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray
    ) -> np.ndarray:
        """Solve LP to maximize expected badness within interval constraints.

        Maximize: sum_i scores[i] * p[i]
        Subject to:
            sum_i p[i] = 1
            lo[i] <= p[i] <= hi[i]
        """
        n = len(scores)

        # Check feasibility
        if lo.sum() > 1.0 + 1e-6 or hi.sum() < 1.0 - 1e-6:
            # Infeasible - fall back to midpoints
            mid = (lo + hi) / 2.0
            return mid / mid.sum() if mid.sum() > 0 else np.ones(n) / n

        # Maximize scores^T p (equivalent to minimize -scores^T p)
        c = -scores

        # Equality constraint: sum(p) = 1
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        # Bounds
        bounds = [(lo[i], hi[i]) for i in range(n)]

        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success:
                return np.clip(result.x, 0, 1)
        except Exception:
            pass

        # Fallback: greedy assignment
        return self._greedy_adversarial(scores, lo, hi)

    def _greedy_adversarial(
        self,
        scores: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray
    ) -> np.ndarray:
        """Greedy adversarial distribution when LP fails.

        Assigns maximum probability to highest-scored observations.
        """
        n = len(scores)
        probs = lo.copy()  # Start with lower bounds
        remaining = 1.0 - probs.sum()

        # Sort by score descending
        order = np.argsort(-scores)

        for i in order:
            if remaining <= 0:
                break
            room = hi[i] - probs[i]
            add = min(room, remaining)
            probs[i] += add
            remaining -= add

        # Ensure sum is 1
        total = probs.sum()
        if abs(total - 1.0) > 1e-6 and total > 0:
            probs = probs / total

        return probs


class LegacyPerceptionAdapter(PerceptionModel):
    """Adapter for legacy perception functions.

    Wraps a simple state -> observation function for backward compatibility.
    """

    def __init__(self, perception_fn: Callable[[Any], Any]):
        """Initialize with legacy perception function.

        Parameters
        ----------
        perception_fn : callable
            Function mapping state to observation
        """
        self.perception_fn = perception_fn

    def sample_observation(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Delegate to legacy perception function."""
        return self.perception_fn(state)
