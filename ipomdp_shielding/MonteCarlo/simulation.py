"""Core Monte Carlo simulation functions for safety evaluation.

This module provides the low-level simulation functions for running
Monte Carlo trials and computing safety metrics.
"""

from typing import Any, Callable, Dict, List, Optional, Set
import random
import numpy as np

from ..Models.ipomdp import IPOMDP
from ..Evaluation.runtime_shield import RuntimeImpShield

from .data_structures import SafetyTrialResult, MCSafetyMetrics, TimestepMetrics
from .action_selectors import ActionSelector
from .perception_models import PerceptionModel, LegacyPerceptionAdapter
from .initial_states import InitialStateGenerator


def run_single_trial(
    trial_id: int,
    ipomdp: IPOMDP,
    initial_state: Any,
    initial_action: Any,
    rt_shield: RuntimeImpShield,
    perception: "PerceptionModel | Callable[[Any], Any]",
    action_selector: ActionSelector,
    trial_length: int,
    store_trajectory: bool = False
) -> SafetyTrialResult:
    """Run a single Monte Carlo trial.

    Parameters
    ----------
    trial_id : int
        Trial identifier
    ipomdp : IPOMDP
        The interval POMDP model
    initial_state : any
        Starting state
    initial_action : any
        Initial action
    rt_shield : RuntimeImpShield
        Runtime shield instance (should be fresh/restarted)
    perception : PerceptionModel or callable
        Perception model (state, ipomdp, context) -> observation
        or legacy callable (state) -> observation
    action_selector : ActionSelector
        Strategy for selecting actions
    trial_length : int
        Maximum number of steps
    store_trajectory : bool
        Whether to store full trajectory (memory intensive)

    Returns
    -------
    SafetyTrialResult
        Complete trial result
    """
    # Wrap legacy perception functions
    if callable(perception) and not isinstance(perception, PerceptionModel):
        perception = LegacyPerceptionAdapter(perception)

    state = initial_state
    action = initial_action
    history = []  # List of (obs, action) pairs
    trajectory = []

    outcome = "safe"  # Default outcome
    fail_step = None
    steps_completed = 0

    # Context for adversarial perception
    perception_context = {
        "rt_shield": rt_shield,
        "history": history
    }

    for step in range(trial_length):
        # Check for failure
        if state == "FAIL":
            outcome = "fail"
            fail_step = step
            break

        # Get observation using perception model
        obs = perception.sample_observation(state, ipomdp, perception_context)

        # Store in history
        history.append((obs, action))

        if store_trajectory:
            trajectory.append((state, obs, action))

        # Get allowed actions from shield
        allowed_actions = rt_shield.next_actions((obs, action))

        # Check if stuck (no allowed actions)
        if not allowed_actions:
            outcome = "stuck"
            steps_completed = step
            break

        # Build context for action selector
        action_selector_context = {
            "rt_shield": rt_shield,
            "history": history
        }

        # Select next action
        action = action_selector.select(history, allowed_actions, context=action_selector_context)

        # Evolve state
        state = ipomdp.evolve(state, action)
        steps_completed = step + 1

    # If loop completed without failure or stuck, outcome remains "safe"

    result = SafetyTrialResult(
        trial_id=trial_id,
        outcome=outcome,
        steps_completed=steps_completed,
        stuck_count=rt_shield.stuck_count,
        fail_step=fail_step,
        trajectory=trajectory if store_trajectory else []
    )

    return result


def run_monte_carlo_trials(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, Set[Any]],
    perception: "PerceptionModel | Callable[[Any], Any]",
    rt_shield_factory: Callable[[], RuntimeImpShield],
    action_selector: ActionSelector,
    initial_generator: InitialStateGenerator,
    num_trials: int,
    trial_length: int,
    store_trajectories: bool = False,
    seed: Optional[int] = None
) -> List[SafetyTrialResult]:
    """Run multiple Monte Carlo trials.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    perception : PerceptionModel or callable
        Perception model (can be adversarial) or legacy callable
    rt_shield_factory : callable
        Factory function returning fresh RuntimeImpShield instance
    action_selector : ActionSelector
        Strategy for selecting actions
    initial_generator : InitialStateGenerator
        Strategy for sampling initial states
    num_trials : int
        Number of trials to run
    trial_length : int
        Maximum steps per trial
    store_trajectories : bool
        Whether to store full trajectories (memory intensive)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of SafetyTrialResult
        Results from all trials
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results = []

    for trial_id in range(num_trials):
        # Generate initial state
        initial_state, initial_action = initial_generator.generate(ipomdp, pp_shield)

        # Create fresh shield instance
        rt_shield = rt_shield_factory()
        rt_shield.restart()

        # Run trial
        result = run_single_trial(
            trial_id=trial_id,
            ipomdp=ipomdp,
            initial_state=initial_state,
            initial_action=initial_action,
            rt_shield=rt_shield,
            perception=perception,
            action_selector=action_selector,
            trial_length=trial_length,
            store_trajectory=store_trajectories
        )

        results.append(result)

    return results


def compute_safety_metrics(
    results: List[SafetyTrialResult]
) -> MCSafetyMetrics:
    """Compute aggregated metrics from trial results.

    Parameters
    ----------
    results : list of SafetyTrialResult
        Results from Monte Carlo trials

    Returns
    -------
    MCSafetyMetrics
        Aggregated safety metrics
    """
    if not results:
        return MCSafetyMetrics(
            num_trials=0,
            fail_rate=0.0,
            stuck_rate=0.0,
            safe_rate=0.0,
            mean_steps=0.0,
            mean_stuck_count=0.0
        )

    num_trials = len(results)

    # Count outcomes
    fail_count = sum(1 for r in results if r.outcome == "fail")
    stuck_count = sum(1 for r in results if r.outcome == "stuck")
    safe_count = sum(1 for r in results if r.outcome == "safe")

    # Compute rates
    fail_rate = fail_count / num_trials
    stuck_rate = stuck_count / num_trials
    safe_rate = safe_count / num_trials

    # Aggregate trajectory metrics
    mean_steps = sum(r.steps_completed for r in results) / num_trials
    mean_stuck_count = sum(r.stuck_count for r in results) / num_trials

    # Build failure step distribution
    fail_step_distribution = [r.fail_step for r in results if r.fail_step is not None]

    return MCSafetyMetrics(
        num_trials=num_trials,
        fail_rate=fail_rate,
        stuck_rate=stuck_rate,
        safe_rate=safe_rate,
        mean_steps=mean_steps,
        mean_stuck_count=mean_stuck_count,
        fail_step_distribution=fail_step_distribution
    )


def compute_timestep_metrics(
    results: List[SafetyTrialResult],
    trial_length: int
) -> TimestepMetrics:
    """Compute cumulative fail/stuck/safe probabilities at each timestep.

    For each timestep t, computes:
    - fail_prob: Fraction of trials that hit FAIL at or before t
    - stuck_prob: Fraction of trials that got stuck at or before t
    - safe_prob: Fraction of trials still running at t (= 1 - fail - stuck)

    Parameters
    ----------
    results : list of SafetyTrialResult
        Trial results from Monte Carlo evaluation
    trial_length : int
        Maximum trial length (for x-axis)

    Returns
    -------
    TimestepMetrics
        Cumulative probabilities at each timestep
    """
    num_trials = len(results)
    if num_trials == 0:
        return TimestepMetrics(
            num_trials=0,
            trial_length=trial_length,
            fail_prob_by_timestep=[],
            stuck_prob_by_timestep=[],
            safe_prob_by_timestep=[]
        )

    # For each trial, determine when it terminated and why
    # termination_step = step at which trial ended (None if completed all steps safely)
    # termination_reason = "fail", "stuck", or None (if safe)
    trial_terminations = []
    for r in results:
        if r.outcome == "fail":
            # Failed at fail_step
            trial_terminations.append((r.fail_step, "fail"))
        elif r.outcome == "stuck":
            # Got stuck at steps_completed
            trial_terminations.append((r.steps_completed, "stuck"))
        else:
            # Completed safely - no termination
            trial_terminations.append((None, None))

    # Compute cumulative probabilities at each timestep
    fail_prob_by_timestep = []
    stuck_prob_by_timestep = []
    safe_prob_by_timestep = []

    for t in range(trial_length):
        # Count trials that have failed at or before timestep t
        fail_count = sum(
            1 for (term_step, term_reason) in trial_terminations
            if term_reason == "fail" and term_step is not None and term_step <= t
        )

        # Count trials that have gotten stuck at or before timestep t
        stuck_count = sum(
            1 for (term_step, term_reason) in trial_terminations
            if term_reason == "stuck" and term_step is not None and term_step <= t
        )

        # Safe = still running (haven't failed or gotten stuck yet)
        safe_count = num_trials - fail_count - stuck_count

        fail_prob_by_timestep.append(fail_count / num_trials)
        stuck_prob_by_timestep.append(stuck_count / num_trials)
        safe_prob_by_timestep.append(safe_count / num_trials)

    return TimestepMetrics(
        num_trials=num_trials,
        trial_length=trial_length,
        fail_prob_by_timestep=fail_prob_by_timestep,
        stuck_prob_by_timestep=stuck_prob_by_timestep,
        safe_prob_by_timestep=safe_prob_by_timestep
    )
