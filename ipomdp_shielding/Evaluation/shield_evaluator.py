"""Experimental evaluation of runtime shields."""

import random
from typing import Callable, Any, Tuple, Optional

from ..Models import IPOMDP
from .runtime_shield import RuntimeImpShield


def evaluate_runtime_shield(
    ipomdp: IPOMDP,
    perceptor: Callable[[Any], Any],
    rt_shield: RuntimeImpShield,
    trials: int = 100,
    trial_length: int = 10,
    start_state_action: Optional[Tuple[Any, Any]] = None
) -> dict:
    """
    Experimentally assess safety of shielding method.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    perceptor : callable
        Function mapping true state to estimated state (perception model)
    rt_shield : RuntimeImpShield
        The runtime shield to evaluate
    trials : int
        Number of simulation trials
    trial_length : int
        Maximum steps per trial
    start_state_action : tuple, optional
        Initial (state, action) pair. If None, random initial state/action.

    Returns
    -------
    dict
        Statistics including average_length, fail_rate, stuck_likelihood
    """
    trace_data = []

    for t in range(trials):
        if start_state_action is None:
            current_state = random.choice(ipomdp.states)
            current_action = random.choice(list(ipomdp.actions))
        else:
            current_state, current_action = start_state_action

        rt_shield.restart()

        fail_flag = False
        for i in range(trial_length):
            current_est = perceptor(current_state)

            actions = rt_shield.next_actions((current_est, current_action))
            current_action = random.choice(actions)

            current_state = ipomdp_evolve(ipomdp, current_state, current_action)

            if current_state == "FAIL":
                trace_data.append({
                    "steps": i,
                    "stuck_count": rt_shield.stuck_count,
                    "failed": True
                })
                fail_flag = True
                break

        if not fail_flag:
            trace_data.append({
                "steps": trial_length,
                "stuck_count": rt_shield.stuck_count,
                "failed": False
            })

    # Compute statistics
    average_length = sum(td["steps"] for td in trace_data) / trials
    fail_rate = sum(1 if td["failed"] else 0 for td in trace_data) / trials
    stuck_likelihood = sum(td["stuck_count"] for td in trace_data) / (average_length * trials)

    print("------------------------------")
    print("#### Statistics #####")
    print("average length:", average_length)
    print("fail rate:", fail_rate)
    print("stuck_likelihood:", stuck_likelihood)

    return {
        "average_length": average_length,
        "fail_rate": fail_rate,
        "stuck_likelihood": stuck_likelihood,
        "trace_data": trace_data
    }
