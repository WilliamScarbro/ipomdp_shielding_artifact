"""CartPole case study for IPOMDP shielding.

This module demonstrates vision-based perception shielding where a CNN estimates
the 4D continuous state (position, velocity, pole angle, angular velocity) from
frame pairs.
"""

from .cartpole import (
    cartpole_states,
    cartpole_actions,
    cartpole_dynamics,
    cartpole_perception,
    cartpole_safe,
    cartpole_safe_action,
    build_cartpole_ipomdp,
    cartpole_evaluation,
    FAIL,
)

from .data_loader import (
    get_bin_edges,
    get_confusion_data,
)

__all__ = [
    'cartpole_states',
    'cartpole_actions',
    'cartpole_dynamics',
    'cartpole_perception',
    'cartpole_safe',
    'cartpole_safe_action',
    'build_cartpole_ipomdp',
    'cartpole_evaluation',
    'get_bin_edges',
    'get_confusion_data',
    'FAIL',
]
