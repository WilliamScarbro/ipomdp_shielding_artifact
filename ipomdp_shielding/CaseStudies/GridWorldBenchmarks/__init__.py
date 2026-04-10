"""Gridworld POMDP benchmarks from Carr et al. (arXiv:2204.00755).

Implements Obstacle and Refuel case studies as noisy IPOMDPs, following the
pattern of CartPole and TaxiNet.  Deterministic PRISM observation functions are
perturbed with an obs_noise parameter to produce non-trivial interval bounds:

    P_lower[s][o_true] = 1 - obs_noise,  P_upper[s][o_true] = 1.0
    P_lower[s][o']     = 0.0,             P_upper[s][o']     = obs_noise * scale(o_true, o')

where scale() = 1.0 (uniform) by default, or a caller-supplied distance-based
function that concentrates noise mass on observations similar to o_true.
"""

from .obstacle import (
    obstacle_states,
    obstacle_actions,
    obstacle_obstacles,
    obstacle_initial_states,
    obstacle_dynamics,
    obstacle_observation_fn,
    obstacle_observations,
    obstacle_safe,
    build_obstacle_ipomdp,
    FAIL as OBSTACLE_FAIL,
)

from .refuel import (
    refuel_states,
    refuel_actions,
    refuel_obstacle_pos,
    refuel_station_positions,
    refuel_dynamics,
    refuel_observation_fn,
    refuel_observations,
    refuel_safe,
    build_refuel_ipomdp,
    FAIL as REFUEL_FAIL,
)

__all__ = [
    # Obstacle
    'obstacle_states',
    'obstacle_actions',
    'obstacle_obstacles',
    'obstacle_initial_states',
    'obstacle_dynamics',
    'obstacle_observation_fn',
    'obstacle_observations',
    'obstacle_safe',
    'build_obstacle_ipomdp',
    'OBSTACLE_FAIL',
    # Refuel
    'refuel_states',
    'refuel_actions',
    'refuel_obstacle_pos',
    'refuel_station_positions',
    'refuel_dynamics',
    'refuel_observation_fn',
    'refuel_observations',
    'refuel_safe',
    'build_refuel_ipomdp',
    'REFUEL_FAIL',
    # Shared utility
    '_make_interval_perception',
]


def _make_interval_perception(
    states, observations, obs_fn, obs_noise, fail_state, fail_obs,
    obs_noise_scale_fn=None,
):
    """Build P_lower, P_upper from a deterministic observation function + noise level.

    For each regular state s with true observation o_true = obs_fn(s):
        P_lower[s][o_true] = 1 - obs_noise,  P_upper[s][o_true] = 1.0
        P_lower[s][o']     = 0.0,             P_upper[s][o']     = obs_noise * scale(o_true, o')

    where scale(o_true, o') = obs_noise_scale_fn(o_true, o') if provided, else 1.0.
    The scale function should return values in [0, 1] and can be used to
    concentrate noise on observations close to o_true (e.g. via a distance kernel).

    For the absorbing FAIL state:
        P_lower[FAIL][fail_obs] = 1.0,  P_upper[FAIL][fail_obs] = 1.0
        P_lower[FAIL][o']       = 0.0,  P_upper[FAIL][o']       = 0.0

    The resulting intervals satisfy:
        sum_o P_lower[s][o] <= 1 <= sum_o P_upper[s][o]  for all s.

    Args:
        states:             List of regular (non-FAIL) states.
        observations:       List of all possible observations.
        obs_fn:             Callable s -> observation (deterministic).
        obs_noise:          Noise level in [0, 1).
        fail_state:         The absorbing FAIL state (added to the output dicts).
        fail_obs:           The observation emitted with certainty by fail_state.
        obs_noise_scale_fn: Optional callable (o_true, o') -> float in [0, 1].
                            Scales the noise budget for each non-true observation.
                            Defaults to uniform (scale = 1 for all o').

    Returns:
        (P_lower, P_upper) each mapping state -> {obs -> float}.
    """
    P_lower = {}
    P_upper = {}
    for s in states:
        o_true = obs_fn(s)
        P_lower[s] = {o: 0.0 for o in observations}
        if obs_noise_scale_fn is not None:
            P_upper[s] = {
                o: (1.0 if o == o_true else obs_noise * obs_noise_scale_fn(o_true, o))
                for o in observations
            }
        else:
            P_upper[s] = {o: obs_noise for o in observations}
            P_upper[s][o_true] = 1.0
        P_lower[s][o_true] = 1.0 - obs_noise

    # FAIL emits fail_obs with certainty (point interval).
    P_lower[fail_state] = {o: 0.0 for o in observations}
    P_upper[fail_state] = {o: 0.0 for o in observations}
    P_lower[fail_state][fail_obs] = 1.0
    P_upper[fail_state][fail_obs] = 1.0

    return P_lower, P_upper
