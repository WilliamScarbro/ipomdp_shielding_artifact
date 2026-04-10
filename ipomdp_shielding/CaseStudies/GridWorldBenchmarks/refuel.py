"""Refuel gridworld POMDP case study.

Implements the Refuel benchmark from:
    Carr et al., "Safe Reinforcement Learning via Shielding under Partial
    Observability", arXiv:2204.00755.

Source model: gridstorm/models/files/refuel.nm
  - N×N grid with 1 obstacle, 3 refuel stations
  - Fuel management: fuel ∈ [0, ENERGY]
  - Slippage = 0.3 (30% chance of extra step)
  - State-dependent action availability

The deterministic PRISM observation function is perturbed by obs_noise to
create a proper interval POMDP (non-trivial P_lower / P_upper bounds).
"""

import math
from typing import Dict, List, Set, Tuple

from ...Models.ipomdp import IPOMDP

FAIL = "FAIL"


def refuel_states(N: int, ENERGY: int, with_fail: bool = False) -> List:
    """Generate all states for the Refuel benchmark.

    Args:
        N:         Grid size (N×N).
        ENERGY:    Maximum fuel level.
        with_fail: Include the FAIL absorbing state.

    Returns:
        List of (ax, ay, fuel) tuples, plus FAIL if requested.
    """
    states = [
        (ax, ay, fuel)
        for ax in range(N)
        for ay in range(N)
        for fuel in range(ENERGY + 1)
    ]
    if with_fail:
        states.append(FAIL)
    return states


def refuel_obstacle_pos(N: int) -> Tuple[int, int]:
    """Return the single obstacle position.

    Args:
        N: Grid size.

    Returns:
        (ax, ay) of the obstacle.
    """
    return (N - 2, N - 2)


def refuel_station_positions(N: int) -> Set[Tuple[int, int]]:
    """Return the set of refuel station positions.

    Args:
        N: Grid size.

    Returns:
        Set of (ax, ay) station positions.
    """
    return {(0, 0), (N // 3, N // 3), (2 * (N // 3) - 1, 2 * (N // 3) - 1)}


def refuel_actions(state, N: int, ENERGY: int) -> List[str]:
    """Return the list of available actions for a given state.

    Action availability (all conditions must hold for movement):
        - 'north': ay > 0  AND  fuel > 0  AND  not canRefuel
        - 'south': ay < N-1  AND  fuel > 0  AND  not canRefuel
        - 'west':  ax > 0  AND  fuel > 0  AND  not canRefuel
        - 'east':  ax < N-1  AND  fuel > 0  AND  not canRefuel
        - 'refuel': atStation  AND  fuel < ENERGY  (canRefuel)

    When fuel == 0 and not at a station, no actions are available (stuck).

    Args:
        state:  (ax, ay, fuel) or FAIL.
        N:      Grid size.
        ENERGY: Maximum fuel level.

    Returns:
        List of available action strings.
    """
    if state == FAIL:
        return []
    ax, ay, fuel = state
    stations = refuel_station_positions(N)
    atStation = (ax, ay) in stations
    canRefuel = atStation and fuel < ENERGY

    available = []
    if canRefuel:
        available.append('refuel')
    if fuel > 0 and not canRefuel:
        if ay > 0:
            available.append('north')
        if ay < N - 1:
            available.append('south')
        if ax > 0:
            available.append('west')
        if ax < N - 1:
            available.append('east')
    return available


def refuel_observation_fn(N: int, ENERGY: int):
    """Return a deterministic function mapping states to observations.

    Observation tuple components (all derived from state (ax, ay, fuel)):
        0: ax > 0                              # can go west
        1: ax < N-1                            # can go east
        2: ay > 0                              # can go north
        3: ay < N-1                            # can go south
        4: ax == N-1 and ay == N-1             # amdone (at goal)
        5: atStation and fuel < ENERGY         # refuelAllowed
        6: fuel == ENERGY                      # fuel full
        7: fuel // max(ENERGY // 2, 1)         # fuel meter (coarse level)

    Note: hascrash (obstacle position) and fuel > 0 are intentionally omitted
    so that the agent cannot directly observe safety-critical state features.
    This makes belief tracking over fuel level and obstacle proximity necessary,
    which is precisely where IPOMDP shielding adds value.

    Args:
        N:      Grid size.
        ENERGY: Maximum fuel level.

    Returns:
        Callable s -> 8-element tuple.
    """
    stations = refuel_station_positions(N)
    obstacle = refuel_obstacle_pos(N)
    fuel_divisor = max(ENERGY // 2, 1)

    def obs_fn(s) -> tuple:
        if s == FAIL:
            # FAIL emits the observation of the obstacle cell with zero fuel.
            # hascrash and fuel > 0 are not in the observation, so FAIL is
            # indistinguishable from a low-fuel non-station non-goal state.
            oax, oay = obstacle
            return (
                oax > 0,
                oax < N - 1,
                oay > 0,
                oay < N - 1,
                False,   # amdone
                False,   # refuelAllowed
                False,   # fuel full
                0,       # fuel meter
            )
        ax, ay, fuel = s
        atStation = (ax, ay) in stations
        return (
            ax > 0,
            ax < N - 1,
            ay > 0,
            ay < N - 1,
            ax == N - 1 and ay == N - 1,   # amdone
            atStation and fuel < ENERGY,    # refuelAllowed
            fuel == ENERGY,                 # fuel full
            fuel // fuel_divisor,           # fuel meter
        )

    return obs_fn


def refuel_observations(N: int, ENERGY: int) -> List[tuple]:
    """Return the list of all possible observations.

    Enumerates all distinct observation tuples produced by
    refuel_observation_fn over the full state space (including FAIL).

    Args:
        N:      Grid size.
        ENERGY: Maximum fuel level.

    Returns:
        Sorted list of distinct 8-element observation tuples.
    """
    obs_fn = refuel_observation_fn(N, ENERGY)
    seen = set()
    for s in refuel_states(N, ENERGY, with_fail=False):
        seen.add(obs_fn(s))
    seen.add(obs_fn(FAIL))
    return sorted(seen)


def refuel_safe(s, N: int, ENERGY: int) -> bool:
    """Check if a state satisfies the safety property.

    Safety predicate: !hascrash AND (fuel > 0 OR atStation)

    Args:
        s:      State (ax, ay, fuel) or FAIL.
        N:      Grid size.
        ENERGY: Maximum fuel level (unused here, kept for consistency).

    Returns:
        True if the state is safe.
    """
    if s == FAIL:
        return False
    ax, ay, fuel = s
    obstacle = refuel_obstacle_pos(N)
    stations = refuel_station_positions(N)
    hascrash = (ax, ay) == obstacle
    atStation = (ax, ay) in stations
    return (not hascrash) and (fuel > 0 or atStation)


def refuel_dynamics(N: int, ENERGY: int) -> Dict:
    """Build the exact transition dictionary for the Refuel benchmark.

    Slippage = 0.3:
        Primary movement (probability 0.7): move 1 step in chosen direction.
        Secondary movement (probability 0.3): move 2 steps in chosen direction.
    Both are clamped at grid boundaries.  Moving into the obstacle → FAIL.
    Each movement step decreases fuel by 1.

    Unavailable actions (wrong direction, no fuel, or forced-refuel) result in
    a self-loop so that T is defined for all (state, action) pairs.

    FAIL is absorbing: all actions from FAIL → FAIL.

    Args:
        N:      Grid size.
        ENERGY: Maximum fuel level.

    Returns:
        T: Dict mapping (state, action) -> {next_state: probability}.
    """
    obstacle = refuel_obstacle_pos(N)
    stations = refuel_station_positions(N)
    all_actions = ['north', 'south', 'east', 'west', 'refuel']

    def _primary(ax, ay, a):
        if a == 'north':
            return (ax, max(ay - 1, 0))
        if a == 'south':
            return (ax, min(ay + 1, N - 1))
        if a == 'west':
            return (max(ax - 1, 0), ay)
        return (min(ax + 1, N - 1), ay)  # 'east'

    def _secondary(ax, ay, a):
        if a == 'north':
            return (ax, max(ay - 2, 0))
        if a == 'south':
            return (ax, min(ay + 2, N - 1))
        if a == 'west':
            return (max(ax - 2, 0), ay)
        return (min(ax + 2, N - 1), ay)  # 'east'

    def _next_state(pos, fuel):
        nax, nay = pos
        if (nax, nay) == obstacle:
            return FAIL
        return (nax, nay, fuel - 1)

    T = {}

    # FAIL: absorbing.
    for a in all_actions:
        T[(FAIL, a)] = {FAIL: 1.0}

    for ax in range(N):
        for ay in range(N):
            for fuel in range(ENERGY + 1):
                s = (ax, ay, fuel)
                atStation = (ax, ay) in stations
                canRefuel = atStation and fuel < ENERGY

                for a in all_actions:
                    if a == 'refuel':
                        if canRefuel:
                            T[(s, 'refuel')] = {(ax, ay, ENERGY): 1.0}
                        else:
                            T[(s, 'refuel')] = {s: 1.0}  # self-loop

                    else:  # movement actions
                        # Check direction is open.
                        dir_open = (
                            (a == 'north' and ay > 0) or
                            (a == 'south' and ay < N - 1) or
                            (a == 'west' and ax > 0) or
                            (a == 'east' and ax < N - 1)
                        )
                        if fuel > 0 and not canRefuel and dir_open:
                            p1 = _primary(ax, ay, a)
                            p2 = _secondary(ax, ay, a)
                            ns1 = _next_state(p1, fuel)
                            ns2 = _next_state(p2, fuel)
                            dist: Dict = {}
                            dist[ns1] = dist.get(ns1, 0.0) + 0.7
                            dist[ns2] = dist.get(ns2, 0.0) + 0.3
                            T[(s, a)] = dist
                        else:
                            T[(s, a)] = {s: 1.0}  # self-loop

    return T


def build_refuel_ipomdp(
    N: int = 7,
    ENERGY: int = 6,
    obs_noise: float = 0.3,
    seed=None,
) -> Tuple[IPOMDP, Dict, List, None]:
    """Build the complete Refuel IPOMDP.

    Args:
        N:         Grid size (default 7).
        ENERGY:    Maximum fuel level (default 6).
        obs_noise: Observation noise level in [0, 1) (default 0.3).
                   Set to 0.3 so that the RL agent fails ~10-15% without
                   shielding; lower values (e.g. 0.05) allow the agent to
                   memorise a safe path, making shielding unnecessary.

    Returns:
        (ipomdp, pp_shield, initial_states, None) where:
        - ipomdp:         Complete IPOMDP model.
        - pp_shield:      Perfect-perception shield mapping state -> set of
                          safe actions (1-step lookahead, conservative).
        - initial_states: List of valid initial states (starting at (0,0)
                          with full fuel — a refuel station).
        - None:           Placeholder for interface compatibility.
    """
    from . import _make_interval_perception

    states = refuel_states(N, ENERGY, with_fail=True)
    regular_states = refuel_states(N, ENERGY, with_fail=False)
    observations = refuel_observations(N, ENERGY)
    actions = ['north', 'south', 'east', 'west', 'refuel']

    T = refuel_dynamics(N, ENERGY)
    obs_fn = refuel_observation_fn(N, ENERGY)
    fail_obs = obs_fn(FAIL)

    # Distance-scaled noise: concentrate uncertainty on observations similar
    # to the true observation.  Observations differing in few elements receive
    # up to obs_noise mass; maximally dissimilar observations receive nearly 0.
    # Uses a Gaussian kernel on normalised Hamming distance over the 8-tuple:
    #   elements 0-6 (bool): each contributes 1 if different
    #   element 7 (fuel_meter int): contributes |Δ| / max_fuel_meter
    #   total distance normalised to [0, 1] by dividing by 8
    # alpha = 5  ->  1 bit different: scale ≈ 0.54,  4 bits: ≈ 0.08,  all: ≈ 0.002
    _max_fm = max(o[7] for o in observations) or 1

    def _refuel_obs_noise_scale(o_true, o_other, _max_fm=_max_fm, _alpha=5.0):
        bool_dist = sum(abs(int(a) - int(b)) for a, b in zip(o_true[:7], o_other[:7]))
        fuel_dist = abs(o_true[7] - o_other[7]) / _max_fm
        return math.exp(-_alpha * (bool_dist + fuel_dist) / 8.0)

    P_lower, P_upper = _make_interval_perception(
        regular_states, observations, obs_fn, obs_noise, FAIL, fail_obs,
        obs_noise_scale_fn=_refuel_obs_noise_scale,
    )

    ipomdp = IPOMDP(states, observations, actions, T, P_lower, P_upper)

    # Perfect-perception shield: action a is safe from s iff all reachable
    # next states satisfy the refuel safety predicate (conservative).
    pp_shield: Dict = {}
    for s in states:
        if s == FAIL or not refuel_safe(s, N, ENERGY):
            pp_shield[s] = set()
        else:
            safe_acts = set()
            for a in actions:
                dist = T.get((s, a), {})
                if dist and all(refuel_safe(ns, N, ENERGY) for ns in dist):
                    safe_acts.add(a)
            pp_shield[s] = safe_acts

    # Start at (0, 0) with full fuel (this is a refuel station).
    initial_states = [(0, 0, ENERGY)]

    return ipomdp, pp_shield, initial_states, None
