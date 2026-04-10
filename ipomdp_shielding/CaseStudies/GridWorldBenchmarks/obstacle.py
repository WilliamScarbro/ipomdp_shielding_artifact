"""Obstacle gridworld POMDP case study.

Implements the Obstacle benchmark from:
    Carr et al., "Safe Reinforcement Learning via Shielding under Partial
    Observability", arXiv:2204.00755.

Source model: gridstorm/models/files/obstacle.nm
  - N×N grid with 5 static obstacles
  - Slippage = 0.1 (10% chance of extra step)
  - Random initial placement over 4 positions

The deterministic PRISM observation function is perturbed by obs_noise to
create a proper interval POMDP (non-trivial P_lower / P_upper bounds).
"""

from typing import Dict, List, Set, Tuple

from ...Models.ipomdp import IPOMDP

FAIL = "FAIL"


def obstacle_states(N: int, with_fail: bool = False) -> List:
    """Generate all grid states for the Obstacle benchmark.

    Args:
        N:         Grid size (N×N).
        with_fail: Include the FAIL absorbing state.

    Returns:
        List of (ax, ay) tuples for ax, ay ∈ [0, N-1], plus FAIL if requested.
    """
    states = [(ax, ay) for ax in range(N) for ay in range(N)]
    if with_fail:
        states.append(FAIL)
    return states


def obstacle_actions() -> List[str]:
    """Return the list of actions (same for every non-FAIL state).

    Returns:
        ['north', 'south', 'east', 'west']
    """
    return ['north', 'south', 'east', 'west']


def obstacle_obstacles(N: int) -> Set[Tuple[int, int]]:
    """Return the set of static obstacle positions.

    Args:
        N: Grid size.

    Returns:
        Set of (ax, ay) positions occupied by obstacles.
    """
    return {(N - 2, N - 2), (N - 1, 1), (1, 0), (N - 1, N - 2), (N - 4, N - 2)}


def obstacle_initial_states(N: int) -> List[Tuple[int, int]]:
    """Return the list of valid initial positions (uniform distribution).

    Args:
        N: Grid size.

    Returns:
        List of (ax, ay) start positions.
    """
    return [(N - 3, N - 2), (1, 1), (2, 1), (1, 3)]


def obstacle_observations() -> List[Tuple[bool, bool]]:
    """Return the list of possible observations.

    Observations are (hascrash, amdone) tuples:
        (False, False) — normal position (not at obstacle, not at goal)
        (True,  False) — at an obstacle, or in FAIL state
        (False, True)  — at the goal (ax == N-1, ay == N-1)

    Returns:
        List of 3 (hascrash, amdone) tuples.
    """
    return [(False, False), (True, False), (False, True)]


def obstacle_observation_fn(N: int):
    """Return a deterministic function mapping states to observations.

    Args:
        N: Grid size.

    Returns:
        Callable s -> (hascrash: bool, amdone: bool).
    """
    obstacles = obstacle_obstacles(N)

    def obs_fn(s) -> Tuple[bool, bool]:
        if s == FAIL:
            return (True, False)
        ax, ay = s
        hascrash = (ax, ay) in obstacles
        amdone = (ax == N - 1) and (ay == N - 1)
        return (hascrash, amdone)

    return obs_fn


def obstacle_safe(s, N: int) -> bool:
    """Check if a state is safe (not at an obstacle, not FAIL).

    Args:
        s: State (ax, ay) or FAIL.
        N: Grid size.

    Returns:
        True if the state satisfies the safety property !hascrash.
    """
    if s == FAIL:
        return False
    ax, ay = s
    return (ax, ay) not in obstacle_obstacles(N)


def obstacle_dynamics(N: int) -> Dict:
    """Build the exact transition dictionary for the Obstacle benchmark.

    Slippage = 0.1:
        Primary movement (probability 0.9): move 1 step in the chosen direction.
        Secondary movement (probability 0.1): move 2 steps in the chosen direction.
    Both are clamped at grid boundaries.  If a candidate next position is an
    obstacle, the agent transitions to FAIL instead.

    FAIL is absorbing: all actions from FAIL → FAIL.
    Obstacle grid positions are absorbing into FAIL (they are in the state space
    for completeness but are never reachable from the initial states).

    Args:
        N: Grid size.

    Returns:
        T: Dict mapping (state, action) -> {next_state: probability}.
    """
    obstacles = obstacle_obstacles(N)
    actions = obstacle_actions()
    states = obstacle_states(N, with_fail=True)

    def primary_secondary(ax: int, ay: int, a: str):
        """Return (primary_pos, secondary_pos) for action a with slippage."""
        if a == 'north':
            return (ax, max(ay - 1, 0)), (ax, max(ay - 2, 0))
        if a == 'south':
            return (ax, min(ay + 1, N - 1)), (ax, min(ay + 2, N - 1))
        if a == 'west':
            return (max(ax - 1, 0), ay), (max(ax - 2, 0), ay)
        # 'east'
        return (min(ax + 1, N - 1), ay), (min(ax + 2, N - 1), ay)

    def redirect(pos):
        """Redirect obstacle positions to FAIL."""
        return FAIL if pos in obstacles else pos

    T = {}
    for s in states:
        for a in actions:
            if s == FAIL:
                T[(s, a)] = {FAIL: 1.0}
                continue
            ax, ay = s
            if (ax, ay) in obstacles:
                # Obstacle positions absorb into FAIL.
                T[(s, a)] = {FAIL: 1.0}
                continue

            p_pos, s_pos = primary_secondary(ax, ay, a)
            ns1 = redirect(p_pos)
            ns2 = redirect(s_pos)

            dist: Dict = {}
            dist[ns1] = dist.get(ns1, 0.0) + 0.9
            dist[ns2] = dist.get(ns2, 0.0) + 0.1
            T[(s, a)] = dist

    return T


def build_obstacle_ipomdp(
    N: int = 7,
    obs_noise: float = 0.1,
    seed=None,
) -> Tuple[IPOMDP, Dict, List, None]:
    """Build the complete Obstacle IPOMDP.

    Args:
        N:         Grid size (default 7).
        obs_noise: Observation noise level in [0, 1) (default 0.1).

    Returns:
        (ipomdp, pp_shield, initial_states, None) where:
        - ipomdp:         Complete IPOMDP model.
        - pp_shield:      Perfect-perception shield mapping state -> set of
                          safe actions (1-step lookahead, conservative).
        - initial_states: List of valid initial (ax, ay) positions.
        - None:           Placeholder for interface compatibility.
    """
    from . import _make_interval_perception

    states = obstacle_states(N, with_fail=True)
    regular_states = obstacle_states(N, with_fail=False)
    observations = obstacle_observations()
    actions = obstacle_actions()

    T = obstacle_dynamics(N)
    obs_fn = obstacle_observation_fn(N)

    P_lower, P_upper = _make_interval_perception(
        regular_states, observations, obs_fn, obs_noise, FAIL, (True, False)
    )

    ipomdp = IPOMDP(states, observations, actions, T, P_lower, P_upper)

    # Perfect-perception shield: include action a from state s iff all
    # reachable next states are safe (conservative 1-step lookahead).
    pp_shield: Dict = {}
    for s in states:
        if s == FAIL or not obstacle_safe(s, N):
            pp_shield[s] = set()
        else:
            safe_acts = set()
            for a in actions:
                dist = T.get((s, a), {})
                if dist and all(obstacle_safe(ns, N) for ns in dist):
                    safe_acts.add(a)
            pp_shield[s] = safe_acts

    initial_states = obstacle_initial_states(N)

    return ipomdp, pp_shield, initial_states, None
