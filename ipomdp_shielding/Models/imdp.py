"""Interval Markov Decision Process model and operations."""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable, Iterable, Set, Callable

from .mdp import MDP

State = Hashable
Action = Hashable


@dataclass
class IMDP:
    """
    Interval Markov Decision Process.

    states: list of all states
    actions: mapping from state -> list of enabled actions
    P_lower: mapping (s, a) -> {s' -> lower bound on P(s' | s, a)}
    P_upper: mapping (s, a) -> {s' -> upper bound on P(s' | s, a)}

    We assume for each (s,a):
      sum_s' P_lower[(s,a)][s'] <= 1 <= sum_s' P_upper[(s,a)][s']
    """
    states: List[State]
    actions: Dict[State, List[Action]]
    P_lower: Dict[Tuple[State, Action], Dict[State, float]]
    P_upper: Dict[Tuple[State, Action], Dict[State, float]]


def imdp_check_lower_upper_consist(imdp: IMDP) -> bool:
    """Check that lower and upper bounds have consistent keys."""
    return imdp.P_lower.keys() == imdp.P_upper.keys()


def imdp_from_mdp(mdp: MDP) -> IMDP:
    """Convert an MDP to an IMDP with exact probabilities (lower = upper)."""
    return IMDP(mdp.states, mdp.actions, mdp.P.copy(), mdp.P.copy())


def product_imdp(fst: IMDP, snd: IMDP) -> IMDP:
    """
    Compute the product of two IMDPs.
    Assumes same and global actions.
    """
    states = [(s1, s2) for s1 in fst.states for s2 in snd.states]
    actions = {s: fst.actions[s[0]] for s in states}

    lower = {}
    upper = {}
    for sl, sr in states:
        for a in actions[(sl, sr)]:
            s2_lower = {}
            s2_upper = {}
            for s2l in fst.P_lower[(sl, a)]:
                for s2r in snd.P_lower[(sr, a)]:
                    s2_lower[(s2l, s2r)] = \
                        fst.P_lower[(sl, a)][s2l] * snd.P_lower[(sr, a)][s2r]
                    s2_upper[(s2l, s2r)] = \
                        fst.P_upper[(sl, a)][s2l] * snd.P_upper[(sr, a)][s2r]
            lower[((sl, sr), a)] = s2_lower
            upper[((sl, sr), a)] = s2_upper

    return IMDP(states, actions, lower, upper)


def collapse_imdp(
    imdp: IMDP,
    new_states: Iterable[State],
    new_actions: Dict[State, List[Action]],
    collapse_fn: Callable[[State], State],
) -> IMDP:
    """
    Collapse an IMDP over old states S into an IMDP over new states S_hat.

    collapse_fn : S_old -> S_new

    For each abstract state X in S_new, let block(X) = { s in S_old | collapse_fn(s) = X }.

    For each (s in block(X), a) we define collapsed probabilities from X to Y by:
      lower_s(X->Y | a) = sum_{t : collapse_fn(t) = Y} lower_old(s->t | a)
      upper_s(X->Y | a) = sum_{t : collapse_fn(t) = Y} upper_old(s->t | a)

    Then the IMDP intervals are:
      lower_new(X->Y | a) = min_{s in block(X)} lower_s(X->Y | a)
      upper_new(X->Y | a) = max_{s in block(X)} upper_s(X->Y | a)
    """
    new_states_set: Set[State] = set(new_states)
    if not new_states_set:
        raise ValueError("new_states must be non-empty.")

    # Precompute old -> new mapping and build blocks
    old_to_new: Dict[State, State] = {}
    blocks: Dict[State, List[State]] = {X: [] for X in new_states_set}

    for s in imdp.states:
        X = collapse_fn(s)
        if X not in new_states_set:
            raise ValueError(
                f"collapse_fn(s)={X!r} not in provided new_states for old state {s!r}."
            )
        old_to_new[s] = X
        blocks[X].append(s)

    # Build new P_lower and P_upper
    P_lower_new: Dict[Tuple[State, Action], Dict[State, float]] = {}
    P_upper_new: Dict[Tuple[State, Action], Dict[State, float]] = {}

    for X, concrete_states in blocks.items():
        for a in new_actions[X]:
            min_prob: Dict[State, float] = {Y: float("inf") for Y in new_states_set}
            max_prob: Dict[State, float] = {Y: 0.0 for Y in new_states_set}

            for s in concrete_states:
                lower_sa = imdp.P_lower.get((s, a), {})
                upper_sa = imdp.P_upper.get((s, a), {})

                collapsed_lower_s = {Y: 0.0 for Y in new_states_set}
                collapsed_upper_s = {Y: 0.0 for Y in new_states_set}

                for t, p_low in lower_sa.items():
                    Y = old_to_new.get(t, None)
                    if Y is None:
                        raise Exception("unknown old state ", t)
                    collapsed_lower_s[Y] += p_low

                for t, p_up in upper_sa.items():
                    Y = old_to_new.get(t, None)
                    if Y is None:
                        continue
                    collapsed_upper_s[Y] += p_up

                for Y in new_states_set:
                    p_low = collapsed_lower_s[Y]
                    p_up = collapsed_upper_s[Y]
                    if p_low < min_prob[Y]:
                        min_prob[Y] = p_low
                    if p_up > max_prob[Y]:
                        max_prob[Y] = p_up

            for Y in new_states_set:
                if min_prob[Y] == float("inf"):
                    min_prob[Y] = 0.0

            P_lower_new[(X, a)] = dict(min_prob)
            P_upper_new[(X, a)] = dict(max_prob)

    return IMDP(
        states=list(new_states_set),
        actions=new_actions,
        P_lower=P_lower_new,
        P_upper=P_upper_new,
    )


def imdp_interval_width_dist(imdp: IMDP, disc: int = 100):
    """Compute distribution of interval widths in the IMDP."""
    widths = []
    for s in imdp.states:
        for a in imdp.actions[s]:
            for s2 in imdp.P_lower[(s, a)]:
                widths.append(imdp.P_upper[(s, a)][s2] - imdp.P_lower[(s, a)][s2])

    counts = []
    xs = [1 / disc * i for i in range(disc)]
    for i in range(disc):
        counts.append(0)
        for w in widths:
            if xs[i] - 0.00001 < w < xs[i] + 1 / disc + 0.00001:
                counts[i] += 1
    return xs, counts
