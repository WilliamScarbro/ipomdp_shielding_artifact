"""Standard (non-interval) Partially Observable Markov Decision Process."""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable, Iterable
from collections import defaultdict

State = Hashable
Action = Hashable
Observation = Hashable


@dataclass
class POMDP:
    """
    Standard Partially Observable Markov Decision Process.

    states       : list of states
    observations : list of possible observations
    actions      : set of actions
    T            : (s, a) -> {s' -> P(s' | s, a)}
    P            : s -> {o -> P(o | s)}
    """
    states: List[State]
    observations: List[Observation]
    actions: List[Action]
    T: Dict[Tuple[State, Action], Dict[State, float]]
    P: Dict[State, Dict[Observation, float]]


def expected_perception_from_data(
    states: Iterable[State],
    observations: Iterable[Observation],
    data: Iterable[Tuple[State, Observation]]
) -> Dict[State, Dict[Observation, float]]:
    """Estimate observation probabilities from data."""
    states = list(states)
    observations = list(observations)

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for s, o in data:
        counts[s][o] += 1
        totals[s] += 1

    result = {}
    for s in states:
        result[s] = {}
        total = totals[s]
        for o in observations:
            if total > 0:
                result[s][o] = counts[s][o] / total
            else:
                result[s][o] = 1.0 / len(observations)

    return result


def product_model(
    model1: Dict[State, Dict[Observation, float]],
    model2: Dict[State, Dict[Observation, float]]
) -> Dict[Tuple[State, State], Dict[Tuple[Observation, Observation], float]]:
    """Compute product of two perception models."""
    result = {}
    for s1 in model1:
        for s2 in model2:
            result[(s1, s2)] = {}
            for o1 in model1[s1]:
                for o2 in model2[s2]:
                    result[(s1, s2)][(o1, o2)] = model1[s1][o1] * model2[s2][o2]
    return result


class POMDP_Belief:
    """Belief tracker for standard POMDP."""

    def __init__(self, pomdp: POMDP):
        self.pomdp = pomdp
        self.restart()

    def restart(self):
        """Reset to uniform prior."""
        n_states = len(self.pomdp.states)
        self.belief = {s: 1.0 / n_states for s in self.pomdp.states}

    def propogate(self, evidence: Tuple[Observation, Action]):
        """Update belief with observation and action."""
        o_t, a_t = evidence

        # Prediction step
        next_belief = {s_next: 0.0 for s_next in self.pomdp.states}
        for s in self.pomdp.states:
            if self.belief[s] == 0.0:
                continue
            trans = self.pomdp.T.get((s, a_t), {})
            for s_next, p_trans in trans.items():
                if p_trans > 0:
                    next_belief[s_next] += p_trans * self.belief[s]

        # Observation update
        alpha = {}
        denom = 0.0
        for s_next in self.pomdp.states:
            z = self.pomdp.P[s_next].get(o_t, 0.0)
            alpha[s_next] = next_belief[s_next] * z
            denom += alpha[s_next]

        if denom > 0:
            self.belief = {s: alpha[s] / denom for s in self.pomdp.states}
        else:
            self.belief = next_belief

    def allowed_probability(self, allowed: Iterable[State]) -> float:
        """Return probability mass in allowed states."""
        allowed_set = set(allowed)
        return sum(self.belief.get(s, 0.0) for s in allowed_set)
