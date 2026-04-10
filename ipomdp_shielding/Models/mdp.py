"""Markov Decision Process model."""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

State = Hashable
Action = Hashable


@dataclass
class MDP:
    """
    Markov Decision Process.

    states: list of all states
    actions: mapping from state -> list of enabled actions
    P: mapping (s, a) -> {s' -> P(s' | s, a)}
    """
    states: List[State]
    actions: Dict[State, List[Action]]
    P: Dict[Tuple[State, Action], Dict[State, float]]
