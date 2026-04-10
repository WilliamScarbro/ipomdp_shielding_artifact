"""Script library for reproducible evaluation runs.

Generates and stores pre-computed trajectories using perfect perception shielding,
enabling fair comparisons between templates on identical run sequences.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Callable, Iterable
import random
import json

from ..Models import IPOMDP


@dataclass
class RunScript:
    """A pre-recorded trajectory of states, observations, and actions.

    Each step contains (state, obs, action) where:
    - state: the true underlying state
    - obs: the observation generated from that state
    - action: the action selected by perfect perception shield
    """
    initial: Tuple[Any, Any]  # (initial_state, initial_action)
    steps: List[Tuple[Any, Any, Any]]  # [(state, obs, action), ...]
    metadata: Dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "initial": list(self.initial),
            "steps": [list(s) for s in self.steps],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RunScript":
        """Reconstruct from dictionary."""
        return cls(
            initial=tuple(d["initial"]),
            steps=[tuple(s) for s in d["steps"]],
            metadata=d.get("metadata", {})
        )


def generate_script(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, set],
    perception: Callable[[Any], Any],
    initial: Tuple[Any, Any],
    length: int,
    action_selector: Optional[Callable[[Iterable], Any]] = None
) -> RunScript:
    """Generate a single run script using perfect perception shielding.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    perception : callable
        Function mapping state to observation
    initial : tuple
        (initial_state, initial_action) pair
    length : int
        Number of steps to generate
    action_selector : callable, optional
        Function to select from safe actions. Defaults to random choice.

    Returns
    -------
    RunScript
        The generated trajectory
    """
    def default_selector(actions):
        action_list = list(actions) if not isinstance(actions, list) else actions
        return random.choice(action_list) if action_list else None

    selector = action_selector if action_selector is not None else default_selector

    state, action = initial
    steps = []

    for _ in range(length):
        if state == "FAIL":
            break

        obs = perception(state)

        # Select action using perfect perception (true state)
        safe_actions = pp_shield.get(state, set())
        if safe_actions:
            action = selector(safe_actions)
        else:
            # No safe actions - use any available action
            action = selector(list(ipomdp.actions))

        steps.append((state, obs, action))

        # Evolve state
        state = ipomdp.evolve(state, action)

    return RunScript(initial=initial, steps=steps)


@dataclass
class ScriptLibrary:
    """Collection of pre-generated run scripts for reproducible evaluation."""

    scripts: List[RunScript] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scripts)

    def __getitem__(self, idx: int) -> RunScript:
        return self.scripts[idx]

    def __iter__(self):
        return iter(self.scripts)

    def add(self, script: RunScript):
        """Add a script to the library."""
        self.scripts.append(script)

    @classmethod
    def generate(
        cls,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, set],
        perception: Callable[[Any], Any],
        initial: Tuple[Any, Any],
        num_scripts: int,
        length: int,
        action_selector: Optional[Callable[[Iterable], Any]] = None,
        seed: Optional[int] = None
    ) -> "ScriptLibrary":
        """Generate a library of scripts.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        perception : callable
            Function mapping state to observation
        initial : tuple
            (initial_state, initial_action) pair
        num_scripts : int
            Number of scripts to generate
        length : int
            Length of each script
        action_selector : callable, optional
            Function to select from safe actions (takes an iterable)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        ScriptLibrary
            The generated library
        """
        if seed is not None:
            random.seed(seed)

        library = cls(metadata={
            "num_scripts": num_scripts,
            "length": length,
            "seed": seed
        })

        for i in range(num_scripts):
            script = generate_script(
                ipomdp, pp_shield, perception, initial, length, action_selector
            )
            script.metadata["script_id"] = i
            library.add(script)

        return library

    def save(self, path: str):
        """Save library to JSON file."""
        data = {
            "metadata": self.metadata,
            "scripts": [s.to_dict() for s in self.scripts]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ScriptLibrary":
        """Load library from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        library = cls(metadata=data.get("metadata", {}))
        for script_data in data["scripts"]:
            library.add(RunScript.from_dict(script_data))

        return library

    def filter_by_length(self, min_length: int) -> "ScriptLibrary":
        """Return new library with only scripts of sufficient length."""
        filtered = ScriptLibrary(metadata=self.metadata.copy())
        for script in self.scripts:
            if script.length >= min_length:
                filtered.add(script)
        return filtered

