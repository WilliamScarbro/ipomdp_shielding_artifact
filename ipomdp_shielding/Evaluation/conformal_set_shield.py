"""Conformal-set TaxiNet shield compatible with cp-control semantics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

FAIL = "FAIL"

CP_CTE_TO_SIGNED = {0: 0, 1: -1, 2: 1, 3: -2, 4: 2}
CP_HE_TO_SIGNED = {0: 0, 1: -1, 2: 1}
CP_ACTION_TO_SIGNED = {0: 0, 1: -1, 2: 1}


def default_tempest_csv_path() -> Path:
    """Resolve the cp-control Tempest shield CSV if present locally."""
    local_artifact = (
        Path(__file__).resolve().parents[1]
        / "CaseStudies"
        / "TaxiNetV2"
        / "artifacts"
        / "compiler"
        / "lib"
        / "temp_extr_dir_3.csv"
    )
    if local_artifact.exists():
        return local_artifact

    cp_control_artifact = Path("/home/dev/cp-control/compiler/lib/temp_extr_dir_3.csv")
    if cp_control_artifact.exists():
        return cp_control_artifact

    return local_artifact


def load_cp_control_tempest_shield(
    path: Optional[Path] = None,
    action_filter: float = 0.7,
) -> Dict[Tuple[int, int], Set[int]]:
    """Load cp-control action probabilities as signed TaxiNet safe-action sets."""
    path = path or default_tempest_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing cp-control Tempest shield CSV: {path}")

    shield: Dict[Tuple[int, int], Set[int]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}
            cte = CP_CTE_TO_SIGNED[int(row["cte_est"])]
            he = CP_HE_TO_SIGNED[int(row["he_est"])]
            allowed = {
                CP_ACTION_TO_SIGNED[action_idx]
                for action_idx in (0, 1, 2)
                if float(row[f"action{action_idx}"]) > action_filter
            }
            shield[(cte, he)] = allowed

    return shield


class ConformalSetIntersectionShield:
    """Filter actions by intersecting cp-control safe actions over a set estimate."""

    def __init__(
        self,
        state_action_shield: Dict[Tuple[int, int], Set[int]],
        all_actions: Sequence[int] = (-1, 0, 1),
        default_action: Optional[int] = None,
    ):
        self.state_action_shield = state_action_shield
        self.all_actions = list(all_actions)
        self.default_action = default_action
        self.stuck_count = 0
        self.error_count = 0
        self._last_allowed = list(self.all_actions)

    @classmethod
    def from_tempest_csv(
        cls,
        path: Optional[Path] = None,
        action_filter: float = 0.7,
        all_actions: Sequence[int] = (-1, 0, 1),
        default_action: Optional[int] = None,
    ) -> "ConformalSetIntersectionShield":
        return cls(
            load_cp_control_tempest_shield(path=path, action_filter=action_filter),
            all_actions=all_actions,
            default_action=default_action,
        )

    @staticmethod
    def _axis_set(axis_observation: Any) -> Set[int]:
        if isinstance(axis_observation, set):
            return set(axis_observation)
        if isinstance(axis_observation, (tuple, list, frozenset)):
            return set(axis_observation)
        return {axis_observation}

    def _default_or_empty(self) -> list[int]:
        self.stuck_count += 1
        self._last_allowed = [] if self.default_action is None else [self.default_action]
        return list(self._last_allowed)

    def _intersect_actions(self, states: Iterable[Tuple[int, int]]) -> Set[int]:
        allowed: Optional[Set[int]] = None
        for state in states:
            state_allowed = self.state_action_shield.get(state, set())
            allowed = set(state_allowed) if allowed is None else allowed & state_allowed
        return allowed if allowed is not None else set()

    def next_actions(self, evidence: Tuple[Any, Any]) -> list[int]:
        obs, _action = evidence
        if obs == FAIL:
            return self._default_or_empty()

        try:
            cte_obs, he_obs = obs
            cte_set = self._axis_set(cte_obs)
            he_set = self._axis_set(he_obs)
            if not cte_set or not he_set:
                return self._default_or_empty()
            candidate_states = [(cte, he) for cte in cte_set for he in he_set]
            allowed = self._intersect_actions(candidate_states)
        except Exception:
            self.error_count += 1
            return self._default_or_empty()

        if not allowed:
            return self._default_or_empty()

        self._last_allowed = [action for action in self.all_actions if action in allowed]
        return list(self._last_allowed)

    def get_action_probs(self):
        return [(action, 1.0 if action in self._last_allowed else 0.0, 0.0) for action in self.all_actions]

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0
        self._last_allowed = list(self.all_actions)

    def initialize(self, _initial_state):
        self.restart()
