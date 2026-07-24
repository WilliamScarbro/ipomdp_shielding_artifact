"""Data loading utilities for the vendored TaxiNetV2 artifact."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ALPHA_LEVEL = "0.95"

SIGNED_CTE_STATES = [-2, -1, 0, 1, 2]
SIGNED_HE_STATES = [-1, 0, 1]

_CTE_INDEX_TO_SIGNED = {idx: val for idx, val in enumerate(SIGNED_CTE_STATES)}
_HE_INDEX_TO_SIGNED = {idx: val for idx, val in enumerate(SIGNED_HE_STATES)}
_ALPHA_TO_SUFFIX = {
    "0.95": "95",
    "0.99": "99",
    "0.995": "995",
}


ConformalAxisObservation = Tuple[int, ...]
ConformalObservation = Tuple[ConformalAxisObservation, ConformalAxisObservation]
PointEstimateObservation = Tuple[int, int]
SignedTaxiState = Tuple[int, int]
ConditionalConformalAxisKey = Tuple[int, int]


@dataclass(frozen=True)
class TaxiNetV2Metadata:
    """Resolved artifact paths and descriptive metadata."""

    artifact_root: str
    compiler_artifact_dir: str
    train_artifact_dir: str
    confidence_level: str
    cte_csv: str
    he_csv: str


@dataclass(frozen=True)
class PairedTaxiNetV2Observation:
    """Row-aligned point and conformal observations from one TaxiNetV2 event."""

    true_state: SignedTaxiState
    point_observation: PointEstimateObservation
    conformal_observation: ConformalObservation


@dataclass(frozen=True)
class AxisPairedTaxiNetV2Observation:
    """One dimension-wise TaxiNetV2 DNN event."""

    true_value: int
    point_value: int
    conformal_observation: ConformalAxisObservation


def _artifact_root() -> Path:
    root = Path(__file__).resolve().parent / "artifacts"
    if not root.exists():
        raise FileNotFoundError(f"TaxiNetV2 artifact not found at {root}")
    return root


def _compiler_artifact_dir() -> Path:
    return _artifact_root() / "compiler" / "lib" / "acc90"


def _train_artifact_dir() -> Path:
    return _artifact_root() / "train" / "models"


def _perception_artifact_dir() -> Path:
    return _artifact_root() / "perception"


def _alpha_suffix(confidence_level: str) -> str:
    if confidence_level not in _ALPHA_TO_SUFFIX:
        supported = ", ".join(sorted(_ALPHA_TO_SUFFIX))
        raise ValueError(
            f"Unsupported confidence_level={confidence_level!r}. Supported: {supported}"
        )
    return _ALPHA_TO_SUFFIX[confidence_level]


def _csv_paths(confidence_level: str) -> Tuple[Path, Path]:
    suffix = _alpha_suffix(confidence_level)
    root = _compiler_artifact_dir()
    cte_path = root / f"real_cte_pred_acc90_conf{suffix}.csv"
    he_path = root / f"real_he_pred_acc90_conf{suffix}.csv"
    if not cte_path.exists() or not he_path.exists():
        raise FileNotFoundError(
            f"Missing TaxiNetV2 conformal CSVs for confidence_level={confidence_level!r}: "
            f"{cte_path}, {he_path}"
        )
    return cte_path, he_path


def _single_estimate_csv_paths() -> Tuple[Path, Path]:
    root = _compiler_artifact_dir()
    cte_path = root / "real_cte_single_pred_acc90.csv"
    he_path = root / "real_he_single_pred_acc90.csv"
    if not cte_path.exists() or not he_path.exists():
        raise FileNotFoundError(
            "Missing TaxiNetV2 single-estimate CSVs. Run "
            "scripts/recreate_taxinet_v2_perception_artifacts.py first: "
            f"{cte_path}, {he_path}"
        )
    return cte_path, he_path


def _paired_axis_csv_paths(confidence_level: str) -> Tuple[Path, Path]:
    suffix = _alpha_suffix(confidence_level)
    root = _perception_artifact_dir()
    cte_path = root / f"taxinet_cte_paired_conformal_conf{suffix}.csv"
    he_path = root / f"taxinet_he_paired_conformal_conf{suffix}.csv"
    if not cte_path.exists() or not he_path.exists():
        raise FileNotFoundError(
            f"Missing TaxiNetV2 paired axis conformal CSVs for "
            f"confidence_level={confidence_level!r}. Run "
            "scripts/recreate_taxinet_v2_perception_artifacts.py first: "
            f"{cte_path}, {he_path}"
        )
    return cte_path, he_path


def get_taxinet_v2_metadata(confidence_level: str = DEFAULT_ALPHA_LEVEL) -> TaxiNetV2Metadata:
    """Return resolved artifact metadata for TaxiNetV2."""
    cte_path, he_path = _csv_paths(confidence_level)
    return TaxiNetV2Metadata(
        artifact_root=str(_artifact_root()),
        compiler_artifact_dir=str(_compiler_artifact_dir()),
        train_artifact_dir=str(_train_artifact_dir()),
        confidence_level=confidence_level,
        cte_csv=str(cte_path),
        he_csv=str(he_path),
    )


def _read_csv_rows(path: Path) -> List[List[int]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        del header
        return [[int(cell) for cell in row] for row in reader if row]


def _read_axis_paired_rows(
    path: Path,
    axis_name: str,
    mapping: Dict[int, int],
) -> List[AxisPairedTaxiNetV2Observation]:
    rows: List[AxisPairedTaxiNetV2Observation] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            f"true_{axis_name}_idx",
            f"true_{axis_name}",
            f"pred_{axis_name}_idx",
            f"pred_{axis_name}",
        }
        required.update(f"{axis_name}{idx}" for idx in mapping)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        for row_idx, row in enumerate(reader):
            true_idx = int(row[f"true_{axis_name}_idx"])
            pred_idx = int(row[f"pred_{axis_name}_idx"])
            true_value = int(row[f"true_{axis_name}"])
            pred_value = int(row[f"pred_{axis_name}"])
            if mapping[true_idx] != true_value:
                raise ValueError(
                    f"{path} row {row_idx} true {axis_name} mismatch: "
                    f"idx {true_idx} maps to {mapping[true_idx]}, row has {true_value}"
                )
            if mapping[pred_idx] != pred_value:
                raise ValueError(
                    f"{path} row {row_idx} pred {axis_name} mismatch: "
                    f"idx {pred_idx} maps to {mapping[pred_idx]}, row has {pred_value}"
                )

            bits = [int(row[f"{axis_name}{idx}"]) for idx in mapping]
            conformal_obs = tuple(mapping[idx] for idx, bit in enumerate(bits) if bit)
            if not conformal_obs:
                raise ValueError(f"{path} row {row_idx} has an empty {axis_name} set")
            if pred_value not in conformal_obs:
                raise ValueError(
                    f"{path} row {row_idx} has pred {pred_value} outside "
                    f"{axis_name} set {conformal_obs}"
                )
            rows.append(
                AxisPairedTaxiNetV2Observation(
                    true_value=true_value,
                    point_value=pred_value,
                    conformal_observation=conformal_obs,
                )
            )
    return rows


def _decode_axis_row(
    row: Sequence[int],
    mapping: Dict[int, int],
) -> Tuple[int, Tuple[int, ...]]:
    true_state = mapping[row[0]]
    predicted = tuple(mapping[idx] for idx, bit in enumerate(row[1:]) if bit)
    return true_state, predicted


def _decode_single_axis_row(
    row: Sequence[int],
    mapping: Dict[int, int],
) -> Tuple[int, int]:
    true_state = mapping[row[0]]
    predicted = mapping[row[1]]
    return true_state, predicted


def _project_axis_prediction(true_state: int, predicted: Sequence[int]) -> int:
    if not predicted:
        return true_state
    if true_state in predicted:
        return true_state
    return min(predicted, key=lambda cand: (abs(cand - true_state), cand))


@lru_cache(maxsize=None)
def _load_paired_axis_rows(
    confidence_level: str,
) -> Tuple[
    List[AxisPairedTaxiNetV2Observation],
    List[AxisPairedTaxiNetV2Observation],
]:
    cte_path, he_path = _paired_axis_csv_paths(confidence_level)
    cte_rows = _read_axis_paired_rows(cte_path, "cte", _CTE_INDEX_TO_SIGNED)
    he_rows = _read_axis_paired_rows(he_path, "he", _HE_INDEX_TO_SIGNED)
    return cte_rows, he_rows


@lru_cache(maxsize=None)
def _load_observation_rows(confidence_level: str) -> List[Tuple[SignedTaxiState, ConformalObservation]]:
    cte_path, he_path = _csv_paths(confidence_level)
    cte_rows = _read_csv_rows(cte_path)
    he_rows = _read_csv_rows(he_path)

    if len(cte_rows) != len(he_rows):
        raise ValueError(
            f"TaxiNetV2 CSV row mismatch for confidence_level={confidence_level!r}: "
            f"{len(cte_rows)} CTE rows vs {len(he_rows)} HE rows"
        )

    paired_rows: List[Tuple[SignedTaxiState, ConformalObservation]] = []
    for cte_row, he_row in zip(cte_rows, he_rows):
        true_cte, cte_obs = _decode_axis_row(cte_row, _CTE_INDEX_TO_SIGNED)
        true_he, he_obs = _decode_axis_row(he_row, _HE_INDEX_TO_SIGNED)
        paired_rows.append(((true_cte, true_he), (cte_obs, he_obs)))

    return paired_rows


@lru_cache(maxsize=None)
def _load_single_estimate_rows() -> List[Tuple[SignedTaxiState, PointEstimateObservation]]:
    cte_path, he_path = _single_estimate_csv_paths()
    cte_rows = _read_csv_rows(cte_path)
    he_rows = _read_csv_rows(he_path)

    if len(cte_rows) != len(he_rows):
        raise ValueError(
            f"TaxiNetV2 single-estimate row mismatch: "
            f"{len(cte_rows)} CTE rows vs {len(he_rows)} HE rows"
        )

    paired_rows: List[Tuple[SignedTaxiState, PointEstimateObservation]] = []
    for cte_row, he_row in zip(cte_rows, he_rows):
        true_cte, cte_obs = _decode_single_axis_row(cte_row, _CTE_INDEX_TO_SIGNED)
        true_he, he_obs = _decode_single_axis_row(he_row, _HE_INDEX_TO_SIGNED)
        paired_rows.append(((true_cte, true_he), (cte_obs, he_obs)))

    return paired_rows


def get_taxinet_v2_observation_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> List[Tuple[SignedTaxiState, ConformalObservation]]:
    """Return empirical TaxiNetV2 observation samples for interval learning."""
    return list(_load_observation_rows(confidence_level))


def get_taxinet_v2_conformal_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> List[Tuple[SignedTaxiState, ConformalObservation]]:
    """Return empirical TaxiNetV2 conformal-set observation samples."""
    return get_taxinet_v2_observation_data(confidence_level)


def get_taxinet_v2_single_estimate_data() -> List[Tuple[SignedTaxiState, PointEstimateObservation]]:
    """Return empirical TaxiNetV2 point-estimate observation samples."""
    return list(_load_single_estimate_rows())


def get_taxinet_v2_paired_observation_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> List[PairedTaxiNetV2Observation]:
    """Return row-aligned point and conformal TaxiNetV2 observations.

    This compatibility view zips the dimension-wise paired axis artifacts by
    row. Runtime experiments should prefer
    ``get_taxinet_v2_axis_paired_conformal_data`` so CTE and HE can be sampled
    independently conditional on their true values.
    """
    cte_rows, he_rows = _load_paired_axis_rows(confidence_level)
    if len(cte_rows) != len(he_rows):
        raise ValueError(
            f"TaxiNetV2 paired axis row mismatch for confidence_level={confidence_level!r}: "
            f"{len(cte_rows)} CTE rows vs {len(he_rows)} HE rows"
        )

    paired_rows: List[PairedTaxiNetV2Observation] = []
    for cte_row, he_row in zip(cte_rows, he_rows):
        paired_rows.append(
            PairedTaxiNetV2Observation(
                true_state=(cte_row.true_value, he_row.true_value),
                point_observation=(cte_row.point_value, he_row.point_value),
                conformal_observation=(
                    cte_row.conformal_observation,
                    he_row.conformal_observation,
                ),
            )
        )

    return paired_rows


def get_taxinet_v2_axis_paired_conformal_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> Tuple[
    List[AxisPairedTaxiNetV2Observation],
    List[AxisPairedTaxiNetV2Observation],
]:
    """Return per-axis paired point/conformal samples with signed values."""
    cte_rows, he_rows = _load_paired_axis_rows(confidence_level)
    return list(cte_rows), list(he_rows)


def get_taxinet_v2_single_axis_data() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return point-estimate samples split into CTE and HE axes."""
    cte_data: List[Tuple[int, int]] = []
    he_data: List[Tuple[int, int]] = []

    for (true_cte, true_he), (cte_obs, he_obs) in _load_single_estimate_rows():
        cte_data.append((true_cte, cte_obs))
        he_data.append((true_he, he_obs))

    return cte_data, he_data


def get_taxinet_v2_conformal_axis_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> Tuple[
    List[Tuple[int, ConformalAxisObservation]],
    List[Tuple[int, ConformalAxisObservation]],
]:
    """Return conformal-set samples split into CTE and HE axes."""
    cte_data: List[Tuple[int, ConformalAxisObservation]] = []
    he_data: List[Tuple[int, ConformalAxisObservation]] = []

    for (true_cte, true_he), (cte_obs, he_obs) in _load_observation_rows(confidence_level):
        cte_data.append((true_cte, cte_obs))
        he_data.append((true_he, he_obs))

    return cte_data, he_data


def get_taxinet_v2_conditional_conformal_axis_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> Tuple[
    Dict[ConditionalConformalAxisKey, List[ConformalAxisObservation]],
    Dict[ConditionalConformalAxisKey, List[ConformalAxisObservation]],
]:
    """Return conformal sets grouped by ``(true_axis_value, point_estimate)``.

    This is the conditional model needed for the two-stage TaxiNetV2 conformal
    semantics:
    1. sample a point estimate from ``P(estimate | true_state)``
    2. sample a conformal set from empirical ``P(set | true_state, estimate)``
    """
    from collections import defaultdict

    cte_model: DefaultDict[ConditionalConformalAxisKey, List[ConformalAxisObservation]] = defaultdict(list)
    he_model: DefaultDict[ConditionalConformalAxisKey, List[ConformalAxisObservation]] = defaultdict(list)

    cte_rows, he_rows = _load_paired_axis_rows(confidence_level)
    for row in cte_rows:
        cte_model[(row.true_value, row.point_value)].append(row.conformal_observation)
    for row in he_rows:
        he_model[(row.true_value, row.point_value)].append(row.conformal_observation)

    return dict(cte_model), dict(he_model)


def get_taxinet_v2_single_test_models() -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Return per-axis point-estimate samples grouped by true signed state."""
    cte_model = {state: [] for state in SIGNED_CTE_STATES}
    he_model = {state: [] for state in SIGNED_HE_STATES}

    for (true_cte, true_he), (cte_obs, he_obs) in _load_single_estimate_rows():
        cte_model[true_cte].append(cte_obs)
        he_model[true_he].append(he_obs)

    return cte_model, he_model


def get_taxinet_v2_projected_test_models(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Project conformal sets to concrete observations for legacy perceptor APIs.

    This is a compatibility adapter only. The primary TaxiNetV2 IPOMDP uses the
    conformal-set observations directly.
    """
    cte_model = {state: [] for state in SIGNED_CTE_STATES}
    he_model = {state: [] for state in SIGNED_HE_STATES}

    for (true_cte, true_he), (cte_obs, he_obs) in _load_observation_rows(confidence_level):
        cte_model[true_cte].append(_project_axis_prediction(true_cte, cte_obs))
        he_model[true_he].append(_project_axis_prediction(true_he, he_obs))

    return cte_model, he_model


def get_scarbro_split_indices(split: str) -> List[int]:
    """Load vendored train/cal/val/test split indices for TaxiNetV2."""
    path = _train_artifact_dir() / f"{split}_indices.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing TaxiNetV2 split file: {path}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading TaxiNetV2 split indices requires torch to be installed."
        ) from exc

    indices = torch.load(path, map_location="cpu")
    return [int(idx) for idx in indices]
