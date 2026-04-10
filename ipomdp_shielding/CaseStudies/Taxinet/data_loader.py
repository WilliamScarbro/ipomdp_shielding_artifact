"""Data loading utilities for TaxiNet case study."""

import os
from pathlib import Path


def _find_data_dir():
    """Find the artifacts directory containing data files."""
    # Try common locations
    candidates = [
        Path(__file__).parent / "artifacts",
        Path("./artifacts"),
        Path("../artifacts"),
        Path(__file__).parent.parent.parent.parent / "src" / "artifacts",
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path("./artifacts")  # Default fallback

def _data_dir():
    return Path(__file__).parent / "./artifacts"


def get_cte_data(data_dir=None):
    """
    Load cross-track error observation data.

    Returns list of (true_cte, estimated_cte) tuples.
    """
    if data_dir is None:
        data_dir = _data_dir()

    cte_file = Path(data_dir) / "cte_totals"

    with open(cte_file) as f:
        vals = f.read().split("\n")

    out = []
    for cte in range(-2, 3):
        for cte_est in range(-2, 3):
            reps = int(vals[(cte + 2) * 5 + cte_est + 2])
            for _ in range(reps):
                out.append((cte, cte_est))
    return out


def get_he_data(data_dir=None):
    """
    Load heading error observation data.

    Returns list of (true_he, estimated_he) tuples.
    """
    if data_dir is None:
        data_dir = _data_dir()

    he_file = Path(data_dir) / "he_totals"

    with open(he_file) as f:
        vals = f.read().split("\n")

    out = []
    for he in range(-1, 2):
        for he_est in range(-1, 2):
            reps = int(vals[(he + 1) * 3 + he_est + 1])
            for _ in range(reps):
                out.append((he, he_est))
    return out
