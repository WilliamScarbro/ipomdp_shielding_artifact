"""Data loading utilities for CartPole case study.

Loads confusion matrices and bin edges from the artifacts/ directory.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def _data_dir():
    """Return the path to the artifacts/ data directory."""
    return Path(__file__).parent / "artifacts"


def get_bin_edges(data_dir: Optional[Path] = None) -> np.ndarray:
    """Load bin edges for state discretization.

    Args:
        data_dir: Optional path to data directory. If None, uses default artifacts/ directory.

    Returns:
        np.ndarray of shape (4,) where each element is an array of bin edges for that dimension:
        - edges[0]: x position bin edges (length n_x + 1)
        - edges[1]: x velocity bin edges (length n_xdot + 1)
        - edges[2]: theta (angle) bin edges (length n_theta + 1)
        - edges[3]: theta_dot (angular velocity) bin edges (length n_thetadot + 1)

        Note: For uniform discretization with k bins, edges[i] has length k+1.
        For non-uniform discretization, each dimension can have different numbers of bins.
    """
    if data_dir is None:
        data_dir = _data_dir()

    with open(data_dir / "bin_edges.npy", "rb") as f:
        return np.load(f, allow_pickle=True)


def get_confusion_data(dimension: str, data_dir: Optional[Path] = None) -> List[Tuple[int, int]]:
    """Load perception data for one state dimension.

    Args:
        dimension: One of "x", "x_dot", "theta", "theta_dot"
        data_dir: Optional path to data directory. If None, uses default artifacts/ directory.

    Returns:
        List of (true_bin, estimated_bin) tuples representing the confusion data.
        Each tuple appears once per observation in the dataset.
    """
    if data_dir is None:
        data_dir = _data_dir()

    valid_dims = ["x", "x_dot", "theta", "theta_dot"]
    if dimension not in valid_dims:
        raise ValueError(f"dimension must be one of {valid_dims}, got {dimension}")

    confusion_matrix = np.load(data_dir / f"{dimension}_confusion.npy")

    # Unpack confusion matrix to list of (true_bin, estimated_bin) tuples
    data = []
    for true_bin in range(confusion_matrix.shape[0]):
        for est_bin in range(confusion_matrix.shape[1]):
            count = int(confusion_matrix[true_bin, est_bin])
            data.extend([(true_bin, est_bin)] * count)

    return data
