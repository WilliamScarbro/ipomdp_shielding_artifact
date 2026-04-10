"""Experiment I/O utilities for reproducible metadata and standardized output.

Provides helpers to:
- Collect experiment metadata (git SHA, timestamp, machine info, full config)
- Compute binomial confidence intervals
- Save results in a standardized format
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def get_git_sha() -> Optional[str]:
    """Get current git SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_machine_info() -> Dict[str, str]:
    """Collect basic machine info."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }


def build_metadata(config: Any, extra: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a metadata dict from a config object.

    Includes timestamp, git SHA, machine info, and the full config
    serialized as a dict.

    Parameters
    ----------
    config : dataclass or object
        Experiment config. If it has __dict__, it will be serialized.
        Callable fields are stored by their qualified name.
    extra : dict, optional
        Additional metadata to merge in (e.g., cache_hit info).
    """
    # Serialize config
    config_dict = _serialize_config(config)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "machine": get_machine_info(),
        "config": config_dict,
    }
    if extra:
        meta.update(extra)
    return meta


def _serialize_config(config: Any) -> Dict[str, Any]:
    """Serialize a config object to a JSON-compatible dict."""
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        try:
            return asdict(config)
        except Exception:
            pass

    d = {}
    for key, val in (config.__dict__ if hasattr(config, "__dict__") else {}).items():
        d[key] = _serialize_value(val)
    return d


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if callable(val):
        return f"{val.__module__}.{val.__qualname__}" if hasattr(val, "__qualname__") else str(val)
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if hasattr(val, "name"):  # enums
        return val.name
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return str(val)


def clopper_pearson_ci(
    successes: int, trials: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute Clopper-Pearson exact binomial confidence interval.

    Parameters
    ----------
    successes : number of successes
    trials : total number of trials
    alpha : significance level (default 0.05 for 95% CI)

    Returns
    -------
    (lower, upper) : bounds of the CI
    """
    from scipy import stats

    if trials == 0:
        return (0.0, 1.0)
    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    if successes == trials:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (float(lower), float(upper))


def add_rate_cis(
    results_dict: Dict[str, Any],
    num_trials: int,
    ci_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Add confidence intervals to a results dict that has fail_rate, stuck_rate, safe_rate.

    Modifies results_dict in place and returns it.
    """
    for rate_key in ["fail_rate", "stuck_rate", "safe_rate"]:
        if rate_key in results_dict:
            rate = results_dict[rate_key]
            successes = round(rate * num_trials)
            lo, hi = clopper_pearson_ci(successes, num_trials, ci_alpha)
            results_dict[f"{rate_key}_ci_low"] = lo
            results_dict[f"{rate_key}_ci_high"] = hi
    return results_dict


def save_experiment_results(
    path: str,
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    tidy_rows: Optional[List[Dict]] = None,
) -> None:
    """Save experiment results to JSON (and optionally CSV).

    Parameters
    ----------
    path : str
        Path for the main JSON results file.
    results : dict
        The experiment results.
    metadata : dict
        Metadata from build_metadata().
    tidy_rows : list of dicts, optional
        If provided, also write a results_tidy.csv next to the JSON.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    output = {
        "metadata": metadata,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if tidy_rows:
        import csv
        csv_path = path.replace(".json", "_tidy.csv")
        if tidy_rows:
            # Rows may not share identical keys (e.g., sanity-check rows add notes).
            # Use the union of keys to avoid DictWriter errors.
            all_keys = set()
            for row in tidy_rows:
                all_keys.update(row.keys())
            fieldnames = sorted(all_keys)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(tidy_rows)
