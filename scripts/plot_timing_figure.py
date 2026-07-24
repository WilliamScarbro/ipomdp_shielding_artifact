"""Plot the per-step shield inference-time figure (paper Fig. `inference_timing_summary`).

Reads results/timing_benchmark/shield_timing.json (bundled; regenerable via
`run_experiments.sh reproduce-timing`) and emits a log-scale grouped bar chart of
mean per-step inference time for each shield on each case study.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TIMING_JSON = ROOT / "results" / "timing_benchmark" / "shield_timing.json"
OUT_PATH = ROOT / "figures" / "inference_timing_summary.pdf"

SHIELD_ORDER = ["observation", "single_belief", "carr", "forward_sampling", "envelope"]
SHIELD_LABELS = {
    "observation": "Observation",
    "single_belief": "Single-Belief",
    "carr": "Carr",
    "forward_sampling": "Fwd-Sampling",
    "envelope": "Envelope",
}
SHIELD_COLORS = {
    "observation": "#7f7f7f",
    "single_belief": "#3a73b8",
    "carr": "#9467bd",
    "forward_sampling": "#2aa3a3",
    "envelope": "#e07b3a",
}
CASE_LABELS = {
    "taxinet": "TaxiNet",
    "obstacle": "Obstacle",
    "cartpole_lowacc": "CartPole",
    "cartpole": "CartPole",
    "refuel_v2": "Refuel",
    "refuel": "Refuel",
}


def _mean_seconds(entry) -> float | None:
    """Extract a mean per-step time in seconds from a shield timing entry."""
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for key in ("mean_s", "mean_seconds", "mean", "mean_time_s", "mean_per_step_s"):
            if key in entry and isinstance(entry[key], (int, float)):
                return float(entry[key])
        for key in ("mean_ms", "mean_time_ms"):
            if key in entry and isinstance(entry[key], (int, float)):
                return float(entry[key]) / 1e3
        for key in ("mean_us", "mean_time_us"):
            if key in entry and isinstance(entry[key], (int, float)):
                return float(entry[key]) / 1e6
    return None


def main() -> None:
    with TIMING_JSON.open() as handle:
        data = json.load(handle)

    cases = [c for c in ("taxinet", "obstacle", "cartpole_lowacc", "cartpole",
                         "refuel_v2", "refuel") if c in data]
    # de-duplicate cartpole/refuel variants, keep first present
    seen_labels: set[str] = set()
    ordered_cases = []
    for c in cases:
        lbl = CASE_LABELS.get(c, c)
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        ordered_cases.append(c)
    cases = ordered_cases

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n_shields = len(SHIELD_ORDER)
    x = np.arange(len(cases))
    width = 0.8 / n_shields

    for i, shield in enumerate(SHIELD_ORDER):
        ys = []
        for c in cases:
            secs = _mean_seconds(data.get(c, {}).get(shield))
            ys.append(secs if (secs and secs > 0) else np.nan)
        offsets = x + (i - (n_shields - 1) / 2) * width
        ax.bar(offsets, ys, width, color=SHIELD_COLORS[shield],
               label=SHIELD_LABELS[shield], zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(c, c) for c in cases])
    ax.set_ylabel("Mean per-step inference time (s)")
    ax.set_title("Shield inference time per step (log scale)")
    ax.grid(axis="y", alpha=0.3, which="both", zorder=0)
    ax.legend(fontsize=8, ncol=n_shields, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
