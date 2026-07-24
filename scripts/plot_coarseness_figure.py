"""Plot the TaxiNet LFP-envelope coarseness figure (paper `coarse_taxinet_results`).

Reads results/final/coarse_taxinet_results.json (bundled; regenerable via
`run_experiments.sh reproduce-coarse`) and plots the per-step mean max-gap with a
p10-p90 percentile band, plus the per-step mean gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "final" / "coarse_taxinet_results.json"
OUT_PATH = ROOT / "figures" / "coarse_taxinet_results.png"


def main() -> None:
    with RESULTS_JSON.open() as handle:
        summary = json.load(handle)["results"]

    ts_max = summary["timestep_avg_max_gap"]
    ts_mean = summary["timestep_avg_mean_gap"]
    p10 = summary.get("timestep_max_gap_p10")
    p90 = summary.get("timestep_max_gap_p90")
    timesteps = list(range(len(ts_max)))

    fig, ax = plt.subplots(figsize=(8, 5))
    if p10 and p90:
        ax.fill_between(timesteps, p10, p90, alpha=0.15, color="steelblue",
                        label="Max gap p10-p90")
    ax.plot(timesteps, ts_max, "o-", color="steelblue",
            label="Mean max gap (avg over trajectories)")
    ax.plot(timesteps, ts_mean, "s--", color="firebrick", alpha=0.8,
            label="Mean (per-action) gap")

    g = summary["overall_max_gap"]
    mg = summary["overall_mean_gap"]
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Coarseness gap (sampled under-approx - LFP envelope)")
    ax.set_title(
        "TaxiNet LFP Envelope Coarseness Over Time\n"
        f"overall max-gap mean={g['mean']:.3f} median={g['median']:.3f} "
        f"p90={g['p90']:.3f}; mean-gap mean={mg['mean']:.3f}"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
