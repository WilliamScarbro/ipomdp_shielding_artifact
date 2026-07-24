"""Plot the TaxiNet perception-variability figure (paper `perception_variability_taxinet`).

Reads results/final/perception_variability/perception_variability_taxinet_results.json
(bundled; regenerable via `run_experiments.sh reproduce-perception`) and reuses the
sweep module's overlay plotter to compare the LFP envelope against the time-varying and
fixed-probability forward-sampled under-approximations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from ipomdp_shielding.experiments.sweeps.perception_variability_sweep import _plot_overlay


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = (
    ROOT / "results" / "final" / "perception_variability"
    / "perception_variability_taxinet_results.json"
)
OUT_PATH = ROOT / "figures" / "perception_variability_taxinet.png"


def main() -> None:
    with RESULTS_JSON.open() as handle:
        payload = json.load(handle)
    per_mode = payload["results"]
    cfg = payload["metadata"]["config"]
    config = SimpleNamespace(
        case_study_name=cfg.get("case_study_name", "taxinet"),
        sampler_budget=cfg.get("sampler_budget", 500),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _plot_overlay(per_mode, config, str(OUT_PATH))


if __name__ == "__main__":
    main()
