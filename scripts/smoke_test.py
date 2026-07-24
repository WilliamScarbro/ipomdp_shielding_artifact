"""Quick end-to-end sanity check for the artifact.

1. Regenerates every paper figure from the bundled result JSONs and asserts the
   key outputs are non-empty (exercises all plotting + bundled data).
2. Runs a tiny real Monte Carlo threshold sweep (2 trials) for cartpole_lowacc
   using the bundled trained agent, to exercise the simulation/shield pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty output at {path}")


def check_figures() -> None:
    print("[smoke] regenerating all paper figures from bundled results ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_paper_figures.py")],
                   cwd=ROOT, check=True)
    for name in [
        "summary_v7_bars.png",
        "summary_v7_safe_bars.png",
        "pareto_v7_taxinet.png",
        "pareto_v7_obstacle.png",
        "inference_timing_summary.pdf",
        "coarse_taxinet_results.png",
        "perception_variability_taxinet.png",
        "alpha_sweep_taxinet_fail.png",
        "taxinet_v2_extremal_comparison.png",
    ]:
        _assert_nonempty(ROOT / "figures" / name)
    print("[smoke] figures OK")


def check_real_experiment() -> None:
    print("[smoke] running a tiny (2-trial) cartpole_lowacc threshold sweep ...")
    from ipomdp_shielding.experiments.run_threshold_sweep import (
        run_sweep_for_case_study, save_sweep,
    )
    params = {
        "num_trials": 2,
        "trial_length": 5,
        "exclude_envelope": True,
        "config_name": "rl_shield_cartpole_lowacc_v7",
    }
    with tempfile.TemporaryDirectory(prefix="ipomdp_smoke_") as tmp:
        tmp_root = Path(tmp)
        shutil.copytree(ROOT / "results", tmp_root / "results")
        import os
        old = Path.cwd()
        try:
            os.chdir(tmp_root)
            sweep, setup, base = run_sweep_for_case_study("cartpole_lowacc", params)
            save_sweep("cartpole_lowacc", sweep, base, params, setup,
                       output_dir="results/sweep_v7/threshold")
            _assert_nonempty(tmp_root / "results/sweep_v7/threshold/cartpole_lowacc_sweep.json")
        finally:
            os.chdir(old)
    print("[smoke] real experiment OK")


def main() -> None:
    check_figures()
    check_real_experiment()
    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
