"""Regenerate every paper figure/table from the bundled result JSONs.

This is the fast, deterministic reproduction path: it does not run any Monte Carlo
experiment, it only replots the committed results. Outputs land in ./figures/.

Figure -> source mapping (see README.md):
  Fig. 1  summary_v7_bars.png            main summary, lowest-failure operating point
  Fig. 2  summary_v7_safe_bars.png       main summary, highest-safe operating point
  Fig. 3  pareto_v7_taxinet.png          TaxiNet Pareto (+ pareto_v7_obstacle.png)
  Fig. 4  taxinet_v2_extremal_comparison.png  conformal baseline comparison (+ Table 3)
          inference_timing_summary.pdf   per-step shield inference time
          coarse_taxinet_results.png     LFP envelope coarseness
          perception_variability_taxinet.png  time-varying vs fixed perception
          alpha_sweep_taxinet_{fail,stuck,safe}.png  alpha sensitivity sweep
  Tables 1/2 are printed by evaluation_summary + the two-state LP check.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PY = sys.executable


def _run_module(module: str) -> None:
    print(f"\n=== {module} ===")
    subprocess.run([PY, "-m", module], cwd=ROOT, check=True)


def _run_script(script: str) -> None:
    print(f"\n=== scripts/{script} ===")
    subprocess.run([PY, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Main summary bars (Fig. 1/2), Pareto (Fig. 3), per-case bars, timing table, md.
    _run_module("ipomdp_shielding.experiments.plot_sweep_v7")
    sweep_out = ROOT / "results" / "sweep_v7"
    for name in [
        "summary_v7_bars.png",
        "summary_v7_safe_bars.png",
        "pareto_v7_taxinet.png",
        "pareto_v7_obstacle.png",
        "barchart_v7_taxinet.png",
        "barchart_v7_obstacle.png",
        "barchart_v7_cartpole_lowacc.png",
        "barchart_v7_refuel_v2.png",
    ]:
        src = sweep_out / name
        if src.exists():
            shutil.copy2(src, FIG_DIR / name)

    # 2. Auxiliary main-paper figures.
    _run_script("plot_timing_figure.py")
    _run_script("plot_coarseness_figure.py")
    _run_script("plot_perception_variability_figure.py")

    # 3. Supplement figures + table.
    _run_script("regenerate_alpha_sweep_figures.py")
    _run_script("regenerate_taxinet_v2_extremal_comparison.py")

    # 4. Two-state LP worked example (supplement verification).
    _run_script("check_two_state_lp.py")

    print("\n" + "=" * 70)
    print(f"All paper figures written to {FIG_DIR}")
    print("=" * 70)
    for png in sorted(FIG_DIR.glob("*")):
        print(f"  {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
