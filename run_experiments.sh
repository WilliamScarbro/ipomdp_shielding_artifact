#!/usr/bin/env bash
set -euo pipefail

# Entry points for the IPOMDP shielding paper artifact.
#
#   ./run_experiments.sh smoke_test
#       Fast (~2 min) end-to-end check. Regenerates every paper figure/table from
#       the bundled result data and runs a tiny live shielding sweep to exercise
#       the simulation pipeline. Figures are written to ./figures/.
#
#   ./run_experiments.sh reproduce_results
#       Full reproduction (~12 h). Reruns every Monte Carlo experiment from the
#       bundled trained agents + optimized perception realizations (no retraining),
#       overwriting the result JSONs under results/, then regenerates every figure
#       and table into ./figures/.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

reproduce_results() {
    # Main sweeps -> Fig. 1/2/3, Tables 1/2
    "$PYTHON" -m ipomdp_shielding.experiments.run_v7_all_sweeps
    # Per-step shield timing figure
    "$PYTHON" -m ipomdp_shielding.experiments.run_timing_benchmark
    # TaxiNet envelope coarseness figure
    "$PYTHON" -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_final
    # Perception-variability figure
    "$PYTHON" -m ipomdp_shielding.experiments.sweeps.perception_variability_sweep configs.perception_variability_taxinet
    # Alpha sensitivity sweep figures
    "$PYTHON" -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep
    # Conformal baseline -> Fig. 4, Table 3
    "$PYTHON" scripts/recreate_taxinet_v2_single_estimate_csvs.py
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison_conf99
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison_conf995
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_operating_pareto_sweep
    # Replot everything from the freshly computed results
    "$PYTHON" scripts/plot_paper_figures.py
}

case "${1:-help}" in
    smoke_test)        "$PYTHON" scripts/smoke_test.py ;;
    reproduce_results) reproduce_results ;;
    help|*)
        sed -n '4,15p' "$0"
        exit 1
        ;;
esac
