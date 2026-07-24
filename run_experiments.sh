#!/usr/bin/env bash
set -euo pipefail

# Entry point for the IPOMDP shielding paper artifact.
#
#   ./run_experiments.sh figures       Regenerate every paper figure/table from the
#                                       bundled result JSONs (fast, deterministic).
#   ./run_experiments.sh smoke         Quick end-to-end sanity check.
#
#   Reproduce underlying experiments from scratch (long; uses bundled trained
#   agents + optimized realizations under results/cache/, so no retraining):
#   ./run_experiments.sh reproduce-main        Main sweeps  -> Fig. 1/2/3, Tables 1/2 (~8 h)
#   ./run_experiments.sh reproduce-timing      Per-step shield timing figure
#   ./run_experiments.sh reproduce-coarse      TaxiNet coarseness figure
#   ./run_experiments.sh reproduce-perception  Perception-variability figure
#   ./run_experiments.sh reproduce-alpha       Alpha sensitivity sweep figures
#   ./run_experiments.sh reproduce-conformal   Conformal baseline (Fig. 4, Table 3)
#   ./run_experiments.sh reproduce-all         Everything above, then figures

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

reproduce_main() {
    "$PYTHON" -m ipomdp_shielding.experiments.run_v7_all_sweeps
}
reproduce_timing() {
    "$PYTHON" -m ipomdp_shielding.experiments.run_timing_benchmark
}
reproduce_coarse() {
    "$PYTHON" -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_final
}
reproduce_perception() {
    "$PYTHON" -m ipomdp_shielding.experiments.sweeps.perception_variability_sweep configs.perception_variability_taxinet
}
reproduce_alpha() {
    "$PYTHON" -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep
}
reproduce_conformal() {
    "$PYTHON" scripts/recreate_taxinet_v2_single_estimate_csvs.py
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison_conf99
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_comparison --config rl_shield_taxinet_v2_comparison_conf995
    "$PYTHON" -m ipomdp_shielding.experiments.run_taxinet_v2_operating_pareto_sweep
}

case "${1:-help}" in
    figures)              "$PYTHON" scripts/plot_paper_figures.py ;;
    smoke)                "$PYTHON" scripts/smoke_test.py ;;
    reproduce-main)       reproduce_main ;;
    reproduce-timing)     reproduce_timing ;;
    reproduce-coarse)     reproduce_coarse ;;
    reproduce-perception) reproduce_perception ;;
    reproduce-alpha)      reproduce_alpha ;;
    reproduce-conformal)  reproduce_conformal ;;
    reproduce-all)
        reproduce_main
        reproduce_timing
        reproduce_coarse
        reproduce_perception
        reproduce_alpha
        reproduce_conformal
        "$PYTHON" scripts/plot_paper_figures.py
        ;;
    help|*)
        sed -n '3,25p' "$0"
        exit 1
        ;;
esac
