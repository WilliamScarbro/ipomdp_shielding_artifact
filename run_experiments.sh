#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

case "${1:-help}" in
    smoke)
        "$PYTHON" scripts/smoke_test.py
        ;;
    plot)
        "$PYTHON" -m ipomdp_shielding.experiments.plot_experiment_summary
        ;;
    reproduce)
        "$PYTHON" -m ipomdp_shielding.experiments.run_all_sweeps
        "$PYTHON" -m ipomdp_shielding.experiments.plot_experiment_summary
        ;;
    help|*)
        echo "Usage: $0 {smoke|plot|reproduce}"
        exit 1
        ;;
esac
