"""Perception variability coarseness sweep.

Three-way coarseness comparison on shared trajectories:
  * LFP envelope (over-approximation)
  * Forward-sampled with varying-per-step likelihoods (existing)
  * Forward-sampled with fixed-per-trajectory full P(o|s) realizations (new)

For each timestep we record the per-trajectory max safe gap to LFP
under each mode and aggregate across trajectories. Output is a tidy
CSV (one row per timestep per mode) plus a summary JSON and an
overlay plot.

Usage:
    python -m ipomdp_shielding.experiments.sweeps.perception_variability_sweep \
        configs.perception_variability_taxinet
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Dict, List

import numpy as np

from ..experiment_io import build_metadata, save_experiment_results
from ..run_coarse_experiment import (
    create_lfp_propagator,
    create_sampler,
    generate_trajectories,
)
from ...Evaluation.coarseness_evaluator import (
    CoarsenessEvaluator,
    CoarsenessReport,
)
from ...Propagators import FixedRealizationSampledBelief


# --------------------------------------------------------------------
# Per-mode aggregation
# --------------------------------------------------------------------

def _max_safe_gap_series(report: CoarsenessReport, mode: str) -> List[float]:
    """Per-timestep max safe gap for a given mode ('varying' or 'fixed_realization')."""
    if mode == "varying":
        return [s.max_safe_gap for s in report.snapshots]
    return [s.max_safe_gap_for(mode) for s in report.snapshots]


def _mean_safe_gap_series(report: CoarsenessReport, mode: str) -> List[float]:
    if mode == "varying":
        return [s.mean_safe_gap for s in report.snapshots]
    return [s.mean_safe_gap_for(mode) for s in report.snapshots]


def _overall_max(report: CoarsenessReport, mode: str) -> float:
    series = _max_safe_gap_series(report, mode)
    return max(series) if series else 0.0


def _overall_mean(report: CoarsenessReport, mode: str) -> float:
    series = _mean_safe_gap_series(report, mode)
    return float(np.mean(series)) if series else 0.0


def aggregate_per_mode(reports: List[CoarsenessReport], mode: str) -> Dict:
    """Compute per-timestep + overall summaries for one mode."""
    all_max = [_overall_max(r, mode) for r in reports]
    all_mean = [_overall_mean(r, mode) for r in reports]

    max_len = max((len(r.snapshots) for r in reports), default=0)
    per_step = []
    for t in range(max_len):
        max_vals = []
        mean_vals = []
        for r in reports:
            if t < len(r.snapshots):
                if mode == "varying":
                    max_vals.append(r.snapshots[t].max_safe_gap)
                    mean_vals.append(r.snapshots[t].mean_safe_gap)
                else:
                    max_vals.append(r.snapshots[t].max_safe_gap_for(mode))
                    mean_vals.append(r.snapshots[t].mean_safe_gap_for(mode))
        per_step.append({
            "t": t,
            "mean_max_gap": float(np.mean(max_vals)) if max_vals else 0.0,
            "median_max_gap": float(np.median(max_vals)) if max_vals else 0.0,
            "p10_max_gap": float(np.percentile(max_vals, 10)) if max_vals else 0.0,
            "p90_max_gap": float(np.percentile(max_vals, 90)) if max_vals else 0.0,
            "mean_mean_gap": float(np.mean(mean_vals)) if mean_vals else 0.0,
        })

    overall_max = {
        "mean": float(np.mean(all_max)) if all_max else 0.0,
        "std": float(np.std(all_max)) if all_max else 0.0,
        "median": float(np.median(all_max)) if all_max else 0.0,
        "p10": float(np.percentile(all_max, 10)) if all_max else 0.0,
        "p90": float(np.percentile(all_max, 90)) if all_max else 0.0,
    }
    overall_mean_gap = {
        "mean": float(np.mean(all_mean)) if all_mean else 0.0,
        "std": float(np.std(all_mean)) if all_mean else 0.0,
        "median": float(np.median(all_mean)) if all_mean else 0.0,
    }
    return {
        "num_trajectories": len(reports),
        "overall_max_gap": overall_max,
        "overall_mean_gap": overall_mean_gap,
        "timestep": per_step,
    }


# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------

def _plot_overlay(per_mode: Dict[str, Dict], config, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"varying": "steelblue", "fixed_realization": "firebrick"}
    labels = {"varying": "Varying per step",
              "fixed_realization": "Fixed per trajectory"}

    for mode, summary in per_mode.items():
        ts = summary["timestep"]
        xs = [row["t"] for row in ts]
        max_means = [row["mean_max_gap"] for row in ts]
        mean_means = [row["mean_mean_gap"] for row in ts]
        p10 = [row["p10_max_gap"] for row in ts]
        p90 = [row["p90_max_gap"] for row in ts]
        c = colors.get(mode, "gray")
        ax.fill_between(xs, p10, p90, alpha=0.15, color=c,
                        label=f"{labels.get(mode, mode)} — max gap p10–p90")
        ax.plot(xs, max_means, "o-", color=c,
                label=f"{labels.get(mode, mode)} — mean max gap")
        ax.plot(xs, mean_means, "s--", color=c, alpha=0.7,
                label=f"{labels.get(mode, mode)} — mean (per-action) gap")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Coarseness gap (sampled under-approx − LFP envelope)")
    K = config.sampler_budget
    ax.set_title(
        f"LFP Envelope vs. Forward-Sampled Under-Approximations ({config.case_study_name.upper()})\n"
        f"K={K}, n={per_mode['varying']['num_trajectories']} trajectories"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ipomdp_shielding.experiments.sweeps."
            "perception_variability_sweep <config_module>"
        )
        sys.exit(1)

    config_module_name = sys.argv[1]
    config_module = importlib.import_module(
        f".{config_module_name}", package="ipomdp_shielding.experiments"
    )
    config = config_module.config

    print("=" * 70)
    print(f"PERCEPTION VARIABILITY SWEEP: {config.case_study_name.upper()}")
    print(f"Trajectories: {config.num_trajectories}, "
          f"Length: {config.trajectory_length}, Seed: {config.seed}")
    print(f"Sampler budget (K): {config.sampler_budget}, "
          f"varying K_samples: {config.sampler_k}")
    print("=" * 70)

    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(
        seed=config.seed, **config.ipomdp_kwargs
    )
    print(f"States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    print("\nGenerating shared trajectories...")
    library = generate_trajectories(ipomdp, pp_shield, config, seed=config.seed)

    reports: List[CoarsenessReport] = []
    t0 = time.time()
    for i, script in enumerate(library):
        if script.length == 0:
            continue

        lfp = create_lfp_propagator(ipomdp)
        sampler = create_sampler(ipomdp, config, seed=config.seed + i)
        fixed = FixedRealizationSampledBelief(
            ipomdp=ipomdp,
            num_realizations=config.sampler_budget,
            rng=np.random.default_rng(config.seed + 1000 + i),
        )

        evaluator = CoarsenessEvaluator(
            lfp, sampler, pp_shield,
            extra_samplers={"fixed_realization": fixed},
        )
        evaluator.restart()

        history = [(obs, action) for (_, obs, action) in script.steps]
        report = evaluator.run_trajectory(history)
        reports.append(report)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            v = _overall_max(report, "varying")
            f = _overall_max(report, "fixed_realization")
            print(f"  [{i+1}/{len(library)}] varying_max={v:.4f}  "
                  f"fixed_max={f:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    per_mode = {
        "varying": aggregate_per_mode(reports, "varying"),
        "fixed_realization": aggregate_per_mode(reports, "fixed_realization"),
    }

    # Tidy CSV: one row per (mode, timestep)
    tidy_rows = []
    for mode, summary in per_mode.items():
        for row in summary["timestep"]:
            tidy_rows.append({"mode": mode, **row})

    metadata = build_metadata(config, extra={"total_time_s": total_time})
    save_experiment_results(config.results_path, per_mode, metadata, tidy_rows)
    print(f"\nResults saved to {config.results_path}")

    figures_dir = os.path.join(
        os.path.dirname(config.results_path) or ".", "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(
        figures_dir, f"perception_variability_{config.case_study_name}.png"
    )
    _plot_overlay(per_mode, config, plot_path)

    # Brief stdout summary
    print("\n--- Overall max safe gap to LFP ---")
    for mode, summary in per_mode.items():
        g = summary["overall_max_gap"]
        print(f"  {mode:>20s}: mean={g['mean']:.4f}  std={g['std']:.4f}  "
              f"median={g['median']:.4f}")


if __name__ == "__main__":
    main()
