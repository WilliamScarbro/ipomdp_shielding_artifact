"""Coarseness evaluation experiment runner.

Measures the gap between the LFP over-approximation (BeliefPolytope via LP)
and the forward-sampled under-approximation (concrete belief points) over
multiple trajectories generated under perfect-perception shielding.

Usage:
    python -m ipomdp_shielding.experiments.run_coarse_experiment <config_module>

Example:
    python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim
"""

import os
import sys
import json
import time
import importlib
import random

import numpy as np

from .experiment_io import build_metadata, save_experiment_results

from ..Propagators import (
    LFPPropagator,
    BeliefPolytope,
    TemplateFactory,
    ForwardSampledBelief,
)
from ..Propagators.lfp_propagator import default_solver
from ..Evaluation.coarseness_evaluator import CoarsenessEvaluator
from ..Evaluation.script_library import ScriptLibrary
from ..Evaluation.script_library import generate_script
from ..MonteCarlo import UniformPerceptionModel


# ============================================================
# Helpers
# ============================================================

def create_lfp_propagator(ipomdp):
    """Create an LFP propagator with canonical templates."""
    n = len(ipomdp.states)
    template = TemplateFactory.canonical(n)
    polytope = BeliefPolytope.uniform_prior(n)
    solver = default_solver()
    return LFPPropagator(ipomdp, template, solver, polytope)


def create_sampler(ipomdp, config, seed=None):
    """Create a ForwardSampledBelief propagator."""
    rng = np.random.default_rng(seed)
    return ForwardSampledBelief(
        ipomdp=ipomdp,
        budget=config.sampler_budget,
        K_samples=config.sampler_k,
        likelihood_strategy=config.sampler_likelihood_strategy,
        pruning_strategy=config.sampler_pruning_strategy,
        rng=rng,
    )


def generate_trajectories(ipomdp, pp_shield, config, seed=None):
    """Generate trajectories under perfect-perception shielding with uniform perception."""
    perception_model = UniformPerceptionModel()
    if seed is not None:
        random.seed(seed)

    library = ScriptLibrary(metadata={
        "num_scripts": config.num_trajectories,
        "length": config.trajectory_length,
        "seed": seed,
    })

    def perception_fn(state):
        return perception_model.sample_observation(state, ipomdp, {})

    for i in range(config.num_trajectories):
        perception_model.begin_trajectory(ipomdp)
        try:
            script = generate_script(
                ipomdp=ipomdp,
                pp_shield=pp_shield,
                perception=perception_fn,
                initial=(config.initial_state, config.initial_action),
                length=config.trajectory_length,
            )
        finally:
            perception_model.end_trajectory()
        script.metadata["script_id"] = i
        library.add(script)

    return library


# ============================================================
# Run coarseness evaluation on one trajectory
# ============================================================

def evaluate_trajectory(evaluator, script):
    """Run CoarsenessEvaluator on a single RunScript.

    Returns a CoarsenessReport.
    """
    history = [(obs, action) for (_, obs, action) in script.steps]
    evaluator.restart()
    return evaluator.run_trajectory(history)


# ============================================================
# Aggregate reports
# ============================================================

def aggregate_reports(reports):
    """Compute summary statistics across multiple CoarsenessReports.

    Only tracks safe_gap since safe and unsafe gaps are symmetric
    (safe_gap == unsafe_gap by construction).

    Includes distributional summaries: median + 10th/90th percentiles.
    """
    all_max_gap = [r.overall_max_safe_gap for r in reports]
    all_mean_gap = [r.overall_mean_safe_gap for r in reports]

    # Per-timestep aggregation
    max_len = max(len(r.snapshots) for r in reports) if reports else 0
    timestep_max_gap = []
    timestep_mean_gap = []
    timestep_max_gap_median = []
    timestep_max_gap_p10 = []
    timestep_max_gap_p90 = []

    for t in range(max_len):
        max_vals = []
        mean_vals = []
        for r in reports:
            if t < len(r.snapshots):
                max_vals.append(r.snapshots[t].max_safe_gap)
                mean_vals.append(r.snapshots[t].mean_safe_gap)
        timestep_max_gap.append(float(np.mean(max_vals)) if max_vals else 0.0)
        timestep_mean_gap.append(float(np.mean(mean_vals)) if mean_vals else 0.0)
        timestep_max_gap_median.append(float(np.median(max_vals)) if max_vals else 0.0)
        timestep_max_gap_p10.append(float(np.percentile(max_vals, 10)) if max_vals else 0.0)
        timestep_max_gap_p90.append(float(np.percentile(max_vals, 90)) if max_vals else 0.0)

    return {
        "num_trajectories": len(reports),
        "overall_max_gap": {
            "mean": float(np.mean(all_max_gap)),
            "std": float(np.std(all_max_gap)),
            "min": float(np.min(all_max_gap)),
            "max": float(np.max(all_max_gap)),
            "median": float(np.median(all_max_gap)),
            "p10": float(np.percentile(all_max_gap, 10)),
            "p90": float(np.percentile(all_max_gap, 90)),
        },
        "overall_mean_gap": {
            "mean": float(np.mean(all_mean_gap)),
            "std": float(np.std(all_mean_gap)),
            "median": float(np.median(all_mean_gap)),
        },
        "timestep_avg_max_gap": timestep_max_gap,
        "timestep_avg_mean_gap": timestep_mean_gap,
        "timestep_max_gap_median": timestep_max_gap_median,
        "timestep_max_gap_p10": timestep_max_gap_p10,
        "timestep_max_gap_p90": timestep_max_gap_p90,
    }


# ============================================================
# Printing
# ============================================================

def print_report(summary, case_study_name):
    """Print a formatted summary of coarseness results."""
    print("\n" + "=" * 70)
    print("COARSENESS EVALUATION RESULTS")
    print("=" * 70)
    print(f"Case study: {case_study_name}")
    print(f"Trajectories evaluated: {summary['num_trajectories']}")

    print("\n--- Overall Max Gap (sampled_min_allowed - lfp_min_allowed) ---")
    g = summary["overall_max_gap"]
    print(f"  Mean: {g['mean']:.4f}  Std: {g['std']:.4f}  "
          f"Min: {g['min']:.4f}  Max: {g['max']:.4f}")

    print("\n--- Overall Mean Gap ---")
    mg = summary["overall_mean_gap"]
    print(f"  Mean: {mg['mean']:.4f}  Std: {mg['std']:.4f}")

    print("\n--- Per-Timestep Average Max Gap ---")
    ts = summary["timestep_avg_max_gap"]
    for t, val in enumerate(ts):
        bar = "#" * int(val * 50)
        print(f"  t={t:2d}: {val:.4f} {bar}")


def try_plot(summary, config):
    """Plot coarseness over time with distributional bands.

    Shows mean per-trajectory max gap (with 10th/90th percentile band)
    and mean per-trajectory mean gap for reference. Safe and unsafe gaps
    are identical by construction, so only one is shown.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
        return

    ts_max = summary["timestep_avg_max_gap"]
    ts_mean = summary["timestep_avg_mean_gap"]
    ts_p10 = summary.get("timestep_max_gap_p10", [0.0] * len(ts_max))
    ts_p90 = summary.get("timestep_max_gap_p90", ts_max)
    timesteps = list(range(len(ts_max)))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Percentile band
    ax.fill_between(timesteps, ts_p10, ts_p90, alpha=0.2, color="steelblue",
                    label="Max gap (10th–90th percentile)")
    ax.plot(timesteps, ts_max, "o-", color="steelblue",
            label="Mean max gap (avg over trajectories)")
    ax.plot(timesteps, ts_mean, "s--", color="darkorange", alpha=0.8,
            label="Mean gap (avg over trajectories)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Coarseness gap (sampled under-approx − LFP envelope)")
    ax.set_title(f"LFP Coarseness Over Time ({config.case_study_name.upper()})\n"
                 f"budget={config.sampler_budget}, K={config.sampler_k}, "
                 f"n={summary['num_trajectories']} trajectories")
    ax.legend()
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = config.results_path.replace(".json", ".png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ipomdp_shielding.experiments.run_coarse_experiment <config_module>")
        print("Example: python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim")
        sys.exit(1)

    # Import config
    config_module_name = sys.argv[1]
    try:
        config_module = importlib.import_module(f".{config_module_name}", package="ipomdp_shielding.experiments")
        config = config_module.config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print("=" * 70)
    print(f"COARSENESS EXPERIMENT: {config.case_study_name.upper()}")
    print(f"Trajectories: {config.num_trajectories}, Length: {config.trajectory_length}, Seed: {config.seed}")
    print(f"Sampler budget: {config.sampler_budget}, K: {config.sampler_k}")
    print(f"Likelihood strategy: {config.sampler_likelihood_strategy.name}")
    print(f"Pruning strategy: {config.sampler_pruning_strategy.name}")
    print("=" * 70)

    # 1. Build IPOMDP
    print(f"\nBuilding {config.case_study_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(seed=config.seed, **config.ipomdp_kwargs)
    n = len(ipomdp.states)
    print(f"  States: {n}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # 2. Create propagators
    print("\nCreating LFP propagator (canonical templates)...")
    lfp = create_lfp_propagator(ipomdp)

    print("Creating ForwardSampledBelief propagator...")
    sampler = create_sampler(ipomdp, config, seed=config.seed)

    # 3. Create coarseness evaluator
    evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

    # 4. Generate trajectories
    print(f"\nGenerating {config.num_trajectories} trajectories (length {config.trajectory_length})...")
    library = generate_trajectories(ipomdp, pp_shield, config, seed=config.seed)
    actual_lengths = [s.length for s in library]
    print(f"  Trajectory lengths: min={min(actual_lengths)}, max={max(actual_lengths)}, "
          f"mean={np.mean(actual_lengths):.1f}")

    # 5. Run coarseness evaluation
    print("\nEvaluating coarseness...")
    reports = []
    t0 = time.time()

    for i, script in enumerate(library):
        if script.length == 0:
            continue
        report = evaluate_trajectory(evaluator, script)
        reports.append(report)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(library)}] max_gap={report.overall_max_safe_gap:.4f}  "
                  f"mean_gap={report.overall_mean_safe_gap:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # 6. Aggregate and report
    summary = aggregate_reports(reports)

    print_report(summary, config.case_study_name)

    # 7. Save results with metadata
    metadata = build_metadata(config, extra={"total_time_s": total_time})
    save_experiment_results(config.results_path, summary, metadata)
    print(f"\nResults saved to {config.results_path}")

    # 8. Plot if possible
    try_plot(summary, config)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
