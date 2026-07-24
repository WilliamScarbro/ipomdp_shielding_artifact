"""Coarseness evaluation of the LFP propagator on the TaxiNet model.

Measures the gap between the LFP over-approximation (BeliefPolytope via LP)
and the forward-sampled under-approximation (concrete belief points) over
multiple trajectories generated under perfect-perception shielding.

Usage:
    python -m ipomdp_shielding.experiments.coarse_experiment
"""

import os
import json
import time
import random

import numpy as np

from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Propagators import (
    LFPPropagator,
    BeliefPolytope,
    TemplateFactory,
    ForwardSampledBelief,
    LikelihoodSamplingStrategy,
    PruningStrategy,
)
from ..Propagators.lfp_propagator import default_solver
from ..Evaluation.coarseness_evaluator import CoarsenessEvaluator
from ..Evaluation.script_library import ScriptLibrary, generate_script
from ..MonteCarlo import UniformPerceptionModel


# ============================================================
# Configuration
# ============================================================

SEED = 42
NUM_TRAJECTORIES = 100
TRAJECTORY_LENGTH = 20
INITIAL_STATE = (0, 0)
INITIAL_ACTION = 0

# Forward-sampled belief parameters
SAMPLER_BUDGET = 200
SAMPLER_K = 20
SAMPLER_LIKELIHOOD_STRATEGY = LikelihoodSamplingStrategy.HYBRID
SAMPLER_PRUNING_STRATEGY = PruningStrategy.FARTHEST_POINT

RESULTS_PATH = "./data/coarseness_results.json"


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


def create_sampler(ipomdp, seed=None):
    """Create a ForwardSampledBelief propagator."""
    rng = np.random.default_rng(seed)
    return ForwardSampledBelief(
        ipomdp=ipomdp,
        budget=SAMPLER_BUDGET,
        K_samples=SAMPLER_K,
        likelihood_strategy=SAMPLER_LIKELIHOOD_STRATEGY,
        pruning_strategy=SAMPLER_PRUNING_STRATEGY,
        rng=rng,
    )


def generate_trajectories(ipomdp, pp_shield, num, length, seed=None):
    """Generate trajectories under perfect-perception shielding with uniform perception."""
    perception_model = UniformPerceptionModel()
    if seed is not None:
        random.seed(seed)

    library = ScriptLibrary(metadata={
        "num_scripts": num,
        "length": length,
        "seed": seed,
    })

    def perception_fn(state):
        return perception_model.sample_observation(state, ipomdp, {})

    for i in range(num):
        perception_model.begin_trajectory(ipomdp)
        try:
            script = generate_script(
                ipomdp=ipomdp,
                pp_shield=pp_shield,
                perception=perception_fn,
                initial=(INITIAL_STATE, INITIAL_ACTION),
                length=length,
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
    """
    all_max_gap = [r.overall_max_safe_gap for r in reports]
    all_mean_gap = [r.overall_mean_safe_gap for r in reports]

    # Per-timestep aggregation (average across trajectories)
    max_len = max(len(r.snapshots) for r in reports) if reports else 0
    timestep_max_gap = []
    timestep_mean_gap = []

    for t in range(max_len):
        max_vals = []
        mean_vals = []
        for r in reports:
            if t < len(r.snapshots):
                max_vals.append(r.snapshots[t].max_safe_gap)
                mean_vals.append(r.snapshots[t].mean_safe_gap)
        timestep_max_gap.append(float(np.mean(max_vals)) if max_vals else 0.0)
        timestep_mean_gap.append(float(np.mean(mean_vals)) if mean_vals else 0.0)

    return {
        "num_trajectories": len(reports),
        "overall_max_gap": {
            "mean": float(np.mean(all_max_gap)),
            "std": float(np.std(all_max_gap)),
            "min": float(np.min(all_max_gap)),
            "max": float(np.max(all_max_gap)),
        },
        "overall_mean_gap": {
            "mean": float(np.mean(all_mean_gap)),
            "std": float(np.std(all_mean_gap)),
        },
        "timestep_avg_max_gap": timestep_max_gap,
        "timestep_avg_mean_gap": timestep_mean_gap,
    }


# ============================================================
# Printing
# ============================================================

def print_report(summary):
    """Print a formatted summary of coarseness results."""
    print("\n" + "=" * 70)
    print("COARSENESS EVALUATION RESULTS")
    print("=" * 70)
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


def try_plot(summary):
    """Attempt to plot coarseness over time. Silently skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
        return

    ts_max = summary["timestep_avg_max_gap"]
    ts_mean = summary["timestep_avg_mean_gap"]
    timesteps = list(range(len(ts_max)))

    _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(timesteps, ts_max, "o-", label="Max Gap (avg over trajectories)")
    ax.plot(timesteps, ts_mean, "s--", label="Mean Gap (avg over trajectories)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Coarseness Gap")
    ax.set_title("LFP Coarseness Over Time (TaxiNet)")
    ax.legend()
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_PATH.replace(".json", ".png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("COARSENESS EXPERIMENT: LFP Propagator on TaxiNet")
    print(f"Trajectories: {NUM_TRAJECTORIES}, Length: {TRAJECTORY_LENGTH}, Seed: {SEED}")
    print(f"Sampler budget: {SAMPLER_BUDGET}, K: {SAMPLER_K}")
    print(f"Likelihood strategy: {SAMPLER_LIKELIHOOD_STRATEGY.name}")
    print(f"Pruning strategy: {SAMPLER_PRUNING_STRATEGY.name}")
    print("=" * 70)

    # 1. Build TaxiNet IPOMDP
    print("\nBuilding TaxiNet IPOMDP...")
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp(seed=SEED)
    n = len(ipomdp.states)
    print(f"  States: {n}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # 2. Create propagators
    print("\nCreating LFP propagator (canonical templates)...")
    lfp = create_lfp_propagator(ipomdp)

    print("Creating ForwardSampledBelief propagator...")
    sampler = create_sampler(ipomdp, seed=SEED)

    # 3. Create coarseness evaluator
    evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

    # 4. Generate trajectories
    print(f"\nGenerating {NUM_TRAJECTORIES} trajectories (length {TRAJECTORY_LENGTH})...")
    library = generate_trajectories(
        ipomdp, pp_shield,
        num=NUM_TRAJECTORIES,
        length=TRAJECTORY_LENGTH,
        seed=SEED,
    )
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
    summary["config"] = {
        "seed": SEED,
        "num_trajectories": NUM_TRAJECTORIES,
        "trajectory_length": TRAJECTORY_LENGTH,
        "sampler_budget": SAMPLER_BUDGET,
        "sampler_K": SAMPLER_K,
        "likelihood_strategy": SAMPLER_LIKELIHOOD_STRATEGY.name,
        "pruning_strategy": SAMPLER_PRUNING_STRATEGY.name,
        "total_time_s": total_time,
    }

    print_report(summary)

    # 7. Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # 8. Plot if possible
    try_plot(summary)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
