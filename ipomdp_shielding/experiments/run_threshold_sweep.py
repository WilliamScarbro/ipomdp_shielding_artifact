"""Threshold sweep experiment: Pareto frontier of fail rate vs stuck rate.

Sweeps shield_threshold ∈ [0.50, 0.60, ..., 0.95] for single_belief and
envelope shields, with RL selector and both perception regimes.

Reuses prelim caches (RL agent, optimized realizations) from the final run.
Saves per-case-study JSON files to results/threshold_sweep/.

Usage:
    python -m ipomdp_shielding.experiments.run_threshold_sweep             # original
    python -m ipomdp_shielding.experiments.run_threshold_sweep --expanded  # 200-trial v2
"""

import dataclasses
import importlib
import json
import os
import sys
import time

from .experiment_io import add_rate_cis
from .run_rl_shield_experiment import (
    setup,
    build_grid,
    run_experiment,
)


THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Original sweep parameters (v1).
SWEEP_PARAMS = {
    "taxinet":  {"num_trials": 50,  "trial_length": 20, "exclude_envelope": False},
    "cartpole": {"num_trials": 15,  "trial_length": 15, "exclude_envelope": False},
    "obstacle": {"num_trials": 25,  "trial_length": 25, "exclude_envelope": False},
    "refuel":   {"num_trials": 30,  "trial_length": 30, "exclude_envelope": True},
}

# Expanded sweep (v2): 200 trials, CartPole envelope excluded (dominated by
# single_belief), Refuel upgraded to v2 (safety predicates hidden from obs).
EXPANDED_PARAMS = {
    "taxinet":  {"num_trials": 200, "trial_length": 20, "exclude_envelope": False},
    "cartpole": {"num_trials": 200, "trial_length": 15, "exclude_envelope": True},
    "obstacle": {"num_trials": 200, "trial_length": 25, "exclude_envelope": False},
    # config_name overrides the default rl_shield_{cs}_final lookup.
    "refuel_v2": {"num_trials": 200, "trial_length": 30, "exclude_envelope": True,
                  "config_name": "rl_shield_refuel_v2"},
    # Low-accuracy CartPole: perception trained on 175 ep (P_mid≈0.373 ≈ TaxiNet).
    # Envelope excluded (too slow for marginal benefit at this accuracy level).
    "cartpole_lowacc": {"num_trials": 200, "trial_length": 15, "exclude_envelope": True,
                        "config_name": "rl_shield_cartpole_lowacc"},
}

OUTPUT_DIR = "results/threshold_sweep"
EXPANDED_OUTPUT_DIR = "results/threshold_sweep_expanded"


def _load_base_config(cs_name, config_name=None):
    """Load a case-study config by module name.

    Parameters
    ----------
    cs_name : str
        Case study identifier (used as fallback module suffix).
    config_name : str, optional
        Explicit config module name (e.g. 'rl_shield_refuel_v2').
        Defaults to 'rl_shield_{cs_name}_final'.
    """
    module = config_name if config_name else f"rl_shield_{cs_name}_final"
    mod = importlib.import_module(
        f".configs.{module}",
        package="ipomdp_shielding.experiments",
    )
    return mod.config


def _filter_grid(grid, exclude_envelope):
    """Keep only rl selector + single_belief/envelope (per exclusion flag)."""
    shield_keep = {"single_belief"} if exclude_envelope else {"single_belief", "envelope"}
    return [
        row for row in grid
        if row[1] == "rl" and row[2] in shield_keep   # row[1]=selector, row[2]=shield
    ]


def _metrics_to_dict(metrics):
    """Convert MCSafetyMetrics to JSON-serializable dict with 95% CIs."""
    cell = {
        "fail_rate": metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate": metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "num_trials": metrics.num_trials,
    }
    add_rate_cis(cell, metrics.num_trials)
    return cell


def run_sweep_for_case_study(cs_name, params):
    """Run threshold sweep for one case study.

    Loads the IPOMDP, RL agent, and optimized realizations once, then iterates
    over THRESHOLDS.  At each threshold, builds a filtered grid (rl selector ×
    {single_belief, envelope}) and runs Monte Carlo trials.

    Returns
    -------
    sweep_results : dict  {threshold_str -> {combo_key -> metrics_dict}}
    setup_info    : dict  from setup()
    base_config   : RLShieldExperimentConfig
    """
    print("\n" + "#" * 70)
    print(f"# THRESHOLD SWEEP: {cs_name.upper()}")
    print(f"# Trials: {params['num_trials']}, Length: {params['trial_length']}")
    print(f"# Exclude envelope: {params['exclude_envelope']}")
    print(f"# Thresholds: {THRESHOLDS}")
    print("#" * 70)

    base_config = _load_base_config(cs_name, params.get("config_name"))

    # Override trial budget; keep cache paths and IPOMDP builder from the final config.
    sweep_config = dataclasses.replace(
        base_config,
        num_trials=params["num_trials"],
        trial_length=params["trial_length"],
    )

    # Load IPOMDP once.
    print(f"\nLoading {cs_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = sweep_config.build_ipomdp_fn(**sweep_config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # Load RL agent and optimized realizations once (hits prelim caches every time).
    rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, sweep_config)

    sweep_results = {}
    cs_start = time.time()

    for t_idx, threshold in enumerate(THRESHOLDS):
        t_key = f"{threshold:.2f}"
        print(f"\n--- [{t_idx + 1}/{len(THRESHOLDS)}] Threshold: {threshold} ---")
        t_start = time.time()

        # Build config with this threshold; everything else stays the same.
        t_config = dataclasses.replace(sweep_config, shield_threshold=threshold)

        full_grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, t_config)
        filtered_grid = _filter_grid(full_grid, params["exclude_envelope"])
        print(f"  Grid: {len(filtered_grid)} combinations")

        results, _trial_data, _stats = run_experiment(ipomdp, pp_shield, filtered_grid, t_config)

        sweep_results[t_key] = {
            f"{p}/{s}/{sh}": _metrics_to_dict(m)
            for (p, s, sh), m in results.items()
        }

        t_elapsed = time.time() - t_start
        print(f"  Done in {t_elapsed:.1f}s")

    cs_elapsed = time.time() - cs_start
    print(f"\n{cs_name.upper()} sweep complete in {cs_elapsed:.1f}s ({cs_elapsed / 60:.1f} min)")

    return sweep_results, setup_info, base_config


def save_sweep(cs_name, sweep_results, base_config, params, setup_info, output_dir=None):
    """Save sweep results to {output_dir}/{cs_name}_sweep.json."""
    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{cs_name}_sweep.json")

    output = {
        "metadata": {
            "case_study": cs_name,
            "thresholds": THRESHOLDS,
            "num_trials": params["num_trials"],
            "trial_length": params["trial_length"],
            "exclude_envelope": params["exclude_envelope"],
            "base_config_rl_cache_path": base_config.rl_cache_path,
            "base_config_opt_cache_path": base_config.opt_cache_path,
            "note": (
                "Adversarial perception realizations are reused from prelim caches "
                "(trained at threshold=0.8). This is a documented limitation — "
                "retraining per threshold would require weeks of compute."
            ),
            "setup_info": {k: str(v) for k, v in setup_info.items()},
        },
        "sweep_results": sweep_results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Saved {path}")
    return path


def main():
    expanded = "--expanded" in sys.argv
    sweep_params = EXPANDED_PARAMS if expanded else SWEEP_PARAMS
    output_dir = EXPANDED_OUTPUT_DIR if expanded else OUTPUT_DIR

    overall_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if expanded:
        print("=" * 70)
        print("EXPANDED THRESHOLD SWEEP (200 trials, Refuel v2, CartPole env excluded)")
        print("=" * 70)

    timings = {}

    for cs_name, params in sweep_params.items():
        try:
            sweep_results, setup_info, base_config = run_sweep_for_case_study(cs_name, params)
            path = save_sweep(cs_name, sweep_results, base_config, params, setup_info,
                              output_dir=output_dir)
            print(f"\n>>> {cs_name.upper()} results saved to {path}")
            timings[cs_name] = time.time() - overall_start
        except Exception as exc:
            print(f"\n!!! {cs_name.upper()} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            timings[cs_name] = None

    overall_elapsed = time.time() - overall_start
    hh, rem = divmod(int(overall_elapsed), 3600)
    mm, ss = divmod(rem, 60)

    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP COMPLETE — {hh:02d}h {mm:02d}m {ss:02d}s")
    print(f"Results in {output_dir}/")
    for cs, t in timings.items():
        if t is None:
            print(f"  {cs:<12} FAILED")
        else:
            mm2, ss2 = divmod(int(t), 60)
            hh2, mm2 = divmod(mm2, 60)
            print(f"  {cs:<12} cumulative {hh2:02d}h {mm2:02d}m {ss2:02d}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
