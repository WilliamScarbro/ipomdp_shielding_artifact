"""Threshold sweep for the forward-sampling shield across all v5 case studies.

Sweeps shield_threshold ∈ [0.50, 0.60, ..., 0.95] for the
ForwardSampledBelief-based shield (forward-sampled belief envelope) using the
RL selector and both perception regimes (uniform + adversarial_opt).

Case studies: TaxiNet, Obstacle, CartPole (low-accuracy), Refuel v2.
Forward sampling is feasible for all four (no LP required).

Reuses existing RL-agent and adversarial-realization caches from prior runs.

Usage:
    python -m ipomdp_shielding.experiments.run_forward_sampling_sweep
"""

import dataclasses
import importlib
import json
import os
import time

from .experiment_io import add_rate_cis
from .run_rl_shield_experiment import setup, build_grid, run_experiment


THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

OUTPUT_DIR = "results/forward_sampling_sweep"

# Case study parameters: num_trials, trial_length, config_module name.
# Matches the v5 setup (200 trials, same variants as plot_sweep_v5.py).
SWEEP_CASES = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_final",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_final",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30,
        "config_name": "rl_shield_refuel_v2",
    },
}


def _load_config(config_name):
    mod = importlib.import_module(
        f".configs.{config_name}",
        package="ipomdp_shielding.experiments",
    )
    return mod.config


def _filter_forward_sampling(grid):
    """Keep only rl selector + forward_sampling shield."""
    return [
        row for row in grid
        if row[1] == "rl" and row[2] == "forward_sampling"
    ]


def _metrics_to_dict(metrics):
    cell = {
        "fail_rate":  metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate":  metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "num_trials": metrics.num_trials,
    }
    add_rate_cis(cell, metrics.num_trials)
    return cell


def run_sweep_for_case(cs_name, params):
    print("\n" + "#" * 70)
    print(f"# FORWARD SAMPLING SHIELD SWEEP: {cs_name.upper()}")
    print(f"# Trials: {params['num_trials']}, Length: {params['trial_length']}")
    print(f"# Thresholds: {THRESHOLDS}")
    print("#" * 70)

    base_config = _load_config(params["config_name"])
    sweep_config = dataclasses.replace(
        base_config,
        num_trials=params["num_trials"],
        trial_length=params["trial_length"],
    )

    print(f"\nLoading {cs_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = sweep_config.build_ipomdp_fn(**sweep_config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, sweep_config)

    sweep_results = {}
    cs_start = time.time()

    for t_idx, threshold in enumerate(THRESHOLDS):
        t_key = f"{threshold:.2f}"
        print(f"\n--- [{t_idx + 1}/{len(THRESHOLDS)}] Threshold: {threshold} ---")
        t_start = time.time()

        t_config = dataclasses.replace(sweep_config, shield_threshold=threshold)
        full_grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, t_config)
        fs_grid = _filter_forward_sampling(full_grid)
        print(f"  Grid: {len(fs_grid)} combinations")

        results, _trial_data, _stats = run_experiment(ipomdp, pp_shield, fs_grid, t_config)

        sweep_results[t_key] = {
            f"{p}/{s}/{sh}": _metrics_to_dict(m)
            for (p, s, sh), m in results.items()
        }

        t_elapsed = time.time() - t_start
        print(f"  Done in {t_elapsed:.1f}s")

    cs_elapsed = time.time() - cs_start
    print(f"\n{cs_name.upper()} sweep complete in {cs_elapsed:.1f}s ({cs_elapsed / 60:.1f} min)")

    return sweep_results, setup_info, base_config


def save_sweep(cs_name, sweep_results, base_config, params, setup_info):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{cs_name}_fs_sweep.json")

    output = {
        "metadata": {
            "case_study": cs_name,
            "shield": "forward_sampling",
            "thresholds": THRESHOLDS,
            "num_trials": params["num_trials"],
            "trial_length": params["trial_length"],
            "base_config_rl_cache_path": base_config.rl_cache_path,
            "base_config_opt_cache_path": base_config.opt_cache_path,
            "note": (
                "Adversarial perception realizations reused from prior runs. "
                "Forward-sampled belief envelope: budget=100 points, K_samples=10."
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
    print("=" * 70)
    print("FORWARD SAMPLING SHIELD THRESHOLD SWEEP")
    print(f"Case studies: {list(SWEEP_CASES)}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 70)

    overall_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in SWEEP_CASES.items():
        try:
            sweep_results, setup_info, base_config = run_sweep_for_case(cs_name, params)
            path = save_sweep(cs_name, sweep_results, base_config, params, setup_info)
            print(f"\n>>> {cs_name.upper()} saved to {path}")
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
    print(f"FORWARD SAMPLING SWEEP COMPLETE — {hh:02d}h {mm:02d}m {ss:02d}s")
    print(f"Results in {OUTPUT_DIR}/")
    for cs, t in timings.items():
        if t is None:
            print(f"  {cs:<20} FAILED")
        else:
            mm2, ss2 = divmod(int(t), 60)
            hh2, mm2 = divmod(mm2, 60)
            print(f"  {cs:<20} cumulative {hh2:02d}h {mm2:02d}m {ss2:02d}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
