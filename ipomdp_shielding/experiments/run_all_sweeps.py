"""Artifact sweep runner for the paper experiment bundle.

Usage:
    python -m ipomdp_shielding.experiments.run_all_sweeps
"""

import json
import os
import time

from .run_threshold_sweep import (
    run_sweep_for_case_study as _ts_run,
    save_sweep as _ts_save,
)
from .run_forward_sampling_sweep import (
    run_sweep_for_case as _fs_run,
    THRESHOLDS,
)
from .run_observation_shield_sweep import (
    run_sweep_for_case as _obs_run,
)
from .run_carr_all_case_studies import run_carr_for_case_study as _carr_run


# ── output directories ────────────────────────────────────────────────────────

EXPERIMENT_BASE = "results/experiment"
EXPERIMENT_DATA_DIR = os.path.join(EXPERIMENT_BASE, "threshold")
EXPERIMENT_OBS_DIR = os.path.join(EXPERIMENT_BASE, "obs")
EXPERIMENT_FS_DIR = os.path.join(EXPERIMENT_BASE, "fs")
EXPERIMENT_CARR_DIR = os.path.join(EXPERIMENT_BASE, "carr")


# ── case study parameters ─────────────────────────────────────────────────────
THRESHOLD_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20, "exclude_envelope": False,
        "config_name": "rl_shield_taxinet_artifact",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25, "exclude_envelope": False,
        "config_name": "rl_shield_obstacle_artifact",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15, "exclude_envelope": True,
        "config_name": "rl_shield_cartpole_lowacc_artifact",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30, "exclude_envelope": True,
        "config_name": "rl_shield_refuel_v2_artifact",
    },
}

OBS_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_artifact",
    },
    "cartpole": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_artifact",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_artifact",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_artifact",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30,
        "config_name": "rl_shield_refuel_v2_artifact",
    },
}

FS_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_artifact",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_artifact",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_artifact",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30,
        "config_name": "rl_shield_refuel_v2_artifact",
    },
}

CARR_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_artifact",
    },
    "cartpole": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_artifact",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_artifact",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_artifact",
    },
    # refuel_v2: support-MDP BFS infeasible (344 states, 29 obs)
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


# ── threshold sweep ───────────────────────────────────────────────────────────

def run_threshold_sweep():
    print("\n" + "=" * 70)
    print("ARTIFACT THRESHOLD SWEEP (envelope + single_belief)")
    print("=" * 70)
    os.makedirs(EXPERIMENT_DATA_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in THRESHOLD_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _ts_run(cs_name, params)
            _ts_save(cs_name, sweep_results, base_config, params, setup_info,
                     output_dir=EXPERIMENT_DATA_DIR)
            print(f">>> {cs_name.upper()} threshold sweep saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
        timings[cs_name] = time.time() - t0

    return timings


# ── forward sampling sweep ────────────────────────────────────────────────────

def run_fs_sweep():
    print("\n" + "=" * 70)
    print("ARTIFACT FORWARD SAMPLING SWEEP")
    print("=" * 70)
    os.makedirs(EXPERIMENT_FS_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in FS_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _fs_run(cs_name, params)
            path = os.path.join(EXPERIMENT_FS_DIR, f"{cs_name}_fs_sweep.json")
            _save_json(path, {
                "metadata": {
                    "case_study": cs_name,
                    "shield": "forward_sampling",
                    "thresholds": THRESHOLDS,
                    "num_trials": params["num_trials"],
                    "trial_length": params["trial_length"],
                    "note": (
                        "Artifact bundle: adversarial realizations trained against RL selector. "
                        "Forward-sampled belief envelope: budget=500, K_samples=100."
                    ),
                    "setup_info": {k: str(v) for k, v in setup_info.items()},
                },
                "sweep_results": sweep_results,
            })
            print(f">>> {cs_name.upper()} fs sweep saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
        timings[cs_name] = time.time() - t0

    return timings


# ── observation shield sweep ──────────────────────────────────────────────────

def run_obs_sweep():
    print("\n" + "=" * 70)
    print("ARTIFACT OBSERVATION SHIELD SWEEP")
    print("=" * 70)
    os.makedirs(EXPERIMENT_OBS_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in OBS_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _obs_run(cs_name, params)
            path = os.path.join(EXPERIMENT_OBS_DIR, f"{cs_name}_obs_sweep.json")
            _save_json(path, {
                "metadata": {
                    "case_study": cs_name,
                    "shield": "observation",
                    "thresholds": THRESHOLDS,
                    "num_trials": params["num_trials"],
                    "trial_length": params["trial_length"],
                    "note": (
                        "Artifact bundle: adversarial realizations trained against RL selector. "
                        "Observation shield is memoryless; realization optimised against "
                        "envelope or single_belief per case study."
                    ),
                    "setup_info": {k: str(v) for k, v in setup_info.items()},
                },
                "sweep_results": sweep_results,
            })
            print(f">>> {cs_name.upper()} obs sweep saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
        timings[cs_name] = time.time() - t0

    return timings


# ── Carr comparison ───────────────────────────────────────────────────────────

def run_carr_sweep():
    print("\n" + "=" * 70)
    print("ARTIFACT CARR SUPPORT-BASED SHIELDING")
    print("=" * 70)
    os.makedirs(EXPERIMENT_CARR_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in CARR_PARAMS.items():
        t0 = time.time()
        try:
            result = _carr_run(cs_name, params)
            path = os.path.join(EXPERIMENT_CARR_DIR, f"{cs_name}_carr_results.json")
            _save_json(path, result)
            status = result.get("status", "?")
            print(f">>> {cs_name.upper()} Carr ({status}) saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
            result = {"status": "error", "reason": str(exc)}
            path = os.path.join(EXPERIMENT_CARR_DIR, f"{cs_name}_carr_results.json")
            _save_json(path, result)
        timings[cs_name] = time.time() - t0

    return timings


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ARTIFACT ALL SWEEPS — adversarial realizations trained against RL selector")
    print("=" * 70)
    print(f"Output: {EXPERIMENT_BASE}/")
    print()

    overall_start = time.time()
    all_timings = {}

    all_timings["threshold"] = run_threshold_sweep()
    all_timings["fs"]        = run_fs_sweep()
    all_timings["obs"]       = run_obs_sweep()
    all_timings["carr"]      = run_carr_sweep()

    overall_elapsed = time.time() - overall_start
    hh, rem = divmod(int(overall_elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    print("\n" + "=" * 70)
    print(f"ARTIFACT ALL SWEEPS COMPLETE — {hh:02d}h {mm:02d}m {ss:02d}s")
    print(f"Results in {EXPERIMENT_BASE}/")
    for sweep, timings in all_timings.items():
        for cs, t in timings.items():
            mm2, ss2 = divmod(int(t), 60)
            hh2, mm2 = divmod(mm2, 60)
            print(f"  {sweep}/{cs:<20} {hh2:02d}h {mm2:02d}m {ss2:02d}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
