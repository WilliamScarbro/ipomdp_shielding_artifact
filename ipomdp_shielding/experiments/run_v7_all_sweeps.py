"""V7 master sweep runner.

Bug fix from v6: adversarial perception realizations are now trained against the
RL selector (ShieldCompliantSelector wrapping NeuralActionSelector) rather than
RandomActionSelector.  Using a random agent to optimise the adversarial realization
is incorrect: the adversary should exploit the actual deployed policy.

All adversarial-perception sweep results are regenerated from scratch with new v7
opt-realization caches (v7_ prefix).  Uniform-perception results are unaffected but
are re-collected for consistency.

Output structure
----------------
results/sweep_v7/
  threshold/   {cs}_sweep.json           (envelope + single_belief threshold sweep)
  obs/         {cs}_obs_sweep.json       (observation shield threshold sweep)
  fs/          {cs}_fs_sweep.json        (forward-sampling shield threshold sweep)
  carr/        {cs}_carr_results.json    (Carr support-based shielding)

Runtime estimate: ~8 hours total
  - Adversarial realization training: ~18 min
      TaxiNet (envelope):  ~2 min
      Obstacle (envelope): ~14 min
      CartPole lowacc (single_belief): <1 min
      Refuel v2 (single_belief): <1 min
      CartPole std (single_belief): <1 min
  - Threshold sweep (4 case studies, 200 trials, 9 thresholds): ~7 h
  - Forward sampling sweep: ~10 min
  - Observation shield sweep: ~5 min
  - Carr comparison (3 feasible case studies): ~30 min

Usage:
    python -m ipomdp_shielding.experiments.run_v7_all_sweeps
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
from .experiment_io import add_rate_cis


# ── output directories ────────────────────────────────────────────────────────

V7_BASE      = "results/sweep_v7"
V7_DATA_DIR  = os.path.join(V7_BASE, "threshold")
V7_OBS_DIR   = os.path.join(V7_BASE, "obs")
V7_FS_DIR    = os.path.join(V7_BASE, "fs")
V7_CARR_DIR  = os.path.join(V7_BASE, "carr")


# ── case study parameters ─────────────────────────────────────────────────────
# V7 configs use new opt_cache_path values (v7_ prefix) so stale random-agent
# realizations are never reused.

THRESHOLD_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20, "exclude_envelope": False,
        "config_name": "rl_shield_taxinet_v7",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25, "exclude_envelope": False,
        "config_name": "rl_shield_obstacle_v7",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15, "exclude_envelope": True,
        "config_name": "rl_shield_cartpole_lowacc_v7",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30, "exclude_envelope": True,
        "config_name": "rl_shield_refuel_v2_v7",
    },
}

OBS_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_v7",
    },
    "cartpole": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_v7",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_v7",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_v7",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30,
        "config_name": "rl_shield_refuel_v2_v7",
    },
}

FS_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_v7",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_v7",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_v7",
    },
    "refuel_v2": {
        "num_trials": 200, "trial_length": 30,
        "config_name": "rl_shield_refuel_v2_v7",
    },
}

CARR_PARAMS = {
    "taxinet": {
        "num_trials": 200, "trial_length": 20,
        "config_name": "rl_shield_taxinet_v7",
    },
    "cartpole": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_v7",
    },
    "obstacle": {
        "num_trials": 200, "trial_length": 25,
        "config_name": "rl_shield_obstacle_v7",
    },
    "cartpole_lowacc": {
        "num_trials": 200, "trial_length": 15,
        "config_name": "rl_shield_cartpole_lowacc_v7",
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
    print("V7 THRESHOLD SWEEP (envelope + single_belief)")
    print("=" * 70)
    os.makedirs(V7_DATA_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in THRESHOLD_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _ts_run(cs_name, params)
            _ts_save(cs_name, sweep_results, base_config, params, setup_info,
                     output_dir=V7_DATA_DIR)
            print(f">>> {cs_name.upper()} threshold sweep saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
        timings[cs_name] = time.time() - t0

    return timings


# ── forward sampling sweep ────────────────────────────────────────────────────

def run_fs_sweep():
    print("\n" + "=" * 70)
    print("V7 FORWARD SAMPLING SWEEP")
    print("=" * 70)
    os.makedirs(V7_FS_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in FS_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _fs_run(cs_name, params)
            path = os.path.join(V7_FS_DIR, f"{cs_name}_fs_sweep.json")
            _save_json(path, {
                "metadata": {
                    "case_study": cs_name,
                    "shield": "forward_sampling",
                    "thresholds": THRESHOLDS,
                    "num_trials": params["num_trials"],
                    "trial_length": params["trial_length"],
                    "note": (
                        "V7: adversarial realizations trained against RL selector. "
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
    print("V7 OBSERVATION SHIELD SWEEP")
    print("=" * 70)
    os.makedirs(V7_OBS_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in OBS_PARAMS.items():
        t0 = time.time()
        try:
            sweep_results, setup_info, base_config = _obs_run(cs_name, params)
            path = os.path.join(V7_OBS_DIR, f"{cs_name}_obs_sweep.json")
            _save_json(path, {
                "metadata": {
                    "case_study": cs_name,
                    "shield": "observation",
                    "thresholds": [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                    "num_trials": params["num_trials"],
                    "trial_length": params["trial_length"],
                    "note": (
                        "V7: adversarial realizations trained against RL selector. "
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
    print("V7 CARR SUPPORT-BASED SHIELDING")
    print("=" * 70)
    os.makedirs(V7_CARR_DIR, exist_ok=True)
    timings = {}

    for cs_name, params in CARR_PARAMS.items():
        t0 = time.time()
        try:
            result = _carr_run(cs_name, params)
            path = os.path.join(V7_CARR_DIR, f"{cs_name}_carr_results.json")
            _save_json(path, result)
            status = result.get("status", "?")
            print(f">>> {cs_name.upper()} Carr ({status}) saved.")
        except Exception as exc:
            print(f"!!! {cs_name.upper()} FAILED: {exc}")
            import traceback; traceback.print_exc()
            result = {"status": "error", "reason": str(exc)}
            path = os.path.join(V7_CARR_DIR, f"{cs_name}_carr_results.json")
            _save_json(path, result)
        timings[cs_name] = time.time() - t0

    return timings


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("V7 ALL SWEEPS — adversarial realizations trained against RL selector")
    print("=" * 70)
    print(f"Output: {V7_BASE}/")
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
    print(f"V7 ALL SWEEPS COMPLETE — {hh:02d}h {mm:02d}m {ss:02d}s")
    print(f"Results in {V7_BASE}/")
    for sweep, timings in all_timings.items():
        for cs, t in timings.items():
            mm2, ss2 = divmod(int(t), 60)
            hh2, mm2 = divmod(mm2, 60)
            print(f"  {sweep}/{cs:<20} {hh2:02d}h {mm2:02d}m {ss2:02d}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
