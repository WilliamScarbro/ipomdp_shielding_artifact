"""Run Carr support-based shielding on all four case studies.

Builds the support-MDP offline once per case study, then runs 200 trials with
the RL selector under both uniform and adversarial perception regimes.
Reuses the same RL agent and optimised-realization caches as the expanded
threshold sweep (threshold_sweep_expanded).

Saves one JSON per case study to:
    results/threshold_sweep_expanded/{cs_name}_carr_results.json

Usage:
    python -m ipomdp_shielding.experiments.run_carr_all_case_studies
    python -m ipomdp_shielding.experiments.run_carr_all_case_studies --dry-run
"""

import dataclasses
import importlib
import json
import os
import sys
import time

from .experiment_io import add_rate_cis
from .run_rl_shield_experiment import setup, ShieldCompliantSelector

from ..Evaluation.support_mdp_builder import SupportMDPBuilder
from ..Propagators.belief_support_propagator import BeliefSupportPropagator
from ..MonteCarlo import (
    UniformPerceptionModel,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)


OUTPUT_DIR = "results/threshold_sweep_expanded"

# Same trial budgets as the expanded threshold sweep.
CASE_STUDY_PARAMS = {
    "taxinet":   {"num_trials": 200, "trial_length": 20},
    "cartpole":  {"num_trials": 200, "trial_length": 15},
    "obstacle":  {"num_trials": 200, "trial_length": 25},
    "refuel_v2": {"num_trials": 200, "trial_length": 30,
                  "config_name": "rl_shield_refuel_v2"},
}

# Bail out if the support-MDP BFS exceeds this many support sets.
MAX_SUPPORT_MDP_STATES = 200_000


# ============================================================
# Lightweight per-trial Carr shield wrapper
# ============================================================

class CarrShieldWrapper:
    """Per-trial Carr shield that reuses a pre-built SupportMDPBuilder.

    The support-MDP (expensive to build) is computed once in
    ``build_carr_factory`` and shared across all trials via a closure.
    Each trial gets its own ``BeliefSupportPropagator`` that is reset
    to the full initial support at the start of every trial.

    Interface matches the shield protocol expected by
    ``run_monte_carlo_trials`` / ``run_single_trial``.
    """

    def __init__(self, pomdp, mdp_builder, initial_support):
        self.pomdp = pomdp
        self.mdp_builder = mdp_builder
        self.initial_support = initial_support
        self.propagator = BeliefSupportPropagator(pomdp, initial_support)
        self.stuck_count = 0
        self.error_count = 0

    def restart(self):
        self.propagator.restart()
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state=None):
        self.restart()

    def next_actions(self, evidence):
        """Update support then return safe actions from the winning region."""
        if evidence is not None:
            self.propagator.propogate(evidence)
        current_support = self.propagator.get_support()
        safe_actions = self.mdp_builder.get_safe_actions(current_support)
        if not safe_actions:
            self.stuck_count += 1
        return list(safe_actions)

    def get_action_probs(self):
        return []


# ============================================================
# Support-MDP builder / factory
# ============================================================

def build_carr_factory(ipomdp, pp_shield):
    """Build the support-MDP once and return a per-trial factory.

    Parameters
    ----------
    ipomdp : IPOMDP
    pp_shield : dict   state -> set of safe actions

    Returns
    -------
    factory : callable  () -> CarrShieldWrapper
    stats   : dict      support-MDP statistics
    elapsed : float     total build time in seconds

    Raises
    ------
    RuntimeError  if the support-MDP exceeds MAX_SUPPORT_MDP_STATES
    """
    pomdp = ipomdp.to_pomdp()

    # Avoid states = states with no safe actions in the perfect-perception shield.
    avoid_states = frozenset(
        s for s in ipomdp.states if not pp_shield.get(s, set())
    )
    # Initial support = safe states only (excludes FAIL and other avoid states).
    # Using frozenset(all_states) would include avoid states and immediately
    # place the initial support outside the winning region, making Carr block
    # all actions from step 0.
    initial_support = frozenset(ipomdp.states) - avoid_states

    print(f"  Avoid states : {len(avoid_states)} / {len(ipomdp.states)}")
    print(f"  Initial support size : {len(initial_support)} (safe states only)")

    # Feasibility pre-check: Obstacle (50 states, 3 obs) produced 47k supports and
    # was feasible. Refuel v2 (344 states, 29 obs) produced >5M and is infeasible.
    # Heuristic: skip if state space is large AND observations don't uniquely
    # identify states (ratio > ~5). CartPole (82/82=1.0) safely passes because
    # its 1:1 obs ratio collapses supports to singletons immediately.
    n_states = len(ipomdp.states)
    n_obs = len(ipomdp.observations)
    obs_ratio = n_states / max(n_obs, 1)
    if n_states > 200 and obs_ratio > 5:
        raise RuntimeError(
            f"State space too large for support-MDP BFS "
            f"(n_states={n_states}, n_obs={n_obs}, ratio={obs_ratio:.1f}). "
            f"Declared infeasible."
        )

    print(f"  Building support-MDP (BFS over reachable belief supports)...")

    t0 = time.time()
    mdp_builder = SupportMDPBuilder(pomdp, avoid_states)
    mdp_builder.build_support_mdp(initial_support)
    build_elapsed = time.time() - t0

    n_supports = len(mdp_builder.support_mdp)
    print(f"  Support-MDP : {n_supports} supports  ({build_elapsed:.1f}s)")

    if n_supports > MAX_SUPPORT_MDP_STATES:
        raise RuntimeError(
            f"Support-MDP too large: {n_supports} > {MAX_SUPPORT_MDP_STATES}"
        )

    mdp_builder.compute_winning_region()
    stats = mdp_builder.get_statistics()
    elapsed = time.time() - t0
    print(f"  Winning region : {stats['winning_supports']} / {stats['total_supports']} supports"
          f"  (total build: {elapsed:.1f}s)")

    def factory():
        return CarrShieldWrapper(pomdp, mdp_builder, initial_support)

    return factory, stats, elapsed


# ============================================================
# Per-case-study experiment
# ============================================================

def _load_config(cs_name, config_name=None):
    module = config_name or f"rl_shield_{cs_name}_final"
    mod = importlib.import_module(
        f".configs.{module}",
        package="ipomdp_shielding.experiments",
    )
    return mod.config


def _metrics_to_dict(metrics, num_trials):
    cell = {
        "fail_rate": metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate": metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "num_trials": num_trials,
    }
    add_rate_cis(cell, num_trials)
    return cell


def run_carr_for_case_study(cs_name, params, dry_run=False):
    """Run Carr experiments for one case study.

    Returns a result dict with keys:
      status      : "ok" | "infeasible" | "error"
      results     : {combo_key -> metrics_dict}   (only if status=="ok")
      mdp_stats   : support-MDP statistics
      build_time_s: support-MDP build time
      elapsed_s   : total elapsed seconds
    """
    print(f"\n{'#'*70}")
    print(f"# CARR: {cs_name.upper()}"
          f"  (trials={params['num_trials']}, length={params['trial_length']})")
    print(f"{'#'*70}")

    base_config = _load_config(cs_name, params.get("config_name"))
    exp_config = dataclasses.replace(
        base_config,
        num_trials=params["num_trials"],
        trial_length=params["trial_length"],
    )

    # Load IPOMDP.
    print(f"\nLoading {cs_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = exp_config.build_ipomdp_fn(**exp_config.ipomdp_kwargs)
    print(f"  States={len(ipomdp.states)}  Actions={len(ipomdp.actions)}"
          f"  Observations={len(ipomdp.observations)}")

    if dry_run:
        print("  [dry-run] Skipping support-MDP build and trials.")
        return {"status": "dry_run"}

    # Build Carr shield factory (expensive offline step).
    try:
        carr_factory, mdp_stats, build_time = build_carr_factory(ipomdp, pp_shield)
    except RuntimeError as exc:
        print(f"\n  Carr infeasible: {exc}")
        return {"status": "infeasible", "reason": str(exc)}

    # Load RL agent and optimised realizations (reuses prelim/v2 caches).
    rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, exp_config)

    all_actions = list(ipomdp.actions)
    rl_wrapped = ShieldCompliantSelector(rl_selector, all_actions)

    # Perception regimes.
    uniform_perception = UniformPerceptionModel()
    adversarial_perception = None
    if optimized_perceptions:
        adversarial_perception = (
            optimized_perceptions.get("envelope")
            or next(iter(optimized_perceptions.values()))
        )

    perception_map = {"uniform": uniform_perception}
    if adversarial_perception is not None:
        perception_map["adversarial_opt"] = adversarial_perception

    results = {}

    for p_name, perception in perception_map.items():
        print(f"\n  [{p_name}] Running {params['num_trials']} trials...")
        rl_wrapped.reset_stats()
        t0 = time.time()

        trial_results = run_monte_carlo_trials(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            perception=perception,
            rt_shield_factory=carr_factory,
            action_selector=rl_wrapped,
            initial_generator=RandomInitialState(),
            num_trials=params["num_trials"],
            trial_length=params["trial_length"],
            seed=base_config.seed,
        )
        metrics = compute_safety_metrics(trial_results)
        elapsed = time.time() - t0

        key = f"{p_name}/rl/carr"
        results[key] = _metrics_to_dict(metrics, params["num_trials"])
        print(f"    fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}"
              f"  safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)")

    return {
        "status": "ok",
        "results": results,
        "mdp_stats": {k: int(v) for k, v in mdp_stats.items()},
        "build_time_s": round(build_time, 2),
        "setup_info": {k: str(v) for k, v in setup_info.items()},
    }


# ============================================================
# Main
# ============================================================

def main():
    dry_run = "--dry-run" in sys.argv

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()

    print("=" * 70)
    print("CARR SUPPORT-BASED SHIELDING — ALL CASE STUDIES")
    if dry_run:
        print("(DRY RUN — skipping support-MDP build and trials)")
    print("=" * 70)

    all_results = {}

    for cs_name, params in CASE_STUDY_PARAMS.items():
        t0 = time.time()
        try:
            result = run_carr_for_case_study(cs_name, params, dry_run=dry_run)
        except Exception as exc:
            print(f"\n!!! {cs_name.upper()} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            result = {"status": "error", "reason": str(exc)}

        result["elapsed_s"] = round(time.time() - t0, 1)
        all_results[cs_name] = result

        # Save per-case-study result.
        if not dry_run:
            path = os.path.join(OUTPUT_DIR, f"{cs_name}_carr_results.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n>>> Saved {path}")

    overall_elapsed = time.time() - overall_start
    hh, rem = divmod(int(overall_elapsed), 3600)
    mm, ss = divmod(rem, 60)

    print("\n" + "=" * 70)
    print(f"CARR EXPERIMENTS COMPLETE — {hh:02d}h {mm:02d}m {ss:02d}s")
    for cs, res in all_results.items():
        status = res.get("status", "?")
        elapsed = res.get("elapsed_s", 0)
        mm2, ss2 = divmod(int(elapsed), 60)
        print(f"  {cs:<12} {status:12}  {mm2:02d}m {ss2:02d}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
