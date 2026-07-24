"""Shield inference-time benchmark.

Measures per-step wall-clock time for each shield type on each v5 case study.
Timing covers a single ``shield.next_actions(evidence)`` call, which includes
belief propagation and action filtering.  Results include mean, std, and
selected percentiles (p50, p95, p99).

Shields benchmarked (where feasible):
    no_shield     — passthrough baseline
    observation   — memoryless midpoint-posterior
    single_belief — POMDP point-belief propagation
    forward_sampling — ForwardSampledBelief (budget=100, K=10)
    envelope      — LFP polytope (LP solver); skipped where too slow
    carr          — support-MDP (precomputed); skipped where BFS infeasible

Usage:
    python -m ipomdp_shielding.experiments.run_timing_benchmark
    python -m ipomdp_shielding.experiments.run_timing_benchmark --quick  # fewer steps
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

# ── shield factories ────────────────────────────────────────────────────────
from .run_rl_shield_experiment import (
    create_no_shield_factory,
    create_observation_shield_factory,
    create_single_belief_shield_factory,
    create_envelope_shield_factory,
    create_forward_sampling_shield_factory,
)
from .run_carr_all_case_studies import build_carr_factory

OUTPUT_DIR = Path("results/timing_benchmark")

# ── case study config ───────────────────────────────────────────────────────
CASE_STUDIES = {
    "taxinet": {
        "config_name": "rl_shield_taxinet_final",
        "envelope": True,
        "carr": True,
    },
    "obstacle": {
        "config_name": "rl_shield_obstacle_final",
        "envelope": True,
        "carr": True,
    },
    "cartpole_lowacc": {
        "config_name": "rl_shield_cartpole_lowacc",
        "envelope": False,   # ~1.9 s/step — omit from timed run
        "carr": True,
    },
    "refuel_v2": {
        "config_name": "rl_shield_refuel_v2",
        "envelope": False,   # ~144 s/step — infeasible
        "carr": False,        # BFS infeasible at 344 states × 29 obs
    },
}

# Number of timed next_actions() calls per shield.
N_STEPS_FAST = 300    # observation / single_belief / forward_sampling / carr / none
N_STEPS_ENV  = 30     # envelope (LP is slow)


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_config(config_name: str):
    mod = importlib.import_module(
        f".configs.{config_name}",
        package="ipomdp_shielding.experiments",
    )
    return mod.config


def _random_walk(ipomdp, n_steps: int, seed: int = 0):
    """Generate a reproducible sequence of (action, obs) evidence pairs.

    Simulates the IPOMDP forward using the exact transition model and the
    midpoint observation probabilities so the sequence is realistic.
    Returns list of (action, obs) tuples of length n_steps.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    states = list(ipomdp.states)
    actions = list(ipomdp.actions)
    observations = list(ipomdp.observations)

    # Start from a random state.
    state = rng.choice(states)

    def sample_from(dist: dict):
        keys = list(dist.keys())
        probs = np.array([dist[k] for k in keys], dtype=float)
        probs /= probs.sum()
        idx = np_rng.choice(len(keys), p=probs)
        return keys[idx]

    def mid_O(s):
        """Midpoint observation distribution for state s."""
        row = {}
        for o in observations:
            lo = ipomdp.P_lower[s].get(o, 0.0)
            hi = ipomdp.P_upper[s].get(o, 0.0)
            mid = (lo + hi) / 2.0
            if mid > 0:
                row[o] = mid
        if not row:
            return {observations[0]: 1.0}
        total = sum(row.values())
        return {k: v / total for k, v in row.items()}

    evidence = []
    for _ in range(n_steps):
        action = rng.choice(actions)
        # ipomdp.T is exact: (s, a) -> {s': prob}
        T_dist = ipomdp.T.get((state, action), {state: 1.0})
        if not T_dist:
            T_dist = {state: 1.0}
        state = sample_from(T_dist)
        obs = sample_from(mid_O(state))
        evidence.append((action, obs))

    return evidence


def _time_shield(shield, evidence: list) -> dict:
    """Time each next_actions(ev) call; return statistics in milliseconds."""
    times_ms = []
    shield.restart()
    for action, obs in evidence:
        ev = (obs, action)
        t0 = time.perf_counter()
        shield.next_actions(ev)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "n_steps":  len(arr),
        "mean_ms":  float(np.mean(arr)),
        "std_ms":   float(np.std(arr, ddof=1)),
        "min_ms":   float(np.min(arr)),
        "p50_ms":   float(np.percentile(arr, 50)),
        "p95_ms":   float(np.percentile(arr, 95)),
        "p99_ms":   float(np.percentile(arr, 99)),
        "max_ms":   float(np.max(arr)),
    }


# ── main benchmark loop ──────────────────────────────────────────────────────

def benchmark_case_study(cs_name: str, cs_cfg: dict, quick: bool = False) -> dict:
    print(f"\n{'#' * 60}")
    print(f"# {cs_name.upper()}")
    print(f"{'#' * 60}")

    config = _load_config(cs_cfg["config_name"])
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, "
          f"Actions: {len(ipomdp.actions)}, "
          f"Obs: {len(ipomdp.observations)}")

    n_fast = max(50, N_STEPS_FAST // 3) if quick else N_STEPS_FAST
    n_env  = max(10, N_STEPS_ENV  // 3) if quick else N_STEPS_ENV

    ev_fast = _random_walk(ipomdp, n_fast)
    ev_env  = _random_walk(ipomdp, n_env)

    results = {}
    threshold = 0.90   # fixed threshold for timing (doesn't materially affect speed)
    pomdp = ipomdp.to_pomdp()
    all_actions = list(ipomdp.actions)

    # ── no_shield ──────────────────────────────────────────────────────────
    print(f"  Timing no_shield        ({n_fast} steps) ...", end=" ", flush=True)
    shield = create_no_shield_factory(all_actions)()
    stats = _time_shield(shield, ev_fast)
    results["no_shield"] = stats
    print(f"mean={stats['mean_ms']:.3f} ms  std={stats['std_ms']:.3f} ms")

    # ── observation ────────────────────────────────────────────────────────
    print(f"  Timing observation      ({n_fast} steps) ...", end=" ", flush=True)
    shield = create_observation_shield_factory(ipomdp, pp_shield, threshold)()
    stats = _time_shield(shield, ev_fast)
    results["observation"] = stats
    print(f"mean={stats['mean_ms']:.3f} ms  std={stats['std_ms']:.3f} ms")

    # ── single_belief ──────────────────────────────────────────────────────
    print(f"  Timing single_belief    ({n_fast} steps) ...", end=" ", flush=True)
    shield = create_single_belief_shield_factory(pomdp, pp_shield, threshold)()
    stats = _time_shield(shield, ev_fast)
    results["single_belief"] = stats
    print(f"mean={stats['mean_ms']:.3f} ms  std={stats['std_ms']:.3f} ms")

    # ── forward_sampling ───────────────────────────────────────────────────
    print(f"  Timing forward_sampling ({n_fast} steps) ...", end=" ", flush=True)
    shield = create_forward_sampling_shield_factory(ipomdp, pp_shield, threshold)()
    stats = _time_shield(shield, ev_fast)
    results["forward_sampling"] = stats
    print(f"mean={stats['mean_ms']:.3f} ms  std={stats['std_ms']:.3f} ms")

    # ── envelope ───────────────────────────────────────────────────────────
    if cs_cfg["envelope"]:
        print(f"  Timing envelope         ({n_env} steps) ...", end=" ", flush=True)
        shield = create_envelope_shield_factory(ipomdp, pp_shield, threshold)()
        stats = _time_shield(shield, ev_env)
        results["envelope"] = stats
        print(f"mean={stats['mean_ms']:.1f} ms  std={stats['std_ms']:.1f} ms")
    else:
        print(f"  envelope: skipped (infeasible for {cs_name})")
        results["envelope"] = {"skipped": True, "reason": "LP infeasible at this scale"}

    # ── carr ───────────────────────────────────────────────────────────────
    if cs_cfg["carr"]:
        print(f"  Building Carr support-MDP ...", end=" ", flush=True)
        try:
            factory, _stats_carr, build_elapsed = build_carr_factory(ipomdp, pp_shield)
            print(f"done in {build_elapsed:.1f}s")
            print(f"  Timing carr             ({n_fast} steps) ...", end=" ", flush=True)
            shield = factory()
            stats = _time_shield(shield, ev_fast)
            results["carr"] = stats
            results["carr"]["build_time_s"] = build_elapsed
            print(f"mean={stats['mean_ms']:.3f} ms  std={stats['std_ms']:.3f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results["carr"] = {"skipped": True, "reason": str(e)}
    else:
        print(f"  carr: skipped (infeasible for {cs_name})")
        results["carr"] = {"skipped": True, "reason": "support-MDP BFS infeasible"}

    return results


# ── save + print ─────────────────────────────────────────────────────────────

SHIELD_ORDER = ["no_shield", "observation", "single_belief", "forward_sampling",
                "envelope", "carr"]
SHIELD_LABELS = {
    "no_shield":        "No Shield",
    "observation":      "Observation",
    "single_belief":    "Single-Belief",
    "forward_sampling": "Fwd-Sampling",
    "envelope":         "Envelope",
    "carr":             "Carr",
}


def print_table(all_results: dict) -> None:
    """Print a cross-case timing table to stdout."""
    cases = list(all_results.keys())
    shields = SHIELD_ORDER

    # Header
    col_w = 16
    header = f"{'Shield':<{col_w}}" + "".join(f"  {c.upper()[:14]:>14}" for c in cases)
    print("\n" + "=" * len(header))
    print("SHIELD INFERENCE TIME — mean ± std (ms/step)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for sh in shields:
        row = f"{SHIELD_LABELS[sh]:<{col_w}}"
        for cs in cases:
            d = all_results[cs].get(sh, {})
            if d.get("skipped"):
                cell = "N/A"
            else:
                cell = f"{d['mean_ms']:.2f}±{d['std_ms']:.2f}"
            row += f"  {cell:>14}"
        print(row)

    print("=" * len(header))


def save_results(all_results: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "shield_timing.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {path}")
    return path


def _fmt_ms(mean_ms: float, std_ms: float, p95_ms: float) -> str:
    """Format timing cell with adaptive units and p95."""
    if mean_ms < 0.1:
        # Show in microseconds
        return f"{mean_ms*1000:.1f}±{std_ms*1000:.1f} μs (p95={p95_ms*1000:.1f})"
    if mean_ms >= 1000:
        return f"{mean_ms/1000:.1f}±{std_ms/1000:.1f} s (p95={p95_ms/1000:.1f})"
    return f"{mean_ms:.1f}±{std_ms:.1f} ms (p95={p95_ms:.1f})"


def generate_markdown_table(all_results: dict) -> str:
    """Return a markdown timing table string for inclusion in evaluation summaries."""
    cases = list(all_results.keys())
    case_labels = {
        "taxinet":         "TaxiNet",
        "obstacle":        "Obstacle",
        "cartpole_lowacc": "CartPole low-acc",
        "refuel_v2":       "Refuel v2",
    }

    def _fmt(d: dict) -> str:
        if d.get("skipped"):
            return "—"
        return _fmt_ms(d["mean_ms"], d["std_ms"], d["p95_ms"])

    # Header row
    header_cols = ["Shield"] + [case_labels.get(c, c) for c in cases]
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["---"] * len(header_cols)) + "|",
    ]
    for sh in SHIELD_ORDER:
        row = [SHIELD_LABELS[sh]]
        for cs in cases:
            d = all_results[cs].get(sh, {})
            row.append(_fmt(d))
        lines.append("| " + " | ".join(row) + " |")

    note = (
        "\n*Timing: mean ± std (p95) per step, 300 steps × threshold=0.90, "
        "random-walk trajectories. Envelope: 30 steps (LP-based). "
        "Units: μs = microseconds, ms = milliseconds. "
        "— = infeasible.*"
    )
    return "\n".join(lines) + note


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer steps (fast smoke-test)")
    args = parser.parse_args()

    print("=" * 60)
    print("SHIELD INFERENCE TIME BENCHMARK")
    print(f"  Fast shields: {N_STEPS_FAST if not args.quick else N_STEPS_FAST // 3} steps")
    print(f"  Envelope:     {N_STEPS_ENV  if not args.quick else N_STEPS_ENV  // 3} steps")
    print("=" * 60)

    all_results = {}
    for cs_name, cs_cfg in CASE_STUDIES.items():
        all_results[cs_name] = benchmark_case_study(cs_name, cs_cfg, quick=args.quick)

    print_table(all_results)
    save_results(all_results)

    md = generate_markdown_table(all_results)
    md_path = OUTPUT_DIR / "timing_table.md"
    md_path.write_text(md)
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
