"""TaxiNetV2 comparison: conformal prediction shielding vs. iPOMDP shielding.

Runs TaxiNetV2 comparison bundles where point shields always use the shared
single-estimate confusion model, while conformal prediction shielding consumes
paired point/conformal events at the requested confidence level.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .run_rl_shield_experiment import (
    ShieldCompliantSelector,
    create_conformal_shield_factory,
    create_envelope_shield_factory,
    create_forward_sampling_shield_factory,
    create_no_shield_factory,
    create_observation_shield_factory,
    create_single_belief_shield_factory,
    plot_results,
    print_results_table,
    save_results,
    setup,
)
from .run_taxinet_v2_conformal_rl_sweep import _intervention_stats, _run_dual_observation_trials
from ..CaseStudies.TaxiNetV2 import (
    build_taxinet_v2_single_estimate_ipomdp,
    get_taxinet_v2_conditional_conformal_axis_data,
)
from ..CaseStudies.Taxinet.taxinet import taxinet_cte_states, taxinet_he_states
from ..MonteCarlo import (
    BeliefSelector,
    ModularConditionalConformalTaxiNetPerception,
    RandomActionSelector,
    RandomInitialState,
    UniformPerceptionModel,
    compute_safety_metrics,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scarbro_path() -> Path:
    return _project_root() / "results" / "taxinet_v2" / "scarbro_baseline_import.json"


def load_scarbro_baseline() -> Optional[Dict]:
    path = _scarbro_path()
    if not path.exists():
        print(f"  WARNING: Scarbro baseline not found at {path}")
        return None
    with path.open() as fh:
        return json.load(fh)


def extract_scarbro_tradeoff_points(
    scarbro: Dict,
    confidence_level: str = "0.95",
    default_action_only: bool = True,
) -> List[Dict]:
    points = []
    for variant in scarbro.get("variants", []):
        metadata = variant["metadata"]
        summary = variant["summary"]
        if metadata["confidence_level"] != confidence_level:
            continue
        if default_action_only and not metadata["default_action"]:
            continue
        crash = (summary.get("crash") or {}).get("value")
        stuck = (summary.get("stuck_or_default") or {}).get("value")
        if crash is None or stuck is None:
            continue
        points.append(
            {
                "label": metadata.get("action_filter_label", metadata["action_filter_tag"]),
                "confidence_level": metadata["confidence_level"],
                "action_filter": metadata.get("action_filter"),
                "crash": crash,
                "stuck": stuck,
            }
        )
    return sorted(points, key=lambda point: point["stuck"])


_SHIELD_ORDER = [
    "none",
    "observation",
    "single_belief",
    "envelope",
    "forward_sampling",
    "conformal_prediction",
]
_SHIELD_COLORS = {
    "none": "gray",
    "observation": "darkorange",
    "single_belief": "steelblue",
    "envelope": "green",
    "forward_sampling": "teal",
    "conformal_prediction": "crimson",
}
_SHIELD_LABELS = {
    "none": "No Shield",
    "observation": "Obs. Shield",
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Fwd-Sampling",
    "conformal_prediction": "Conf. Pred.",
}
_SHIELD_MARKERS = {
    "none": "X",
    "observation": "s",
    "single_belief": "D",
    "envelope": "o",
    "forward_sampling": "P",
    "conformal_prediction": "^",
}


def plot_comparison(
    our_results: Dict,
    scarbro: Optional[Dict],
    output_dir: str,
    confidence_level: str = "0.95",
    selector: str = "rl",
) -> None:
    """Generate comparison scatter plot: safety vs. conservatism."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping comparison plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    scarbro_points = []
    if scarbro is not None:
        scarbro_points = extract_scarbro_tradeoff_points(scarbro, confidence_level)

    for perception_name in ["uniform", "adversarial_opt"]:
        fig, ax = plt.subplots(figsize=(7, 6))

        for shield_name in _SHIELD_ORDER:
            key = f"{perception_name}/{selector}/{shield_name}"
            result = our_results.get(key)
            if result is None:
                continue
            ax.scatter(
                result["stuck_rate"],
                result["fail_rate"],
                s=140,
                zorder=5,
                color=_SHIELD_COLORS[shield_name],
                marker=_SHIELD_MARKERS[shield_name],
                label=f"iPOMDP: {_SHIELD_LABELS[shield_name]}",
            )
            if "fail_rate_ci_low" in result:
                ax.errorbar(
                    result["stuck_rate"],
                    result["fail_rate"],
                    yerr=[
                        [result["fail_rate"] - result["fail_rate_ci_low"]],
                        [result["fail_rate_ci_high"] - result["fail_rate"]],
                    ],
                    fmt="none",
                    ecolor=_SHIELD_COLORS[shield_name],
                    capsize=3,
                    alpha=0.6,
                )

        perception_label = "Uniform Perception" if perception_name == "uniform" else "Adversarial Perception"
        ax.set_xlabel("Stuck Rate  (conservatism ->)", fontsize=11)
        ax.set_ylabel("Fail/Crash Rate  (<- safety)", fontsize=11)
        ax.set_title(
            f"TaxiNetV2  |  conf={confidence_level}  |  {perception_label}\n"
            f"Safety-Liveness Tradeoff  (MC, {selector.upper()} selector)",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)

        output_name = f"comparison_{perception_name}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, output_name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_name}")

    print("\nComparison table  (RL selector):")
    print(f"  {'Shield':<22} {'Perception':<18} {'Fail%':>7} {'Stuck%':>7} {'Safe%':>7}")
    print("  " + "-" * 65)
    for perception_name in ["uniform", "adversarial_opt"]:
        for shield_name in _SHIELD_ORDER:
            key = f"{perception_name}/{selector}/{shield_name}"
            result = our_results.get(key)
            if result is None:
                continue
            print(
                f"  {shield_name:<22} {perception_name:<18} "
                f"{result['fail_rate']:>6.1%} {result['stuck_rate']:>6.1%} {result['safe_rate']:>6.1%}"
            )
    if scarbro_points:
        print("\n  Scarbro PRISM (worst-case formal bounds):")
        print(f"  {'Variant':<22} {'crash%':>7} {'stuck%':>7}")
        print("  " + "-" * 40)
        for point in scarbro_points:
            print(f"  {point['label']:<22} {point['crash']:>6.1%} {point['stuck']:>6.1%}")


def _point_perception_for(name: str, optimized_perceptions: Dict[str, Any]):
    if name == "uniform":
        return UniformPerceptionModel()
    if name != "adversarial_opt":
        raise ValueError(f"Unsupported perception regime: {name!r}")
    if "envelope" in optimized_perceptions:
        return optimized_perceptions["envelope"]
    return next(iter(optimized_perceptions.values()))


def _selectors(rl_selector, all_actions: List[Any]) -> Dict[str, Any]:
    return {
        "random": RandomActionSelector(),
        "best": BeliefSelector(mode="best"),
        "rl": ShieldCompliantSelector(rl_selector, all_actions),
    }


def _point_shields(point_ipomdp, pp_shield, config) -> Dict[str, Callable[[], Any]]:
    return {
        "none": create_no_shield_factory(list(point_ipomdp.actions)),
        "observation": create_observation_shield_factory(point_ipomdp, pp_shield, config.shield_threshold),
        "single_belief": create_single_belief_shield_factory(
            point_ipomdp.to_pomdp(),
            pp_shield,
            config.shield_threshold,
        ),
        "envelope": create_envelope_shield_factory(
            point_ipomdp,
            pp_shield,
            config.shield_threshold,
        ),
        "forward_sampling": create_forward_sampling_shield_factory(
            point_ipomdp,
            pp_shield,
            config.shield_threshold,
            budget=getattr(config, "forward_budget", 500),
            K_samples=getattr(config, "forward_k_samples", 100),
        ),
    }


def _run_batch(
    *,
    point_ipomdp,
    pp_shield,
    paired_perception,
    shield_factory: Callable[[], Any],
    selector,
    initial_generator,
    num_trials: int,
    trial_length: int,
    seed: int,
    use_conformal_shield_observation: bool,
):
    if hasattr(selector, "reset_stats"):
        selector.reset_stats()
    trial_results, _shield_stats, _size_stats = _run_dual_observation_trials(
        ipomdp=point_ipomdp,
        pp_shield=pp_shield,
        paired_perception=paired_perception,
        rt_shield_factory=shield_factory,
        action_selector=selector,
        initial_generator=initial_generator,
        num_trials=num_trials,
        trial_length=trial_length,
        seed=seed,
        use_conformal_shield_observation=use_conformal_shield_observation,
        store_trajectories=False,
    )
    metrics = compute_safety_metrics(trial_results)
    intervention = (
        _intervention_stats(selector)
        if isinstance(selector, ShieldCompliantSelector)
        else {
            "primary_count": 0,
            "fallback_count": 0,
            "intervention_rate": 0.0,
        }
    )
    return metrics, trial_results, intervention


def _conformal_shield_factory(pp_shield, all_actions):
    cte_states = taxinet_cte_states()
    he_states = taxinet_he_states()
    return create_conformal_shield_factory(pp_shield, all_actions, cte_states, he_states)


def run(config, skip_run: bool = False) -> None:
    """Run the TaxiNetV2 comparison experiment."""

    results_path = config.results_path
    figures_dir = config.figures_dir
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not skip_run:
        print("=" * 70)
        print("TAXINET V2 COMPARISON EXPERIMENT")
        print(f"Trials: {config.num_trials}, Length: {config.trial_length}, Seed: {config.seed}")
        print(f"Shield threshold: {config.shield_threshold}")
        print("=" * 70)

        confidence_level = config.ipomdp_kwargs.get("confidence_level", "0.95")
        print(f"\nLoading TaxiNetV2 point-estimate IPOMDP (shared across conf={confidence_level})...")
        point_ipomdp, pp_shield, _, _ = build_taxinet_v2_single_estimate_ipomdp(**config.ipomdp_kwargs)
        print(
            f"  States: {len(point_ipomdp.states)}, Actions: {len(point_ipomdp.actions)}, "
            f"Observations: {len(point_ipomdp.observations)}"
        )

        rl_selector, optimized_perceptions, setup_info = setup(point_ipomdp, pp_shield, config)
        if not optimized_perceptions:
            raise ValueError("TaxiNetV2 comparison requires a cached or trained adversarial realization.")

        conditional_cte_sets, conditional_he_sets = get_taxinet_v2_conditional_conformal_axis_data(
            confidence_level
        )
        point_shields = _point_shields(point_ipomdp, pp_shield, config)
        conformal_factory = _conformal_shield_factory(pp_shield, list(point_ipomdp.actions))
        initial_generator = RandomInitialState()

        total_combinations = 2 * 3 * (len(point_shields) + 1)
        print(f"\nExperiment grid: {total_combinations} combinations (2 perceptions x 3 selectors x 6 shields)")

        results = {}
        trial_data = {}
        intervention_stats = {}
        combo_index = 0
        started = time.time()

        for perception_name in ["uniform", "adversarial_opt"]:
            point_perception = _point_perception_for(perception_name, optimized_perceptions)
            conformal_perception = ModularConditionalConformalTaxiNetPerception(
                point_perception,
                point_ipomdp,
                conditional_cte_sets,
                conditional_he_sets,
            )

            for selector_name, selector in _selectors(rl_selector, list(point_ipomdp.actions)).items():
                for shield_name, shield_factory in point_shields.items():
                    combo_index += 1
                    label = f"{perception_name}/{selector_name}/{shield_name}"
                    print(f"\n[{combo_index}/{total_combinations}] Running: {label} ...", end=" ", flush=True)
                    t0 = time.time()
                    metrics, trials, intervention = _run_batch(
                        point_ipomdp=point_ipomdp,
                        pp_shield=pp_shield,
                        paired_perception=point_perception,
                        shield_factory=shield_factory,
                        selector=selector,
                        initial_generator=initial_generator,
                        num_trials=config.num_trials,
                        trial_length=config.trial_length,
                        seed=config.seed,
                        use_conformal_shield_observation=False,
                    )
                    elapsed = time.time() - t0
                    key = (perception_name, selector_name, shield_name)
                    results[key] = metrics
                    trial_data[key] = trials
                    if selector_name == "rl":
                        intervention_stats[key] = intervention
                    print(
                        f"fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}  "
                        f"safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)"
                    )

                combo_index += 1
                label = f"{perception_name}/{selector_name}/conformal_prediction"
                print(f"\n[{combo_index}/{total_combinations}] Running: {label} ...", end=" ", flush=True)
                t0 = time.time()
                metrics, trials, intervention = _run_batch(
                    point_ipomdp=point_ipomdp,
                    pp_shield=pp_shield,
                    paired_perception=conformal_perception,
                    shield_factory=conformal_factory,
                    selector=selector,
                    initial_generator=initial_generator,
                    num_trials=config.num_trials,
                    trial_length=config.trial_length,
                    seed=config.seed,
                    use_conformal_shield_observation=True,
                )
                elapsed = time.time() - t0
                key = (perception_name, selector_name, "conformal_prediction")
                results[key] = metrics
                trial_data[key] = trials
                if selector_name == "rl":
                    intervention_stats[key] = intervention
                print(
                    f"fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}  "
                    f"safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)"
                )

        total_time = time.time() - started
        print_results_table(results, config)

        extra = {
            **setup_info,
            "total_time_s": total_time,
            "intervention_stats": {
                f"{perception}/{selector}/{shield}": stats
                for (perception, selector, shield), stats in intervention_stats.items()
            },
            "note": (
                "Point shields use the shared TaxiNetV2 single-estimate confusion model and shared RL/"
                "adversarial caches across all confidence-level bundles; only conformal_prediction "
                f"depends on confidence_level={confidence_level} via paired conformal-set sampling."
            ),
        }
        save_results(results, config, setup_info=extra)

        print("\nGenerating per-shield time-series figures...")
        plot_results(trial_data, config, intervention_stats=intervention_stats)
        print(f"\nTotal experiment time: {total_time:.1f}s")

    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}; cannot plot comparison.")
        return

    with open(results_path) as fh:
        saved = json.load(fh)
    our_results = saved.get("results", {})

    scarbro = load_scarbro_baseline()
    conf_level = config.ipomdp_kwargs.get("confidence_level", "0.95")

    print("\nGenerating comparison figure...")
    plot_comparison(our_results, scarbro, figures_dir, confidence_level=conf_level)

    print(f"\nAll outputs written to: {figures_dir}")
    print("=" * 70)
    print("TAXINET V2 COMPARISON COMPLETE")
    print("=" * 70)


def main() -> None:
    import importlib

    skip_run = "--plot-only" in sys.argv
    config_name = "rl_shield_taxinet_v2_comparison"
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        config_name = args[idx + 1]

    config_module = importlib.import_module(
        f".configs.{config_name}",
        package="ipomdp_shielding.experiments",
    )
    config = config_module.config
    run(config, skip_run=skip_run)


if __name__ == "__main__":
    main()
