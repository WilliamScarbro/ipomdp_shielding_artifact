"""TaxiNetV2 RL operating sweep: beta for point shields, conf/af for conformal."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .run_rl_shield_experiment import (
    ShieldCompliantSelector,
    create_envelope_shield_factory,
    create_forward_sampling_shield_factory,
    create_single_belief_shield_factory,
    setup,
)
from .run_taxinet_v2_comparison import (
    _point_perception_for,
)
from .run_taxinet_v2_conformal_rl_sweep import (
    _intervention_stats,
    _run_dual_observation_trials,
    _summarize_conformal_sizes,
)
from .configs.rl_shield_taxinet_v2_comparison import config as comparison_config
from ..CaseStudies.TaxiNetV2 import (
    build_taxinet_v2_single_estimate_ipomdp,
    get_taxinet_v2_conditional_conformal_axis_data,
)
from ..Evaluation.conformal_set_shield import ConformalSetIntersectionShield
from ..MonteCarlo import (
    ModularConditionalConformalTaxiNetPerception,
    RandomInitialState,
    compute_safety_metrics,
)


POINT_METHODS = ("single_belief", "envelope", "forward_sampling")
PERCEPTIONS = ("uniform", "adversarial_opt")


@dataclass
class SweepConfig:
    num_trials: int
    trial_length: int
    seed: int
    rl_episodes: int
    rl_episode_length: int
    opt_candidates: int
    opt_trials_per_candidate: int
    opt_iterations: int
    setup_threshold: float
    confidence_method: str
    alpha: float
    error: float
    smoothing: bool
    beta_values: List[float]
    confidence_levels: List[str]
    action_filter_values: List[float]
    rl_cache_path: str
    opt_cache_path: str
    results_path: str
    csv_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=comparison_config.num_trials)
    parser.add_argument("--trial-length", type=int, default=comparison_config.trial_length)
    parser.add_argument("--seed", type=int, default=comparison_config.seed)
    parser.add_argument("--rl-episodes", type=int, default=comparison_config.rl_episodes)
    parser.add_argument("--rl-episode-length", type=int, default=comparison_config.rl_episode_length)
    parser.add_argument("--opt-candidates", type=int, default=comparison_config.opt_candidates)
    parser.add_argument(
        "--opt-trials-per-candidate",
        type=int,
        default=comparison_config.opt_trials_per_candidate,
    )
    parser.add_argument("--opt-iterations", type=int, default=comparison_config.opt_iterations)
    parser.add_argument("--setup-threshold", type=float, default=0.8)
    parser.add_argument("--confidence-method", default="Clopper_Pearson")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--error", type=float, default=0.1)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument(
        "--beta-values",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        default=["0.95", "0.99", "0.995"],
    )
    parser.add_argument(
        "--action-filter-values",
        nargs="+",
        type=float,
        default=[0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/results.json"),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/results.csv"),
    )
    parser.add_argument(
        "--rl-cache-path",
        type=Path,
        default=Path("results/cache/rl_shield_taxinet_v2_agent.pt"),
    )
    parser.add_argument(
        "--opt-cache-path",
        type=Path,
        default=Path("results/cache/rl_shield_taxinet_v2_comparison_point_opt_realization.json"),
    )
    return parser.parse_args()


def _setup_config(args: argparse.Namespace):
    return argparse.Namespace(
        rl_cache_path=str(args.rl_cache_path),
        opt_cache_path=str(args.opt_cache_path),
        rl_episodes=args.rl_episodes,
        rl_episode_length=args.rl_episode_length,
        opt_candidates=args.opt_candidates,
        opt_trials_per_candidate=args.opt_trials_per_candidate,
        opt_iterations=args.opt_iterations,
        shield_threshold=args.setup_threshold,
        trial_length=args.trial_length,
        num_trials=args.num_trials,
        seed=args.seed,
        adversarial_opt_targets=["envelope"],
    )


def _point_factory(point_ipomdp, pp_shield, method: str, beta: float) -> Callable[[], Any]:
    if method == "single_belief":
        return create_single_belief_shield_factory(point_ipomdp.to_pomdp(), pp_shield, threshold=beta)
    if method == "envelope":
        return create_envelope_shield_factory(point_ipomdp, pp_shield, threshold=beta)
    if method == "forward_sampling":
        return create_forward_sampling_shield_factory(point_ipomdp, pp_shield, threshold=beta)
    raise ValueError(f"Unsupported point method: {method!r}")


def _metrics_dict(trial_results, selector, elapsed_seconds: float, size_stats: dict | None = None) -> dict:
    metrics = compute_safety_metrics(trial_results)
    payload = {
        "num_trials": metrics.num_trials,
        "fail_rate": metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate": metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "mean_stuck_count": metrics.mean_stuck_count,
        "fail_step_distribution": metrics.fail_step_distribution,
        "elapsed_seconds": elapsed_seconds,
    }
    payload.update(_intervention_stats(selector))
    if size_stats is not None:
        payload.update(_summarize_conformal_sizes(size_stats))
    return payload


def _run_trials(
    *,
    point_ipomdp,
    pp_shield,
    paired_perception,
    shield_factory: Callable[[], Any],
    selector,
    num_trials: int,
    trial_length: int,
    seed: int,
    use_conformal_shield_observation: bool,
):
    selector.reset_stats()
    started = time.time()
    trial_results, _shield_stats, size_stats = _run_dual_observation_trials(
        ipomdp=point_ipomdp,
        pp_shield=pp_shield,
        paired_perception=paired_perception,
        rt_shield_factory=shield_factory,
        action_selector=selector,
        initial_generator=RandomInitialState(),
        num_trials=num_trials,
        trial_length=trial_length,
        seed=seed,
        use_conformal_shield_observation=use_conformal_shield_observation,
        store_trajectories=False,
    )
    elapsed_seconds = time.time() - started
    metrics = _metrics_dict(
        trial_results,
        selector,
        elapsed_seconds,
        size_stats if use_conformal_shield_observation else None,
    )
    return metrics


def _csv_rows(payload: dict) -> Iterable[dict]:
    for perception, methods in payload["point_sweep"].items():
        for method, by_beta in methods.items():
            for beta_key, metrics in by_beta.items():
                yield {
                    "perception": perception,
                    "family": "point",
                    "method": method,
                    "beta": float(beta_key),
                    "confidence_level": "",
                    "action_filter": "",
                    "fail_rate": metrics["fail_rate"],
                    "stuck_rate": metrics["stuck_rate"],
                    "safe_rate": metrics["safe_rate"],
                    "mean_steps": metrics["mean_steps"],
                    "mean_stuck_count": metrics["mean_stuck_count"],
                    "intervention_rate": metrics["intervention_rate"],
                    "mean_conformal_cartesian_size": "",
                    "elapsed_seconds": metrics["elapsed_seconds"],
                }
    for perception, by_conf in payload["conformal_sweep"].items():
        for confidence_level, by_filter in by_conf.items():
            for filter_key, metrics in by_filter.items():
                yield {
                    "perception": perception,
                    "family": "conformal",
                    "method": "conformal_prediction",
                    "beta": "",
                    "confidence_level": confidence_level,
                    "action_filter": float(filter_key),
                    "fail_rate": metrics["fail_rate"],
                    "stuck_rate": metrics["stuck_rate"],
                    "safe_rate": metrics["safe_rate"],
                    "mean_steps": metrics["mean_steps"],
                    "mean_stuck_count": metrics["mean_stuck_count"],
                    "intervention_rate": metrics["intervention_rate"],
                    "mean_conformal_cartesian_size": metrics.get("mean_conformal_cartesian_size", ""),
                    "elapsed_seconds": metrics["elapsed_seconds"],
                }


def _write_csv(rows: Iterable[dict], path: Path) -> None:
    fieldnames = [
        "perception",
        "family",
        "method",
        "beta",
        "confidence_level",
        "action_filter",
        "fail_rate",
        "stuck_rate",
        "safe_rate",
        "mean_steps",
        "mean_stuck_count",
        "intervention_rate",
        "mean_conformal_cartesian_size",
        "elapsed_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    ipomdp_kwargs = {
        "confidence_method": args.confidence_method,
        "alpha": args.alpha,
        "confidence_level": args.confidence_levels[0],
        "error": args.error,
        "smoothing": not args.no_smoothing,
    }
    point_ipomdp, pp_shield, _, _ = build_taxinet_v2_single_estimate_ipomdp(**ipomdp_kwargs)
    rl_selector, optimized_perceptions, setup_info = setup(point_ipomdp, pp_shield, _setup_config(args))
    point_selector = ShieldCompliantSelector(rl_selector, list(point_ipomdp.actions))
    point_sweep: Dict[str, Dict[str, dict]] = {perception: {method: {} for method in POINT_METHODS} for perception in PERCEPTIONS}
    conformal_sweep: Dict[str, Dict[str, dict]] = {
        perception: {confidence: {} for confidence in args.confidence_levels}
        for perception in PERCEPTIONS
    }

    point_perceptions = {
        perception: _point_perception_for(perception, optimized_perceptions)
        for perception in PERCEPTIONS
    }
    paired_perceptions = {
        perception: {} for perception in PERCEPTIONS
    }
    for confidence_level in args.confidence_levels:
        conditional_cte_sets, conditional_he_sets = get_taxinet_v2_conditional_conformal_axis_data(confidence_level)
        for perception in PERCEPTIONS:
            paired_perceptions[perception][confidence_level] = ModularConditionalConformalTaxiNetPerception(
                point_perceptions[perception],
                point_ipomdp,
                conditional_cte_sets,
                conditional_he_sets,
            )

    for perception in PERCEPTIONS:
        for method in POINT_METHODS:
            for beta in args.beta_values:
                print(f"Running {perception}/{method} beta={beta:.2f}")
                metrics = _run_trials(
                    point_ipomdp=point_ipomdp,
                    pp_shield=pp_shield,
                    paired_perception=point_perceptions[perception],
                    shield_factory=_point_factory(point_ipomdp, pp_shield, method, beta),
                    selector=point_selector,
                    num_trials=args.num_trials,
                    trial_length=args.trial_length,
                    seed=args.seed,
                    use_conformal_shield_observation=False,
                )
                point_sweep[perception][method][f"{beta:.2f}"] = metrics

    for perception in PERCEPTIONS:
        for confidence_level in args.confidence_levels:
            for action_filter in args.action_filter_values:
                print(
                    f"Running {perception}/conformal_prediction "
                    f"conf={confidence_level} action_filter={action_filter:.2f}"
                )
                def shield_factory(action_filter=action_filter):
                    return ConformalSetIntersectionShield.from_tempest_csv(
                        action_filter=action_filter,
                        all_actions=point_ipomdp.actions,
                    )

                metrics = _run_trials(
                    point_ipomdp=point_ipomdp,
                    pp_shield=pp_shield,
                    paired_perception=paired_perceptions[perception][confidence_level],
                    shield_factory=shield_factory,
                    selector=point_selector,
                    num_trials=args.num_trials,
                    trial_length=args.trial_length,
                    seed=args.seed,
                    use_conformal_shield_observation=True,
                )
                metrics["confidence_level"] = confidence_level
                metrics["action_filter"] = action_filter
                conformal_sweep[perception][confidence_level][f"{action_filter:.2f}"] = metrics

    payload = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_study": "taxinet_v2",
            "experiment": "operating_pareto_sweep",
            "shared_rl_controller": True,
            "shared_adversarial_policy": True,
            "shared_point_ipomdp": True,
            "selector": "rl",
            "perceptions": list(PERCEPTIONS),
            "point_methods": list(POINT_METHODS),
            "conformal_method": "conformal_prediction",
            "beta_values": list(args.beta_values),
            "confidence_levels": list(args.confidence_levels),
            "action_filter_values": list(args.action_filter_values),
            "setup_threshold": args.setup_threshold,
            "ipomdp_kwargs": ipomdp_kwargs,
            "setup": setup_info,
            "run_config": asdict(
                SweepConfig(
                    num_trials=args.num_trials,
                    trial_length=args.trial_length,
                    seed=args.seed,
                    rl_episodes=args.rl_episodes,
                    rl_episode_length=args.rl_episode_length,
                    opt_candidates=args.opt_candidates,
                    opt_trials_per_candidate=args.opt_trials_per_candidate,
                    opt_iterations=args.opt_iterations,
                    setup_threshold=args.setup_threshold,
                    confidence_method=args.confidence_method,
                    alpha=args.alpha,
                    error=args.error,
                    smoothing=not args.no_smoothing,
                    beta_values=list(args.beta_values),
                    confidence_levels=list(args.confidence_levels),
                    action_filter_values=list(args.action_filter_values),
                    rl_cache_path=str(args.rl_cache_path),
                    opt_cache_path=str(args.opt_cache_path),
                    results_path=str(args.results_path),
                    csv_path=str(args.csv_path),
                )
            ),
            "note": (
                "Point-shield sweeps vary beta only for single_belief, envelope, and "
                "forward_sampling. Conformal sweeps vary confidence_level and "
                "action_filter only. All operating points reuse the same RL controller "
                "and the same shared adversarial realization cache."
            ),
        },
        "point_sweep": point_sweep,
        "conformal_sweep": conformal_sweep,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2))
    _write_csv(list(_csv_rows(payload)), args.csv_path)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(f"Saved JSON to {args.results_path}")
    print(f"Saved CSV to {args.csv_path}")
    report_beta = 0.80 if any(abs(beta - 0.80) < 1e-9 for beta in args.beta_values) else args.beta_values[0]
    beta_key = f"{report_beta:.2f}"
    for perception in PERCEPTIONS:
        for method in POINT_METHODS:
            metrics = payload["point_sweep"][perception][method][beta_key]
            print(
                f"{perception}/{method} beta={report_beta:.2f} "
                f"fail={metrics['fail_rate']:.1%} stuck={metrics['stuck_rate']:.1%}"
            )


if __name__ == "__main__":
    main()
