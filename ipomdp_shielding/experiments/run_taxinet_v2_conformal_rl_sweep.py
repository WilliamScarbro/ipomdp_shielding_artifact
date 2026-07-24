"""TaxiNetV2 RL sweep comparing point IPOMDP shields to conformal shielding."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..CaseStudies.TaxiNetV2 import (
    build_taxinet_v2_conformal_ipomdp,
    build_taxinet_v2_single_estimate_ipomdp,
    get_taxinet_v2_conditional_conformal_axis_data,
    get_taxinet_v2_metadata,
)
from ..Evaluation.conformal_set_shield import ConformalSetIntersectionShield
from ..MonteCarlo import (
    AdversarialPerceptionModel,
    BoundaryInitialState,
    ModularConditionalConformalTaxiNetPerception,
    NeuralActionSelector,
    PairedPerceptionEvent,
    PerceptionModel,
    SafeInitialState,
    SafetyTrialResult,
    UniformPerceptionModel,
    compute_safety_metrics,
)
from .run_rl_shield_experiment import (
    ShieldCompliantSelector,
    create_envelope_shield_factory,
    create_forward_sampling_shield_factory,
    create_single_belief_shield_factory,
)


POINT_METHODS = ("single_belief", "envelope", "forward_sampling")
CONFORMAL_METHOD = "cp_control_conformal"
POINT_REALIZATION_STRATEGIES = ("uniform", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--trial-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-method", default="Clopper_Pearson")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        default=["0.95", "0.99", "0.995"],
    )
    parser.add_argument("--shield-threshold", type=float, default=0.8)
    parser.add_argument("--action-filter", type=float, default=0.7)
    parser.add_argument("--action-success", type=float, default=0.9)
    parser.add_argument("--forward-budget", type=int, default=500)
    parser.add_argument("--forward-k-samples", type=int, default=100)
    parser.add_argument(
        "--point-realization",
        choices=POINT_REALIZATION_STRATEGIES,
        default="uniform",
    )
    parser.add_argument("--rl-episodes", type=int, default=500)
    parser.add_argument("--rl-episode-length", type=int, default=None)
    parser.add_argument("--rl-cache-path", type=Path, default=None)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("/home/dev/cp-control/train/models/best_model.pth"),
    )
    parser.add_argument("--measured-cte-accuracy", type=float, default=None)
    parser.add_argument("--measured-he-accuracy", type=float, default=None)
    parser.add_argument("--measured-joint-accuracy", type=float, default=None)
    parser.add_argument(
        "--conformal-mode",
        default="shared-event/axis-paired",
        help="Short provenance label for the conformal artifact construction mode.",
    )
    parser.add_argument("--store-trajectories", action="store_true")
    parser.add_argument(
        "--skip-point-baselines",
        action="store_true",
        help="Skip single-belief, envelope, and forward-sampling evaluation.",
    )
    parser.add_argument(
        "--skip-conformal",
        action="store_true",
        help="Skip cp-control conformal evaluation.",
    )
    parser.add_argument(
        "--initial",
        choices=["safe", "boundary"],
        default="safe",
        help="Initial state sampling regime. Both avoid starting in FAIL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/taxinet_v2/conformal_rl_sweep/results.json"),
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    if args.rl_episode_length is None:
        args.rl_episode_length = args.trial_length
    if args.rl_cache_path is None:
        args.rl_cache_path = Path(
            "results/cache/"
            f"rl_taxinet_v2_point_modular_{args.point_realization}_h"
            f"{args.rl_episode_length}_agent.pt"
        )
    if args.csv_output is None:
        args.csv_output = args.output.parent / "results.csv"
    return args


def _initial_generator(name: str):
    if name == "safe":
        return SafeInitialState()
    if name == "boundary":
        return BoundaryInitialState()
    raise ValueError(f"Unsupported initial generator: {name}")


def _json_safe_args(args: argparse.Namespace) -> dict:
    safe = vars(args).copy()
    for key in ("output", "csv_output", "rl_cache_path", "base_checkpoint"):
        safe[key] = str(safe[key])
    return safe


def _axis_size(axis_observation: Any) -> int:
    if axis_observation == "FAIL":
        return 1
    if isinstance(axis_observation, (tuple, list, set, frozenset)):
        return len(axis_observation)
    return 1


def _empty_conformal_size_stats() -> dict:
    return {
        "cte_sizes": [],
        "he_sizes": [],
        "cartesian_sizes": [],
    }


def _summarize_conformal_sizes(size_stats: dict) -> dict:
    cte_sizes = size_stats.get("cte_sizes", [])
    he_sizes = size_stats.get("he_sizes", [])
    cartesian_sizes = size_stats.get("cartesian_sizes", [])
    return {
        "mean_conformal_cte_set_size": float(np.mean(cte_sizes)) if cte_sizes else 0.0,
        "mean_conformal_he_set_size": float(np.mean(he_sizes)) if he_sizes else 0.0,
        "mean_conformal_cartesian_size": float(np.mean(cartesian_sizes)) if cartesian_sizes else 0.0,
    }


def _metrics_to_dict(results: List[SafetyTrialResult]) -> dict:
    metrics = compute_safety_metrics(results)
    return {
        "num_trials": metrics.num_trials,
        "fail_rate": metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate": metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "mean_stuck_count": metrics.mean_stuck_count,
        "fail_step_distribution": metrics.fail_step_distribution,
    }


def _intervention_stats(selector: ShieldCompliantSelector) -> dict:
    total = selector.primary_count + selector.fallback_count
    return {
        "primary_count": selector.primary_count,
        "fallback_count": selector.fallback_count,
        "intervention_rate": selector.fallback_count / total if total else 0.0,
    }


def _build_point_perception(
    strategy: str,
    pp_shield: Dict[Any, set],
) -> PerceptionModel:
    if strategy == "uniform":
        return UniformPerceptionModel()
    if strategy == "adversarial":
        return AdversarialPerceptionModel(pp_shield)
    raise ValueError(f"Unsupported TaxiNetV2 point realization strategy: {strategy}")


def _load_or_train_rl_selector(args: argparse.Namespace, point_ipomdp, pp_shield):
    cache_path = args.rl_cache_path
    if cache_path.exists():
        selector = NeuralActionSelector.load(str(cache_path), point_ipomdp)
        selector.exploration_rate = 0.0
        return selector, {
            "rl_agent_cached": True,
            "rl_cache_path": str(cache_path),
            "point_realization": args.point_realization,
        }

    point_perception = _build_point_perception(args.point_realization, pp_shield)
    selector = NeuralActionSelector(
        actions=list(point_ipomdp.actions),
        observations=point_ipomdp.observations,
        maximize_safety=True,
    )
    train_metrics = selector.train(
        ipomdp=point_ipomdp,
        perception=point_perception,
        num_episodes=args.rl_episodes,
        episode_length=args.rl_episode_length,
        verbose=args.rl_episodes >= 50,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    selector.save(str(cache_path))
    selector.exploration_rate = 0.0
    return selector, {
        "rl_agent_cached": False,
        "rl_cache_path": str(cache_path),
        "rl_training_final_safe_rate": train_metrics["final_safe_rate"],
        "rl_training_final_fail_rate": train_metrics["final_fail_rate"],
        "rl_episodes": args.rl_episodes,
        "rl_episode_length": args.rl_episode_length,
        "point_realization": args.point_realization,
    }


def _run_dual_observation_trials(
    *,
    ipomdp,
    pp_shield: Dict[Any, set],
    paired_perception: Any,
    rt_shield_factory: Callable[[], Any],
    action_selector: ShieldCompliantSelector,
    initial_generator,
    num_trials: int,
    trial_length: int,
    seed: int,
    use_conformal_shield_observation: bool,
    store_trajectories: bool,
) -> Tuple[List[SafetyTrialResult], dict, dict]:
    random.seed(seed)
    np.random.seed(seed)
    trial_results: List[SafetyTrialResult] = []
    shield_stats = {"stuck_count": 0, "error_count": 0}
    size_stats = _empty_conformal_size_stats()

    for trial_id in range(num_trials):
        state, action = initial_generator.generate(ipomdp, pp_shield)
        rt_shield = rt_shield_factory()
        rt_shield.restart()

        selector_history = []
        trajectory = []
        outcome = "safe"
        fail_step: Optional[int] = None
        steps_completed = 0
        perception_context = {"rt_shield": rt_shield, "history": selector_history}
        if hasattr(paired_perception, "begin_trajectory"):
            paired_perception.begin_trajectory(ipomdp, perception_context)
        try:
            for step in range(trial_length):
                if state == "FAIL":
                    outcome = "fail"
                    fail_step = step
                    break

                if hasattr(paired_perception, "sample_event"):
                    event = paired_perception.sample_event(state, ipomdp, perception_context)
                else:
                    point_observation = paired_perception.sample_observation(
                        state,
                        ipomdp,
                        perception_context,
                    )
                    event = PairedPerceptionEvent(
                        point_observation=point_observation,
                        conformal_observation=point_observation,
                    )
                selector_obs = event.point_observation
                shield_obs = (
                    event.conformal_observation
                    if use_conformal_shield_observation
                    else event.point_observation
                )

                if use_conformal_shield_observation and shield_obs != "FAIL":
                    cte_obs, he_obs = shield_obs
                    cte_size = _axis_size(cte_obs)
                    he_size = _axis_size(he_obs)
                    size_stats["cte_sizes"].append(cte_size)
                    size_stats["he_sizes"].append(he_size)
                    size_stats["cartesian_sizes"].append(cte_size * he_size)

                selector_history.append((selector_obs, action))

                if store_trajectories:
                    trajectory.append(
                        {
                            "state": state,
                            "shield_observation": shield_obs,
                            "selector_observation": selector_obs,
                            "previous_action": action,
                        }
                    )

                allowed_actions = rt_shield.next_actions((shield_obs, action))
                if not allowed_actions:
                    outcome = "stuck"
                    steps_completed = step
                    break

                action = action_selector.select(
                    selector_history,
                    allowed_actions,
                    context={"rt_shield": rt_shield, "history": selector_history},
                )
                state = ipomdp.evolve(state, action)
                steps_completed = step + 1
        finally:
            if hasattr(paired_perception, "end_trajectory"):
                paired_perception.end_trajectory()

        shield_stats["stuck_count"] += getattr(rt_shield, "stuck_count", 0)
        shield_stats["error_count"] += getattr(rt_shield, "error_count", 0)
        trial_results.append(
            SafetyTrialResult(
                trial_id=trial_id,
                outcome=outcome,
                steps_completed=steps_completed,
                stuck_count=getattr(rt_shield, "stuck_count", 0),
                fail_step=fail_step,
                trajectory=trajectory if store_trajectories else [],
            )
        )

    return trial_results, shield_stats, size_stats


def _validate_matching_dynamics(point_ipomdp, conformal_ipomdp, confidence_level: str) -> None:
    if point_ipomdp.states != conformal_ipomdp.states:
        raise ValueError(f"State mismatch for conformal confidence {confidence_level}")
    if point_ipomdp.actions != conformal_ipomdp.actions:
        raise ValueError(f"Action mismatch for conformal confidence {confidence_level}")
    if point_ipomdp.T != conformal_ipomdp.T:
        raise ValueError(f"Dynamics mismatch for conformal confidence {confidence_level}")


def _row_for_csv(
    *,
    seed: int,
    initial: str,
    trial_length: int,
    method: str,
    confidence_level: str,
    confidence_level_independent: bool,
    metrics: dict,
) -> dict:
    return {
        "seed": seed,
        "initial": initial,
        "trial_length": trial_length,
        "method": method,
        "confidence_level": confidence_level,
        "confidence_level_independent": confidence_level_independent,
        "num_trials": metrics.get("num_trials", 0),
        "fail_rate": metrics.get("fail_rate", 0.0),
        "stuck_rate": metrics.get("stuck_rate", 0.0),
        "safe_rate": metrics.get("safe_rate", 0.0),
        "mean_steps": metrics.get("mean_steps", 0.0),
        "mean_stuck_count": metrics.get("mean_stuck_count", 0.0),
        "primary_count": metrics.get("primary_count", 0),
        "fallback_count": metrics.get("fallback_count", 0),
        "intervention_rate": metrics.get("intervention_rate", 0.0),
    }


def _write_tidy_csv(rows: Iterable[dict], path: Path) -> None:
    fieldnames = [
        "seed",
        "initial",
        "trial_length",
        "method",
        "confidence_level",
        "confidence_level_independent",
        "num_trials",
        "fail_rate",
        "stuck_rate",
        "safe_rate",
        "mean_steps",
        "mean_stuck_count",
        "primary_count",
        "fallback_count",
        "intervention_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> dict:
    point_ipomdp, pp_shield, _test_cte, _test_he = build_taxinet_v2_single_estimate_ipomdp(
        confidence_method=args.confidence_method,
        alpha=args.alpha,
        action_success=args.action_success,
        smoothing=True,
    )
    rl_selector, setup_info = _load_or_train_rl_selector(args, point_ipomdp, pp_shield)
    initial_generator = _initial_generator(args.initial)

    point_methods: Dict[str, Callable[[], Any]] = {
        "single_belief": create_single_belief_shield_factory(
            point_ipomdp.to_pomdp(), pp_shield, threshold=args.shield_threshold
        ),
        "envelope": create_envelope_shield_factory(
            point_ipomdp, pp_shield, threshold=args.shield_threshold
        ),
        "forward_sampling": create_forward_sampling_shield_factory(
            point_ipomdp,
            pp_shield,
            threshold=args.shield_threshold,
            budget=args.forward_budget,
            K_samples=args.forward_k_samples,
        ),
    }

    point_perception = _build_point_perception(args.point_realization, pp_shield)

    run_point_baselines = not getattr(args, "skip_point_baselines", False)
    run_conformal = not getattr(args, "skip_conformal", False)

    csv_rows = []
    point_baselines: Dict[str, dict] = {}
    trials: Dict[str, Any] = {"point_baselines": {}, "conformal_results": {}}

    if run_point_baselines:
        for method in POINT_METHODS:
            selector = ShieldCompliantSelector(rl_selector, list(point_ipomdp.actions))
            selector.reset_stats()
            started = time.time()
            trial_results, shield_stats, _size_stats = _run_dual_observation_trials(
                ipomdp=point_ipomdp,
                pp_shield=pp_shield,
                paired_perception=point_perception,
                rt_shield_factory=point_methods[method],
                action_selector=selector,
                initial_generator=initial_generator,
                num_trials=args.num_trials,
                trial_length=args.trial_length,
                seed=args.seed,
                use_conformal_shield_observation=False,
                store_trajectories=args.store_trajectories,
            )
            metrics = _metrics_to_dict(trial_results)
            metrics.update(_intervention_stats(selector))
            metrics.update(shield_stats)
            metrics["elapsed_seconds"] = time.time() - started
            metrics["confidence_level_independent"] = True
            point_baselines[method] = metrics
            trials["point_baselines"][method] = [trial.__dict__ for trial in trial_results]
            csv_rows.append(
                _row_for_csv(
                    seed=args.seed,
                    initial=args.initial,
                    trial_length=args.trial_length,
                    method=method,
                    confidence_level="",
                    confidence_level_independent=True,
                    metrics=metrics,
                )
            )

    conformal_results: Dict[str, dict] = {}
    if run_conformal:
        for confidence_level in args.confidence_levels:
            conformal_ipomdp, _conformal_pp, _projected_cte, _projected_he = (
                build_taxinet_v2_conformal_ipomdp(
                    confidence_method=args.confidence_method,
                    alpha=args.alpha,
                    confidence_level=confidence_level,
                    action_success=args.action_success,
                    smoothing=True,
                )
            )
            _validate_matching_dynamics(point_ipomdp, conformal_ipomdp, confidence_level)
            conditional_cte_sets, conditional_he_sets = get_taxinet_v2_conditional_conformal_axis_data(
                confidence_level
            )
            paired_perception = ModularConditionalConformalTaxiNetPerception(
                point_perception,
                point_ipomdp,
                conditional_cte_sets,
                conditional_he_sets,
            )

            def conformal_factory():
                return ConformalSetIntersectionShield.from_tempest_csv(
                    action_filter=args.action_filter,
                    all_actions=point_ipomdp.actions,
                )

            selector = ShieldCompliantSelector(rl_selector, list(point_ipomdp.actions))
            selector.reset_stats()
            started = time.time()
            trial_results, shield_stats, size_stats = _run_dual_observation_trials(
                ipomdp=point_ipomdp,
                pp_shield=pp_shield,
                paired_perception=paired_perception,
                rt_shield_factory=conformal_factory,
                action_selector=selector,
                initial_generator=initial_generator,
                num_trials=args.num_trials,
                trial_length=args.trial_length,
                seed=args.seed,
                use_conformal_shield_observation=True,
                store_trajectories=args.store_trajectories,
            )
            metrics = _metrics_to_dict(trial_results)
            metrics.update(_intervention_stats(selector))
            metrics.update(shield_stats)
            metrics.update(_summarize_conformal_sizes(size_stats))
            metrics["elapsed_seconds"] = time.time() - started
            metrics["action_filter"] = args.action_filter
            metrics["confidence_level"] = confidence_level
            conformal_results.setdefault(confidence_level, {})[CONFORMAL_METHOD] = metrics
            trials["conformal_results"].setdefault(confidence_level, {})[CONFORMAL_METHOD] = [
                trial.__dict__ for trial in trial_results
            ]
            csv_rows.append(
                _row_for_csv(
                    seed=args.seed,
                    initial=args.initial,
                    trial_length=args.trial_length,
                    method=CONFORMAL_METHOD,
                    confidence_level=confidence_level,
                    confidence_level_independent=False,
                    metrics=metrics,
                )
            )

    metadata = {
        "case_study": "taxinet_v2_conformal_rl_sweep",
        "trial_length": args.trial_length,
        "base_perception_model": str(args.base_checkpoint),
        "measured_test_accuracy": {
            "cte_accuracy": args.measured_cte_accuracy,
            "he_accuracy": args.measured_he_accuracy,
            "joint_accuracy": args.measured_joint_accuracy,
        },
        "conformal_mode": args.conformal_mode,
        "dynamics": "cp-control stochastic action perturbation",
        "rl_observations": "point estimates for every method",
        "point_methods_use": (
            "point-estimate cp-control TaxiNet DNN events sampled from the "
            f"point-estimate IPOMDP using modular realization strategy "
            f"{args.point_realization!r}"
        ),
        "conformal_method_uses": (
            "two-stage cp-control TaxiNet DNN events: point estimate sampled from the "
            f"point-estimate IPOMDP using modular realization strategy "
            f"{args.point_realization!r}, then conformal set from conditional "
            "(true state, estimate) artifacts"
        ),
        "point_realization": args.point_realization,
        "confidence_levels": args.confidence_levels,
        "initial": args.initial,
        "run_point_baselines": run_point_baselines,
        "run_conformal": run_conformal,
        "args": _json_safe_args(args),
        "taxinet_v2_metadata": get_taxinet_v2_metadata(args.confidence_levels[0]).__dict__,
        "setup": setup_info,
    }

    _write_tidy_csv(csv_rows, args.csv_output)
    return {
        "metadata": metadata,
        "point_baselines": point_baselines,
        "conformal_results": conformal_results,
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    result = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved TaxiNetV2 conformal RL sweep JSON to {args.output}")
    print(f"Saved tidy CSV to {args.csv_output}")
    for method, metrics in result["point_baselines"].items():
        print(
            f"{method:<22} fail={metrics['fail_rate']:.1%} "
            f"stuck={metrics['stuck_rate']:.1%} safe={metrics['safe_rate']:.1%} "
            f"intervene={metrics['intervention_rate']:.1%}"
        )
    for confidence_level, methods in result["conformal_results"].items():
        metrics = methods[CONFORMAL_METHOD]
        print(
            f"{CONFORMAL_METHOD:<22} conf={confidence_level:<5} "
            f"fail={metrics['fail_rate']:.1%} stuck={metrics['stuck_rate']:.1%} "
            f"safe={metrics['safe_rate']:.1%} intervene={metrics['intervention_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
