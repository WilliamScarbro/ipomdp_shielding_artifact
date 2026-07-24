"""Alpha sweep for RL shielding at multiple shield thresholds beta.

Sweeps over the CI significance `alpha` (which controls Clopper-Pearson
interval width) across a small set of fixed shield thresholds `beta`.
Produces a tidy CSV, Pareto scatter (fail vs stuck) per (perception, beta)
panel without connecting lines, and alpha-trend line plots for fail and
stuck rates.

Usage:
    python -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep [--config CONFIG_MODULE]

Default config: sweeps.rl_alpha_sweep_taxinet
"""

import os
import sys
import csv
import time
import importlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from ..experiment_io import build_metadata, add_rate_cis, save_experiment_results
from ..run_rl_shield_experiment import (
    create_envelope_shield_factory, create_single_belief_shield_factory,
    create_forward_sampling_shield_factory,
    ShieldCompliantSelector,
)
from ...MonteCarlo import (
    UniformPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    NeuralActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
    AdversarialPerceptionModel,
)


@dataclass
class AlphaSweepConfig:
    """Configuration for alpha sweep experiment across a few betas."""

    case_study_name: str = "taxinet"
    build_ipomdp_fn: Callable = None

    # Sweep
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    betas: List[float] = field(default_factory=lambda: [0.8])

    # Per-grid-point evaluation
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_trials: int = 30
    trial_length: int = 20

    # RL training (done once per alpha; beta-independent)
    rl_episodes: int = 500
    rl_episode_length: int = 20
    opt_candidates: int = 20
    opt_trials_per_candidate: int = 10
    opt_iterations: int = 10
    adversarial_opt_targets: List[str] = field(default_factory=lambda: ["envelope"])

    # Shields / perceptions
    shields: List[str] = field(default_factory=lambda: ["single_belief", "envelope"])
    perceptions: List[str] = field(default_factory=lambda: ["uniform", "adversarial_opt"])

    # Forward-sampling shield hyperparameters (only used if "forward_sampling" in shields)
    fs_budget: int = 500
    fs_K_samples: int = 100

    # Output
    results_dir: str = "./data/sweep/rl_alpha"

    # Extra IPOMDP kwargs (besides alpha)
    ipomdp_base_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.build_ipomdp_fn is None:
            from ...CaseStudies.Taxinet import build_taxinet_ipomdp
            self.build_ipomdp_fn = build_taxinet_ipomdp


def run_alpha_sweep(config: AlphaSweepConfig):
    """Run the alpha sweep across the configured betas.

    Outer loop: alpha (rebuild IPOMDP, train RL agent once per alpha because
    RL training is beta-independent).
    Inner loop: beta (train adversarial realization per (alpha,beta), then
    evaluate across seeds / perceptions / shields).
    """
    os.makedirs(config.results_dir, exist_ok=True)
    tidy_rows = []
    summary = {"config": config.__dict__.copy(), "grid_results": {}}
    summary["config"]["build_ipomdp_fn"] = str(config.build_ipomdp_fn)

    total_cells = (len(config.alphas) * len(config.betas) * len(config.seeds)
                   * len(config.shields) * len(config.perceptions))
    print("=" * 70)
    print(f"ALPHA SWEEP - {config.case_study_name.upper()}")
    print(f"Alphas: {config.alphas}")
    print(f"Betas:  {config.betas}")
    print(f"Seeds:  {config.seeds}")
    print(f"Total MC cells: {total_cells}")
    print("=" * 70)

    t0 = time.time()

    for alpha in config.alphas:
        print(f"\n{'='*70}\nALPHA = {alpha}\n{'='*70}")

        ipomdp_kwargs = {**config.ipomdp_base_kwargs, "alpha": alpha}
        ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**ipomdp_kwargs)
        all_actions = list(ipomdp.actions)
        pomdp = ipomdp.to_pomdp()

        cache_prefix = os.path.join(config.results_dir, f"cache_alpha{alpha}")
        rl_cache = f"{cache_prefix}_rl_agent.pt"

        # RL agent (beta-independent; cached per alpha)
        if os.path.exists(rl_cache):
            print(f"  Loading cached RL agent for alpha={alpha}")
            rl_selector = NeuralActionSelector.load(rl_cache, ipomdp)
        else:
            print(f"  Training RL agent for alpha={alpha}...")
            rl_selector = NeuralActionSelector(
                actions=all_actions,
                observations=ipomdp.observations,
                maximize_safety=True,
            )
            rl_selector.train(
                ipomdp=ipomdp,
                perception=AdversarialPerceptionModel(pp_shield),
                num_episodes=config.rl_episodes,
                episode_length=config.rl_episode_length,
                verbose=False,
            )
            rl_selector.save(rl_cache)
        rl_selector.exploration_rate = 0.0
        rl_wrapped = ShieldCompliantSelector(rl_selector, all_actions)

        for beta in config.betas:
            print(f"\n  --- beta = {beta} ---")

            beta_prefix = os.path.join(
                config.results_dir, f"cache_alpha{alpha}_beta{beta}"
            )
            opt_cache = f"{beta_prefix}_opt_realization.json"

            # Adversarial realizations per (alpha, beta) per target
            targets = list(dict.fromkeys(config.adversarial_opt_targets))
            opt_perceptions: Dict[str, FixedRealizationPerceptionModel] = {}
            for target in targets:
                target_cache = (
                    opt_cache if target == "envelope"
                    else opt_cache.replace(".json", f"_{target}.json")
                )
                if os.path.exists(target_cache):
                    print(f"    Loading cached adversarial realization ({target})"
                          f" for alpha={alpha}, beta={beta}")
                    opt_perceptions[target] = FixedRealizationPerceptionModel.load(target_cache)
                    continue

                print(f"    Training adversarial realization ({target}) for"
                      f" alpha={alpha}, beta={beta}...")
                if target == "envelope":
                    rt_factory = create_envelope_shield_factory(ipomdp, pp_shield, beta)
                elif target == "single_belief":
                    rt_factory = create_single_belief_shield_factory(pomdp, pp_shield, beta)
                else:
                    raise ValueError(
                        f"Unknown adversarial_opt_target={target!r}. "
                        "Supported: 'envelope', 'single_belief'."
                    )
                opt_perceptions[target] = train_optimal_realization(
                    ipomdp=ipomdp,
                    pp_shield=pp_shield,
                    rt_shield_factory=rt_factory,
                    action_selector=RandomActionSelector(),
                    initial_generator=RandomInitialState(),
                    num_candidates=config.opt_candidates,
                    num_trials_per_candidate=config.opt_trials_per_candidate,
                    max_iterations=config.opt_iterations,
                    trial_length=config.trial_length,
                    save_path=target_cache,
                    verbose=False,
                )

            perceptions: Dict[str, Any] = {}
            if "uniform" in config.perceptions:
                perceptions["uniform"] = UniformPerceptionModel()
            if "adversarial_opt" in config.perceptions:
                perceptions["adversarial_opt"] = opt_perceptions

            # Shield factories at this beta
            shield_factories = {}
            if "single_belief" in config.shields:
                shield_factories["single_belief"] = create_single_belief_shield_factory(
                    pomdp, pp_shield, beta
                )
            if "envelope" in config.shields:
                shield_factories["envelope"] = create_envelope_shield_factory(
                    ipomdp, pp_shield, beta
                )
            if "forward_sampling" in config.shields:
                shield_factories["forward_sampling"] = create_forward_sampling_shield_factory(
                    ipomdp, pp_shield, beta,
                    budget=config.fs_budget, K_samples=config.fs_K_samples,
                )

            for seed in config.seeds:
                for p_name, perception in perceptions.items():
                    for sh_name, sh_factory in shield_factories.items():
                        adv_target = ""
                        perception_model = perception
                        if p_name == "adversarial_opt" and isinstance(perception, dict):
                            if sh_name in perception:
                                perception_model = perception[sh_name]
                                adv_target = sh_name
                            elif "envelope" in perception:
                                perception_model = perception["envelope"]
                                adv_target = "envelope"
                            else:
                                perception_model = next(iter(perception.values()))
                                adv_target = "unknown"
                        trial_results = run_monte_carlo_trials(
                            ipomdp=ipomdp,
                            pp_shield=pp_shield,
                            perception=perception_model,
                            rt_shield_factory=sh_factory,
                            action_selector=rl_wrapped,
                            initial_generator=RandomInitialState(),
                            num_trials=config.num_trials,
                            trial_length=config.trial_length,
                            seed=seed,
                        )
                        metrics = compute_safety_metrics(trial_results)

                        row = {
                            "alpha": alpha,
                            "beta": beta,
                            "seed": seed,
                            "perception": p_name,
                            "adversarial_opt_target": adv_target,
                            "shield": sh_name,
                            "selector": "rl",
                            "fail_rate": metrics.fail_rate,
                            "stuck_rate": metrics.stuck_rate,
                            "safe_rate": metrics.safe_rate,
                            "mean_steps": metrics.mean_steps,
                            "num_trials": metrics.num_trials,
                        }
                        add_rate_cis(row, metrics.num_trials)
                        tidy_rows.append(row)

                        print(f"      {p_name}/{sh_name}/seed={seed}: "
                              f"fail={metrics.fail_rate:.1%} stuck={metrics.stuck_rate:.1%} "
                              f"safe={metrics.safe_rate:.1%}")

    total_time = time.time() - t0
    print(f"\nTotal sweep time: {total_time:.1f}s")

    csv_path = os.path.join(config.results_dir, "results_tidy.csv")
    if tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tidy_rows)
        print(f"Tidy CSV saved to {csv_path}")

    agg = _aggregate_sweep(tidy_rows)
    summary["aggregated"] = agg
    summary["total_time_s"] = total_time

    metadata = build_metadata(config, extra={"total_time_s": total_time})
    json_path = os.path.join(config.results_dir, "sweep_summary.json")
    save_experiment_results(json_path, summary, metadata)
    print(f"Summary saved to {json_path}")

    _plot_pareto(agg, config)
    _plot_alpha_trends(agg, config)

    return tidy_rows, summary


def _aggregate_sweep(tidy_rows, ci_alpha: float = 0.05):
    """Aggregate per-seed results per (alpha,beta,perception,shield).

    For binary trial outcomes (fail/stuck/safe), reports the trial-weighted
    pooled mean and a Clopper-Pearson 95% binomial CI over all trials across
    all seeds (n_trials_total per cell). For continuous metrics (mean_steps)
    falls back to mean and std of per-seed means.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in tidy_rows:
        key = (row["alpha"], row["beta"], row["perception"], row["shield"])
        groups[key].append(row)

    agg = []
    for (alpha, beta, perc, shield), rows in sorted(groups.items()):
        total_trials = int(sum(r["num_trials"] for r in rows))

        def _pool(field):
            if total_trials == 0:
                return 0.0, 0.0, 0.0, 0
            n_succ = int(round(sum(float(r[field]) * int(r["num_trials"]) for r in rows)))
            n_succ = max(0, min(total_trials, n_succ))
            p = n_succ / total_trials
            lo, hi = proportion_confint(n_succ, total_trials, alpha=ci_alpha, method="beta")
            return float(p), float(lo), float(hi), n_succ

        fail_p, fail_lo, fail_hi, fail_n = _pool("fail_rate")
        stuck_p, stuck_lo, stuck_hi, stuck_n = _pool("stuck_rate")
        safe_p, safe_lo, safe_hi, safe_n = _pool("safe_rate")
        mean_steps = [r["mean_steps"] for r in rows]
        agg.append({
            "alpha": alpha,
            "beta": beta,
            "perception": perc,
            "shield": shield,
            "fail_rate_mean": fail_p,
            "fail_rate_ci_low": fail_lo,
            "fail_rate_ci_high": fail_hi,
            "fail_count": fail_n,
            "stuck_rate_mean": stuck_p,
            "stuck_rate_ci_low": stuck_lo,
            "stuck_rate_ci_high": stuck_hi,
            "stuck_count": stuck_n,
            "safe_rate_mean": safe_p,
            "safe_rate_ci_low": safe_lo,
            "safe_rate_ci_high": safe_hi,
            "safe_count": safe_n,
            "mean_steps_mean": float(np.mean(mean_steps)),
            "mean_steps_std": float(np.std(mean_steps)),
            "n_seeds": len(rows),
            "n_trials_total": total_trials,
            "ci_alpha": ci_alpha,
        })
    return agg


PERCEPTION_LABELS = {
    "uniform": "Uniform Random",
    "adversarial_opt": "Adversarial Optimized",
}

SHIELD_STYLES = {
    "single_belief":    {"color": "steelblue",  "marker": "o", "label": "single_belief"},
    "envelope":         {"color": "seagreen",   "marker": "s", "label": "envelope"},
    "forward_sampling": {"color": "darkorange", "marker": "^", "label": "forward_sampling"},
}


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _filter(agg, perc, shield, beta):
    return [r for r in agg
            if r["perception"] == perc
            and r["shield"] == shield
            and abs(r["beta"] - beta) < 1e-9]


def _plot_pareto(agg, config):
    """Pareto SCATTER (no connecting lines), one panel per (perception, beta)."""
    try:
        plt = _setup_mpl()
    except ImportError:
        print("matplotlib not available, skipping Pareto plot")
        return

    figures_dir = os.path.join(config.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    betas = sorted(config.betas)
    perceptions_present = [p for p in config.perceptions
                           if any(r["perception"] == p for r in agg)]
    if not perceptions_present:
        print("  No aggregated rows to plot")
        return

    nr = len(perceptions_present)
    nc = len(betas)
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4.3 * nr), squeeze=False)

    for r, perc in enumerate(perceptions_present):
        for c, beta in enumerate(betas):
            ax = axes[r, c]
            for shield in config.shields:
                subset = _filter(agg, perc, shield, beta)
                if not subset:
                    continue
                subset.sort(key=lambda x: x["alpha"])
                xs = [x["stuck_rate_mean"] for x in subset]
                ys = [x["fail_rate_mean"] for x in subset]
                ts = [x["alpha"] for x in subset]
                style = SHIELD_STYLES.get(shield, {"color": None, "marker": "o", "label": shield})
                ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                           s=45, label=style["label"], zorder=3,
                           edgecolors="white", linewidths=0.5)
                # Annotate every alpha point (small font)
                for x, y, t in zip(xs, ys, ts):
                    ax.annotate(f"{t:g}", (x, y),
                                textcoords="offset points", xytext=(5, 3),
                                fontsize=6.5, color=style["color"])
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            if r == nr - 1:
                ax.set_xlabel("Stuck rate", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{PERCEPTION_LABELS.get(perc, perc)}\nFail rate", fontsize=9)
            if r == 0:
                ax.set_title(rf"$\beta={beta:g}$", fontsize=10)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=8)

    alphas = sorted(config.alphas)
    fig.suptitle(
        f"Pareto scatter — {config.case_study_name.upper()} "
        rf"(RL selector, $\alpha\in[{alphas[0]:g}, {alphas[-1]:g}]$, n={len(alphas)} points)",
        fontsize=11,
    )
    plt.tight_layout()
    fname = "pareto_alpha.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_alpha_trends(agg, config):
    """Line plots of fail rate vs alpha and stuck rate vs alpha per (perception, beta)."""
    try:
        plt = _setup_mpl()
    except ImportError:
        print("matplotlib not available, skipping trend plots")
        return

    figures_dir = os.path.join(config.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    betas = sorted(config.betas)
    perceptions_present = [p for p in config.perceptions
                           if any(r["perception"] == p for r in agg)]
    if not perceptions_present:
        return

    for metric_key, metric_label, suffix in [
        ("fail_rate_mean", "Fail rate", "fail"),
        ("stuck_rate_mean", "Stuck rate", "stuck"),
        ("safe_rate_mean", "Safe rate", "safe"),
    ]:
        nr = len(perceptions_present)
        nc = len(betas)
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4.0 * nr), squeeze=False)

        for r, perc in enumerate(perceptions_present):
            for c, beta in enumerate(betas):
                ax = axes[r, c]
                for shield in config.shields:
                    subset = _filter(agg, perc, shield, beta)
                    if not subset:
                        continue
                    subset.sort(key=lambda x: x["alpha"])
                    xs = [x["alpha"] for x in subset]
                    ys = [x[metric_key] for x in subset]
                    lo_key = metric_key.replace("_mean", "_ci_low")
                    hi_key = metric_key.replace("_mean", "_ci_high")
                    yerr_lo = [max(y - x[lo_key], 0.0) for x, y in zip(subset, ys)]
                    yerr_hi = [max(x[hi_key] - y, 0.0) for x, y in zip(subset, ys)]
                    style = SHIELD_STYLES.get(shield, {"color": None, "marker": "o", "label": shield})
                    ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker=style["marker"],
                                color=style["color"], label=style["label"],
                                linewidth=1.4, markersize=5,
                                capsize=2, elinewidth=0.8, alpha=0.9)
                ax.set_ylim(-0.05, 1.05)
                if r == nr - 1:
                    ax.set_xlabel(r"$\alpha$", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{PERCEPTION_LABELS.get(perc, perc)}\n{metric_label}", fontsize=9)
                if r == 0:
                    ax.set_title(rf"$\beta={beta:g}$", fontsize=10)
                ax.grid(True, alpha=0.3)
                if r == 0 and c == 0:
                    ax.legend(loc="best", fontsize=8)

        fig.suptitle(
            f"{metric_label} vs " + r"$\alpha$" + f" — {config.case_study_name.upper()} "
            + r"($\beta$ values " + f"{betas})",
            fontsize=11,
        )
        plt.tight_layout()
        fname = f"alpha_vs_{suffix}.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")
    print(f"All sweep figures saved to {figures_dir}")


def main():
    """Entry point for alpha sweep."""
    config_module_name = "sweeps.rl_alpha_sweep_taxinet"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--config="):
        config_module_name = sys.argv[1].split("=", 1)[1]
    elif len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_module_name = sys.argv[2]

    try:
        mod = importlib.import_module(
            f".{config_module_name}", package="ipomdp_shielding.experiments"
        )
        config = mod.config
    except ImportError:
        print(f"Config module '{config_module_name}' not found, using defaults")
        config = AlphaSweepConfig()

    run_alpha_sweep(config)


if __name__ == "__main__":
    main()
