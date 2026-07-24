"""Plot and summarize TaxiNetV2 operating sweep results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


POINT_METHODS = ("single_belief", "envelope", "forward_sampling")
POINT_LABELS = {
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Forward-Sampling",
}
POINT_COLORS = {
    "single_belief": "#1565C0",
    "envelope": "#E65100",
    "forward_sampling": "#00838F",
}
POINT_MARKERS = {
    "single_belief": "o",
    "envelope": "s",
    "forward_sampling": "P",
}
CONF_COLORS = {
    "0.95": "#8E24AA",
    "0.99": "#5E35B1",
    "0.995": "#283593",
}
CONF_MARKERS = {
    "0.95": "^",
    "0.99": "D",
    "0.995": "X",
}
CONFORMAL_COLOR = "#5E35B1"
BEST_BAR_ORDER = ("envelope", "forward_sampling", "single_belief", "conformal")
BEST_BAR_LABELS = {
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Fwd-Smp",
    "conformal": "Conformal",
}
BEST_BAR_COLORS = {
    "single_belief": POINT_COLORS["single_belief"],
    "envelope": POINT_COLORS["envelope"],
    "forward_sampling": POINT_COLORS["forward_sampling"],
    "conformal": CONFORMAL_COLOR,
}
BEST_BAR_SUBTITLE = {
    "safe": "Highest Safety per Method",
    "fail": "Lowest Failure per Method",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/results.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/figures"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/evaluation_summary.md"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _curve(results: dict, perception: str, method: str) -> List[Tuple[float, dict]]:
    points = [
        (float(beta_key), metrics)
        for beta_key, metrics in results["point_sweep"][perception][method].items()
    ]
    return sorted(points, key=lambda item: item[0])


def _conf_curve(results: dict, perception: str, confidence_level: str) -> List[Tuple[float, dict]]:
    points = [
        (float(filter_key), metrics)
        for filter_key, metrics in results["conformal_sweep"][perception][confidence_level].items()
    ]
    return sorted(points, key=lambda item: item[0])


def make_pareto_figure(results: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, perception in zip(axes, ("uniform", "adversarial_opt")):
        for method in POINT_METHODS:
            curve = _curve(results, perception, method)
            xs = [metrics["stuck_rate"] * 100 for _, metrics in curve]
            ys = [metrics["fail_rate"] * 100 for _, metrics in curve]
            ax.scatter(
                xs,
                ys,
                color=POINT_COLORS[method],
                marker=POINT_MARKERS[method],
                s=65,
                label=rf"{POINT_LABELS[method]} ($\beta$)",
            )
        for confidence_level in results["metadata"]["confidence_levels"]:
            curve = _conf_curve(results, perception, confidence_level)
            xs = [metrics["stuck_rate"] * 100 for _, metrics in curve]
            ys = [metrics["fail_rate"] * 100 for _, metrics in curve]
            ax.scatter(
                xs,
                ys,
                color=CONF_COLORS[confidence_level],
                marker=CONF_MARKERS[confidence_level],
                s=70,
                label=rf"Conformal $\kappa$={confidence_level}",
            )
        ax.set_title("Uniform perception" if perception == "uniform" else "Adversarial perception")
        ax.set_xlabel("Stuck rate (%)")
        ax.set_ylabel("Fail rate (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, framealpha=0.95)
    fig.suptitle(r"TaxiNetV2 RL Operating Sweep: $\beta$ for point shields, $\kappa/\tau$ for conformal", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_safe_score(metrics: dict) -> Tuple[float, float, float]:
    return (-metrics["safe_rate"], metrics["fail_rate"], metrics["stuck_rate"])


def _best_fail_score(metrics: dict) -> Tuple[float, float, float]:
    return (metrics["fail_rate"], metrics["stuck_rate"], -metrics["safe_rate"])


def _best_point_method(results: dict, perception: str, method: str, objective: str) -> Tuple[str, dict]:
    rows = [
        (rf"$\beta$={beta:.2f}", metrics)
        for beta, metrics in _curve(results, perception, method)
    ]
    scorer = _best_safe_score if objective == "safe" else _best_fail_score
    return min(rows, key=lambda item: scorer(item[1]))


def _best_conformal(results: dict, perception: str, objective: str) -> Tuple[str, dict]:
    rows = []
    for confidence_level in results["metadata"]["confidence_levels"]:
        for action_filter, metrics in _conf_curve(results, perception, confidence_level):
            rows.append((rf"$\kappa$={confidence_level}, $\tau$={action_filter:.1f}", metrics))
    scorer = _best_safe_score if objective == "safe" else _best_fail_score
    return min(rows, key=lambda item: scorer(item[1]))


def _best_bar_rows(results: dict, perception: str, objective: str) -> Dict[str, dict]:
    best_rows: Dict[str, dict] = {}
    for method in POINT_METHODS:
        setting, metrics = _best_point_method(results, perception, method, objective)
        best_rows[method] = {"setting": setting, **metrics}
    setting, metrics = _best_conformal(results, perception, objective)
    best_rows["conformal"] = {"setting": setting, **metrics}
    return best_rows


def _draw_best_bars(ax, best_rows: Dict[str, dict], title: str) -> None:
    methods = [method for method in BEST_BAR_ORDER if method in best_rows]
    fails = [best_rows[method]["fail_rate"] * 100 for method in methods]
    stucks = [best_rows[method]["stuck_rate"] * 100 for method in methods]
    safes = [best_rows[method]["safe_rate"] * 100 for method in methods]
    colors = [BEST_BAR_COLORS[method] for method in methods]
    x = np.arange(len(methods))
    width = 0.58

    ax.bar(x, fails, width, color=colors, alpha=0.9, zorder=3)
    ax.bar(
        x,
        stucks,
        width,
        bottom=fails,
        color=colors,
        alpha=0.34,
        hatch="///",
        edgecolor="white",
        zorder=3,
    )
    ax.bar(
        x,
        safes,
        width,
        bottom=np.array(fails) + np.array(stucks),
        color="white",
        edgecolor=colors,
        linewidth=1.4,
        zorder=3,
    )

    for i, method in enumerate(methods):
        total = fails[i] + stucks[i] + safes[i]
        setting = (
            best_rows[method]["setting"]
            .replace(", ", "\n")
        )
        ax.text(x[i], total + 1.2, setting, ha="center", va="bottom", fontsize=7.2)
        if fails[i] >= 6:
            ax.text(
                x[i],
                fails[i] / 2,
                f"{fails[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
        if stucks[i] >= 6:
            ax.text(
                x[i],
                fails[i] + stucks[i] / 2,
                f"{stucks[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                fontweight="bold",
            )
        if safes[i] >= 6:
            ax.text(
                x[i],
                fails[i] + stucks[i] + safes[i] / 2,
                f"{safes[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color="#263238",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([BEST_BAR_LABELS[method] for method in methods], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 114)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def make_combined_bar_figure(results: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.1), sharey=True)

    panel_specs = [
        ("fail", "uniform", "Lowest Failure / Uniform"),
        ("fail", "adversarial_opt", "Lowest Failure / Adversarial"),
        ("safe", "uniform", "Highest Safe / Uniform"),
        ("safe", "adversarial_opt", "Highest Safe / Adversarial"),
    ]
    for ax, (objective, perception, title) in zip(axes.flat, panel_specs):
        rows = _best_bar_rows(results, perception, objective)
        _draw_best_bars(ax, rows, title)

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.9, label="Fail %")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.34, hatch="///", edgecolor="white", label="Stuck %")
    safe_patch = mpatches.Patch(facecolor="white", edgecolor="#455A64", linewidth=1.2, label="Safe %")
    fig.legend(
        handles=[fail_patch, stuck_patch, safe_patch],
        fontsize=8,
        loc="lower center",
        ncol=3,
        framealpha=0.95,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_best_bar_figure(results: dict, out_path: Path, objective: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.suptitle(BEST_BAR_SUBTITLE[objective], fontsize=11)

    for ax, perception in zip(axes, ("uniform", "adversarial_opt")):
        rows = _best_bar_rows(results, perception, objective)
        title = "Uniform perception" if perception == "uniform" else "Adversarial perception"
        _draw_best_bars(ax, rows, title)

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.9, label="Fail %")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.34, hatch="///", edgecolor="white", label="Stuck %")
    safe_patch = mpatches.Patch(facecolor="white", edgecolor="#455A64", linewidth=1.2, label="Safe %")
    axes[1].legend(
        handles=[fail_patch, stuck_patch, safe_patch],
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_entry(rows: Iterable[Tuple[str, dict]], key: str) -> Tuple[str, dict]:
    return min(rows, key=lambda item: item[1][key])


def _iter_perception_rows(results: dict, perception: str) -> Iterable[Tuple[str, dict]]:
    for method in POINT_METHODS:
        for beta, metrics in _curve(results, perception, method):
            yield (rf"{POINT_LABELS[method]} $\beta$={beta:.2f}", metrics)
    for confidence_level in results["metadata"]["confidence_levels"]:
        for action_filter, metrics in _conf_curve(results, perception, confidence_level):
            yield (rf"Conformal $\kappa$={confidence_level}, $\tau$={action_filter:.1f}", metrics)


def _table_lines(results: dict, perception: str) -> List[str]:
    lines = [
        "| Operating point | Fail | Stuck | Safe | Intervention |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in _iter_perception_rows(results, perception):
        lines.append(
            f"| {label} | {metrics['fail_rate']:.1%} | {metrics['stuck_rate']:.1%} | "
            f"{metrics['safe_rate']:.1%} | {metrics['intervention_rate']:.1%} |"
        )
    return lines


def _best_table_lines(best_rows: Dict[str, dict]) -> List[str]:
    return [
        f"| {BEST_BAR_LABELS[method]} | `{best_rows[method]['setting']}` | "
        f"{best_rows[method]['fail_rate']:.1%} | "
        f"{best_rows[method]['stuck_rate']:.1%} | "
        f"{best_rows[method]['safe_rate']:.1%} |"
        for method in BEST_BAR_ORDER
    ]


def write_summary(
    results: dict,
    figure_path: Path,
    safest_bar_figure_path: Path,
    lowest_fail_bar_figure_path: Path,
    combined_bar_figure_path: Path,
    summary_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    meta = results["metadata"]
    uniform_best_fail = _best_entry(_iter_perception_rows(results, "uniform"), "fail_rate")
    adv_best_fail = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "fail_rate")
    uniform_best_stuck = _best_entry(_iter_perception_rows(results, "uniform"), "stuck_rate")
    adv_best_stuck = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "stuck_rate")
    uniform_safest_by_method = _best_bar_rows(results, "uniform", "safe")
    adv_safest_by_method = _best_bar_rows(results, "adversarial_opt", "safe")
    uniform_lowest_fail_by_method = _best_bar_rows(results, "uniform", "fail")
    adv_lowest_fail_by_method = _best_bar_rows(results, "adversarial_opt", "fail")
    lines: List[str] = [
        "# TaxiNetV2 Operating Pareto Sweep",
        "",
        "## Setup",
        "",
        f"- Selector: `{meta['selector']}` only.",
        f"- Shared RL controller cache: `{meta['setup']['rl_cache_path']}`.",
        f"- Shared adversarial realization cache: `{meta['setup']['opt_cache_paths_by_target']['envelope']}`.",
        f"- Beta grid for point shields: `{meta['beta_values']}`.",
        f"- Conformal confidence levels: `{meta['confidence_levels']}`.",
        f"- Conformal action-filter grid: `{meta['action_filter_values']}`.",
        f"- Trials per point: `{meta['run_config']['num_trials']}` with horizon `{meta['run_config']['trial_length']}` and seed `{meta['run_config']['seed']}`.",
        "",
        "## High-Level Findings",
        "",
        f"- Uniform lowest fail: `{uniform_best_fail[0]}` at `{uniform_best_fail[1]['fail_rate']:.1%}` fail and `{uniform_best_fail[1]['stuck_rate']:.1%}` stuck.",
        f"- Adversarial lowest fail: `{adv_best_fail[0]}` at `{adv_best_fail[1]['fail_rate']:.1%}` fail and `{adv_best_fail[1]['stuck_rate']:.1%}` stuck.",
        f"- Uniform lowest stuck: `{uniform_best_stuck[0]}` at `{uniform_best_stuck[1]['stuck_rate']:.1%}` stuck and `{uniform_best_stuck[1]['fail_rate']:.1%}` fail.",
        f"- Adversarial lowest stuck: `{adv_best_stuck[0]}` at `{adv_best_stuck[1]['stuck_rate']:.1%}` stuck and `{adv_best_stuck[1]['fail_rate']:.1%}` fail.",
        "- Only the conformal shield moves with `conf` and `af`; `single_belief`, `envelope`, and `forward_sampling` move only with `beta`.",
        "- The RL controller and adversarial realization are held fixed across the whole sweep, so the plotted differences are shield-operating-point differences rather than controller/retraining differences.",
        "- The safest stacked-bar figure keeps each method's highest-safe operating point, tie-broken by lower fail and then lower stuck.",
        "- The lowest-fail stacked-bar figure keeps each method's lowest-fail operating point, tie-broken by lower stuck and then higher safe.",
        "",
        "## Safest Point per Method",
        "",
        "### Uniform perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *_best_table_lines(uniform_safest_by_method),
        "",
        "### Shared adversarial perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *_best_table_lines(adv_safest_by_method),
        "",
        "## Lowest-Fail Point per Method",
        "",
        "### Uniform perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *_best_table_lines(uniform_lowest_fail_by_method),
        "",
        "### Shared adversarial perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *_best_table_lines(adv_lowest_fail_by_method),
        "",
        "## Uniform RL Results",
        "",
        *_table_lines(results, "uniform"),
        "",
        "## Adversarial RL Results",
        "",
        *_table_lines(results, "adversarial_opt"),
        "",
        "## Figure",
        "",
        f"- Pareto scatter: ![]({figure_path.relative_to(summary_path.parent)})",
        f"- Safest-point bars: ![]({safest_bar_figure_path.relative_to(summary_path.parent)})",
        f"- Lowest-fail bars: ![]({lowest_fail_bar_figure_path.relative_to(summary_path.parent)})",
        f"- Combined operating-point bars: ![]({combined_bar_figure_path.relative_to(summary_path.parent)})",
    ]
    summary_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results = _load(args.results_json)
    figure_path = make_pareto_figure(results, args.figures_dir / "pareto_scatter.png")
    safest_bar_figure_path = make_best_bar_figure(results, args.figures_dir / "safest_method_bars.png", "safe")
    lowest_fail_bar_figure_path = make_best_bar_figure(results, args.figures_dir / "lowest_fail_method_bars.png", "fail")
    combined_bar_figure_path = make_combined_bar_figure(results, args.figures_dir / "operating_point_method_bars_2x2.png")
    old_bar_figure_path = args.figures_dir / "best_method_bars.png"
    if old_bar_figure_path.exists():
        os.remove(old_bar_figure_path)
    write_summary(
        results,
        figure_path,
        safest_bar_figure_path,
        lowest_fail_bar_figure_path,
        combined_bar_figure_path,
        args.summary_md,
    )
    print(f"Wrote figure to {figure_path}")
    print(f"Wrote figure to {safest_bar_figure_path}")
    print(f"Wrote figure to {lowest_fail_bar_figure_path}")
    print(f"Wrote figure to {combined_bar_figure_path}")
    print(f"Wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
