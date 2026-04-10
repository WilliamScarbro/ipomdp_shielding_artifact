"""Generate the artifact evaluation summary from bundled experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("results/experiment")
DATA = BASE / "threshold"
OBS = BASE / "obs"
FS = BASE / "fs"
OUTDIR = BASE
MD_PATH = Path("evaluation_summary.md")

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
LABEL_T = {0.90, 0.95}

COLORS = {
    "envelope": "#E65100",
    "single_belief": "#1565C0",
    "observation": "#2E7D32",
    "carr": "#6A1B9A",
    "forward_sampling": "#00838F",
}
MARKERS = {
    "envelope": "s",
    "single_belief": "o",
    "observation": "^",
    "carr": "D",
    "forward_sampling": "P",
}
SHIELD_LABELS = {
    "envelope": "Envelope",
    "single_belief": "Single-Belief",
    "observation": "Observation",
    "carr": "Carr",
    "forward_sampling": "Fwd-Sampling",
}
BAR_ORDER = ["envelope", "single_belief", "observation", "carr", "forward_sampling"]

CASES = {
    "taxinet": {
        "label": "TaxiNet\n(16 states, 16 obs)",
        "long": "TaxiNet (16 states, 16 obs)",
        "sweep": DATA / "taxinet_sweep.json",
        "carr": BASE / "carr" / "taxinet_carr_results.json",
        "obs": OBS / "taxinet_obs_sweep.json",
        "fs": FS / "taxinet_fs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "obstacle": {
        "label": "Obstacle\n(50 states, 3 obs)",
        "long": "Obstacle (50 states, 3 obs)",
        "sweep": DATA / "obstacle_sweep.json",
        "carr": BASE / "carr" / "obstacle_carr_results.json",
        "obs": OBS / "obstacle_obs_sweep.json",
        "fs": FS / "obstacle_fs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "cartpole_lowacc": {
        "label": "CartPole low-acc\n(82 states, P_mid=0.373)",
        "long": "CartPole low-acc (82 states, P_mid=0.373)",
        "sweep": DATA / "cartpole_lowacc_sweep.json",
        "carr": BASE / "carr" / "cartpole_lowacc_carr_results.json",
        "obs": OBS / "cartpole_lowacc_obs_sweep.json",
        "fs": FS / "cartpole_lowacc_fs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
    "refuel_v2": {
        "label": "Refuel v2\n(344 states, 29 obs)",
        "long": "Refuel v2 (344 states, 29 obs)",
        "sweep": DATA / "refuel_v2_sweep.json",
        "carr": None,
        "obs": OBS / "refuel_v2_obs_sweep.json",
        "fs": FS / "refuel_v2_fs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
}


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def sweep_points(data: dict, perception: str, shield: str) -> list[tuple]:
    sweep_results = data.get("sweep_results", {})
    out = []
    for threshold in THRESHOLDS:
        row = sweep_results.get(f"{threshold:.2f}", {}).get(f"{perception}/rl/{shield}")
        if row:
            out.append((row["fail_rate"], row["stuck_rate"], threshold))
    return out


def best_point(data: dict, perception: str, shield: str):
    pts = sweep_points(data, perception, shield)
    if not pts:
        return None
    fail_rate, stuck_rate, threshold = min(pts, key=lambda point: (point[0], point[1]))
    return (threshold, fail_rate, stuck_rate)


def carr_result(data, perception: str):
    if data is None or data.get("status") == "infeasible":
        return None
    row = data.get("results", {}).get(f"{perception}/rl/carr")
    return (row["fail_rate"], row["stuck_rate"]) if row else None


def collect_best(sweep_data, obs_data, carr_data, perception: str, sweep_shields: list[str], fs_data=None) -> dict:
    raw = {}
    for shield in sweep_shields:
        point = best_point(sweep_data, perception, shield)
        if point:
            raw[shield] = {"fail": point[1], "stuck": point[2], "threshold": point[0]}
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            fail_rate, stuck_rate, threshold = min(pts, key=lambda point: (point[0], point[1]))
            raw["observation"] = {"fail": fail_rate, "stuck": stuck_rate, "threshold": threshold}
    carr = carr_result(carr_data, perception)
    if carr:
        raw["carr"] = {"fail": carr[0], "stuck": carr[1], "threshold": None}
    if fs_data:
        pts = sweep_points(fs_data, perception, "forward_sampling")
        if pts:
            fail_rate, stuck_rate, threshold = min(pts, key=lambda point: (point[0], point[1]))
            raw["forward_sampling"] = {"fail": fail_rate, "stuck": stuck_rate, "threshold": threshold}
    return {shield: raw[shield] for shield in BAR_ORDER if shield in raw}


def _draw_pareto(ax, curves: dict, carr_pt, title: str) -> None:
    for shield, pts in curves.items():
        color = COLORS[shield]
        marker = MARKERS[shield]
        for fail_rate, stuck_rate, threshold in pts:
            ax.scatter(fail_rate * 100, stuck_rate * 100, color=color, marker=marker, s=55, zorder=3)
            if threshold in LABEL_T:
                ax.annotate(
                    f"{threshold:.2f}",
                    (fail_rate * 100, stuck_rate * 100),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                    color=color,
                )
    if carr_pt:
        fail_rate, stuck_rate = carr_pt
        ax.scatter(fail_rate * 100, stuck_rate * 100, color=COLORS["carr"], marker="D", s=90, zorder=4)
        ax.annotate(
            "Carr",
            (fail_rate * 100, stuck_rate * 100),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7.5,
            color=COLORS["carr"],
            fontweight="bold",
        )
    ax.set_xlabel("Fail rate (%)")
    ax.set_ylabel("Stuck rate (%)")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)


def make_pareto_figure(case_name: str, case_cfg: dict) -> str:
    sweep = load_json(case_cfg["sweep"])
    obs = load_json(case_cfg["obs"]) if case_cfg["obs"] else None
    carr = load_json(case_cfg["carr"]) if case_cfg["carr"] else None
    fs = load_json(case_cfg["fs"]) if case_cfg.get("fs") and case_cfg["fs"].exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(f"Pareto Scatter — {case_cfg['long']}", fontsize=10)

    for ax, perception, label in zip(
        axes,
        ["uniform", "adversarial_opt"],
        ["Uniform perception", "Adversarial perception"],
    ):
        curves = {shield: sweep_points(sweep, perception, shield) for shield in case_cfg["sweep_shields"]}
        if obs:
            curves["observation"] = sweep_points(obs, perception, "observation")
        if fs:
            curves["forward_sampling"] = sweep_points(fs, perception, "forward_sampling")
        _draw_pareto(ax, curves, carr_result(carr, perception), label)

    legend_shields = [
        shield
        for shield in BAR_ORDER
        if shield in case_cfg["sweep_shields"]
        or shield == "observation"
        or (shield == "carr" and carr is not None)
        or (shield == "forward_sampling" and fs is not None)
    ]
    handles = [
        plt.scatter([], [], color=COLORS[shield], marker=MARKERS[shield], s=55, label=SHIELD_LABELS[shield])
        for shield in legend_shields
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = OUTDIR / f"pareto_experiment_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


def _draw_bars(ax, best_dict: dict, title: str, ylim: float = 115) -> None:
    shields = list(best_dict.keys())
    fails = [best_dict[shield]["fail"] * 100 for shield in shields]
    stucks = [best_dict[shield]["stuck"] * 100 for shield in shields]
    colors = [COLORS.get(shield, "#888") for shield in shields]
    positions, width = np.arange(len(shields)), 0.52

    ax.bar(positions, fails, width, color=colors, alpha=0.90, zorder=3)
    ax.bar(positions, stucks, width, bottom=fails, color=colors, alpha=0.32, hatch="///", edgecolor="white", zorder=3)

    for idx, shield in enumerate(shields):
        total = fails[idx] + stucks[idx]
        threshold = best_dict[shield].get("threshold")
        label = f"t={threshold:.2f}" if threshold is not None else "—"
        ax.text(positions[idx], total + ylim * 0.015, label, ha="center", va="bottom", fontsize=7.5)
        if fails[idx] >= 6:
            ax.text(
                positions[idx],
                fails[idx] / 2,
                f"{fails[idx]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([SHIELD_LABELS.get(shield, shield) for shield in shields], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def make_bar_figure(case_name: str, case_cfg: dict) -> str:
    sweep = load_json(case_cfg["sweep"])
    obs = load_json(case_cfg["obs"]) if case_cfg["obs"] else None
    carr = load_json(case_cfg["carr"]) if case_cfg["carr"] else None
    fs = load_json(case_cfg["fs"]) if case_cfg.get("fs") and case_cfg["fs"].exists() else None

    totals = []
    for perception in ["uniform", "adversarial_opt"]:
        best = collect_best(sweep, obs, carr, perception, case_cfg["sweep_shields"], fs_data=fs)
        for row in best.values():
            totals.append((row["fail"] + row["stuck"]) * 100)
    ylim = max(max(totals) * 1.18, 12) if totals else 12

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharey=True)
    fig.suptitle(f"Best Operating Point — {case_cfg['long']}", fontsize=10)

    for ax, perception, label in zip(
        axes,
        ["uniform", "adversarial_opt"],
        ["Uniform perception", "Adversarial perception"],
    ):
        best = collect_best(sweep, obs, carr, perception, case_cfg["sweep_shields"], fs_data=fs)
        _draw_bars(ax, best, label, ylim=ylim)

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///", edgecolor="white", label="Stuck %")
    axes[1].legend(handles=[fail_patch, stuck_patch], fontsize=8, loc="upper left" if max(totals) > 80 else "upper right")

    plt.tight_layout()
    out = OUTDIR / f"barchart_experiment_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


def make_summary_figure() -> str:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Best Operating Point — All Case Studies", fontsize=12, y=1.01)

    for col, (case_name, case_cfg) in enumerate(CASES.items()):
        sweep = load_json(case_cfg["sweep"])
        obs = load_json(case_cfg["obs"]) if case_cfg["obs"] else None
        carr = load_json(case_cfg["carr"]) if case_cfg["carr"] else None
        fs = load_json(case_cfg["fs"]) if case_cfg.get("fs") and case_cfg["fs"].exists() else None

        totals = []
        for perception in ["uniform", "adversarial_opt"]:
            best = collect_best(sweep, obs, carr, perception, case_cfg["sweep_shields"], fs_data=fs)
            for row in best.values():
                totals.append((row["fail"] + row["stuck"]) * 100)
        ylim = max(max(totals) * 1.18, 12)

        for row_index, (perception, label) in enumerate([("uniform", "Uniform"), ("adversarial_opt", "Adversarial")]):
            ax = axes[row_index][col]
            best = collect_best(sweep, obs, carr, perception, case_cfg["sweep_shields"], fs_data=fs)
            _draw_bars(ax, best, f"{case_cfg['label']}\n({label})", ylim=ylim)
            if col > 0:
                ax.set_ylabel("")

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///", edgecolor="white", label="Stuck %")
    fig.legend(handles=[fail_patch, stuck_patch], fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    out = OUTDIR / "summary_experiment_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _best_row(sweep_data, obs_data, carr_data, case_cfg, perception, fs_data=None) -> dict:
    row = {}
    for shield in case_cfg["sweep_shields"]:
        point = best_point(sweep_data, perception, shield)
        if point:
            row[shield] = point
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            fail_rate, stuck_rate, threshold = min(pts, key=lambda point: (point[0], point[1]))
            row["observation"] = (threshold, fail_rate, stuck_rate)
    carr = carr_result(carr_data, perception)
    if carr:
        row["carr"] = (None, carr[0], carr[1])
    if fs_data:
        pts = sweep_points(fs_data, perception, "forward_sampling")
        if pts:
            fail_rate, stuck_rate, threshold = min(pts, key=lambda point: (point[0], point[1]))
            row["forward_sampling"] = (threshold, fail_rate, stuck_rate)
    return row


def generate_markdown(figures: dict) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        "**Shields compared**: Envelope, Single-Belief, Observation, Carr, Fwd-Sampling  ",
        "*(where feasible — see per-case notes)*",
        "",
        "**Case studies**:",
        "TaxiNet (16 states, 16 obs) · Obstacle (50 states, 3 obs) · "
        "CartPole low-acc (82 states, P_mid=0.373) · Refuel v2 (344 states, 29 obs)",
        "",
        "**Trials**: 200 per combination. Bar charts show the best operating",
        "threshold (min fail%, then min stuck%) for each shield.",
        "Pareto plots are shown only for TaxiNet and Obstacle.",
        "",
        "---",
        "",
        "## Cross-Case Summary",
        "",
        f"![Overview bar charts]({figures['summary']})",
        "",
        "### Best operating points",
        "",
        "| Case study | Shield | t | Fail% (unif) | Stuck% (unif) | Fail% (adv) | Stuck% (adv) |",
        "|---|---|---|---|---|---|---|",
    ]

    for case_name, case_cfg in CASES.items():
        sweep = load_json(case_cfg["sweep"])
        obs = load_json(case_cfg["obs"]) if case_cfg["obs"] else None
        carr = load_json(case_cfg["carr"]) if case_cfg["carr"] else None
        fs = load_json(case_cfg["fs"]) if case_cfg.get("fs") and case_cfg["fs"].exists() else None
        uniform = _best_row(sweep, obs, carr, case_cfg, "uniform", fs_data=fs)
        adversarial = _best_row(sweep, obs, carr, case_cfg, "adversarial_opt", fs_data=fs)
        shields = [shield for shield in BAR_ORDER if shield in uniform or shield in adversarial]
        for idx, shield in enumerate(shields):
            case_col = case_cfg["long"] if idx == 0 else ""
            threshold = f"{uniform[shield][0]:.2f}" if shield in uniform and uniform[shield][0] is not None else "—"
            fail_uniform = _pct(uniform[shield][1]) if shield in uniform else "N/A"
            stuck_uniform = _pct(uniform[shield][2]) if shield in uniform else "N/A"
            fail_adv = _pct(adversarial[shield][1]) if shield in adversarial else "N/A"
            stuck_adv = _pct(adversarial[shield][2]) if shield in adversarial else "N/A"
            lines.append(
                f"| {case_col} | {SHIELD_LABELS[shield]} | {threshold} | {fail_uniform} | {stuck_uniform} | {fail_adv} | {stuck_adv} |"
            )
    lines.append("")

    for case_name, case_cfg in CASES.items():
        sweep = load_json(case_cfg["sweep"])
        obs = load_json(case_cfg["obs"]) if case_cfg["obs"] else None
        carr = load_json(case_cfg["carr"]) if case_cfg["carr"] else None
        fs = load_json(case_cfg["fs"]) if case_cfg.get("fs") and case_cfg["fs"].exists() else None

        lines.extend([f"## {case_cfg['long']}", ""])
        if case_cfg["pareto"]:
            lines.extend([f"![Pareto scatter]({figures['pareto'][case_name]})", ""])
        lines.extend([f"![Bar chart — best threshold per shield]({figures['bar'][case_name]})", ""])
        lines.extend(
            [
                "### Best operating points",
                "",
                "| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        uniform = _best_row(sweep, obs, carr, case_cfg, "uniform", fs_data=fs)
        adversarial = _best_row(sweep, obs, carr, case_cfg, "adversarial_opt", fs_data=fs)
        for shield in BAR_ORDER:
            if shield not in uniform and shield not in adversarial:
                continue
            t_uniform = f"{uniform[shield][0]:.2f}" if shield in uniform and uniform[shield][0] is not None else "—"
            fail_uniform = _pct(uniform[shield][1]) if shield in uniform else "N/A"
            stuck_uniform = _pct(uniform[shield][2]) if shield in uniform else "N/A"
            t_adv = f"{adversarial[shield][0]:.2f}" if shield in adversarial and adversarial[shield][0] is not None else "—"
            fail_adv = _pct(adversarial[shield][1]) if shield in adversarial else "N/A"
            stuck_adv = _pct(adversarial[shield][2]) if shield in adversarial else "N/A"
            lines.append(
                f"| {SHIELD_LABELS[shield]} | {t_uniform} | {fail_uniform} | {stuck_uniform} | {t_adv} | {fail_adv} | {stuck_adv} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    figures = {"pareto": {}, "bar": {}, "summary": None}
    for case_name, case_cfg in CASES.items():
        print(f"\n── {case_cfg['long']} ──")
        if case_cfg["pareto"]:
            figures["pareto"][case_name] = make_pareto_figure(case_name, case_cfg)
            print(f"  Pareto → {figures['pareto'][case_name]}")
        figures["bar"][case_name] = make_bar_figure(case_name, case_cfg)
        print(f"  Bar    → {figures['bar'][case_name]}")

    figures["summary"] = make_summary_figure()
    print(f"\n  Summary → {figures['summary']}")

    markdown = generate_markdown(figures)
    MD_PATH.write_text(markdown)
    print(f"\n  Markdown → {MD_PATH}")


if __name__ == "__main__":
    main()
