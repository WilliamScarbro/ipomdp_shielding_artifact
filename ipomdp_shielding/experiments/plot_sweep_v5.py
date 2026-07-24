"""Generate v5 evaluation summary: Pareto scatter plots + stacked bar charts.

- Pareto scatter (no connecting lines, subset of threshold labels) for case
  studies where data spans both axes: TaxiNet, Obstacle.
- Stacked bar charts (fail% solid + stuck% hatched) at the best threshold per
  shield, for all four case studies.
- Separate uniform / adversarial subplots in every figure.
- Only most-recent variants: CartPole low-accuracy (P_mid=0.373), Refuel v2.

Usage:
    python -m ipomdp_shielding.experiments.plot_sweep_v5
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
VERSION  = "v5"
BASE     = Path("results")
DATA     = BASE / "threshold_sweep_expanded"
OBS      = BASE / "observation_shield_sweep"
FS       = BASE / "forward_sampling_sweep"
TIMING   = BASE / "timing_benchmark" / "shield_timing.json"
OUTDIR   = BASE / f"sweep_{VERSION}"
MD_PATH  = OUTDIR / "evaluation_summary.md"

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
LABEL_T    = {0.90, 0.95}          # thresholds annotated on scatter plots

# ── visual style ───────────────────────────────────────────────────────────────
COLORS = {
    "envelope":         "#E65100",   # deep orange
    "single_belief":    "#1565C0",   # blue
    "observation":      "#2E7D32",   # dark green
    "carr":             "#6A1B9A",   # purple
    "forward_sampling": "#00838F",   # teal
}
MARKERS = {
    "envelope":         "s",
    "single_belief":    "o",
    "observation":      "^",
    "carr":             "D",
    "forward_sampling": "P",
}
SHIELD_LABELS = {
    "envelope":         "Envelope",
    "single_belief":    "Single-Belief",
    "observation":      "Observation",
    "carr":             "Carr",
    "forward_sampling": "Fwd-Sampling",
}
# Display order for bar charts (left → right)
BAR_ORDER = ["envelope", "single_belief", "observation", "carr", "forward_sampling"]

# ── case study configuration ───────────────────────────────────────────────────
CASES = {
    "taxinet": {
        "label": "TaxiNet\n(16 states, 16 obs)",
        "long":  "TaxiNet (16 states, 16 obs)",
        "sweep": DATA / "taxinet_sweep.json",
        "carr":  DATA / "taxinet_carr_results.json",
        "obs":   OBS  / "taxinet_obs_sweep.json",
        "fs":    FS   / "taxinet_fs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "obstacle": {
        "label": "Obstacle\n(50 states, 3 obs)",
        "long":  "Obstacle (50 states, 3 obs)",
        "sweep": DATA / "obstacle_sweep.json",
        "carr":  DATA / "obstacle_carr_results.json",
        "obs":   OBS  / "obstacle_obs_sweep.json",
        "fs":    FS   / "obstacle_fs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "cartpole_lowacc": {
        "label": "CartPole low-acc\n(82 states, P_mid=0.373)",
        "long":  "CartPole low-acc (82 states, P_mid=0.373)",
        "sweep": DATA / "cartpole_lowacc_sweep.json",
        "carr":  DATA / "cartpole_lowacc_carr_results.json",
        "obs":   OBS  / "cartpole_lowacc_obs_sweep.json",
        "fs":    FS   / "cartpole_lowacc_fs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
    "refuel_v2": {
        "label": "Refuel v2\n(344 states, 29 obs)",
        "long":  "Refuel v2 (344 states, 29 obs)",
        "sweep": DATA / "refuel_v2_sweep.json",
        "carr":  None,              # support-MDP BFS infeasible
        "obs":   OBS  / "refuel_v2_obs_sweep.json",
        "fs":    FS   / "refuel_v2_fs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
}

# ── data helpers ───────────────────────────────────────────────────────────────

def load_json(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def sweep_points(data: dict, perception: str, shield: str) -> list[tuple]:
    """Return [(fail, stuck, t)] for each threshold present in sweep data."""
    sr = data.get("sweep_results", {})
    out = []
    for t in THRESHOLDS:
        r = sr.get(f"{t:.2f}", {}).get(f"{perception}/rl/{shield}")
        if r:
            out.append((r["fail_rate"], r["stuck_rate"], t))
    return out


def _sort_key(objective: str):
    """Key function for selecting the 'best' point under a given objective.

    objective="fail": min fail, then min stuck.
    objective="safe": max safe (= min -safe), then min fail, then min stuck.
    """
    if objective == "fail":
        return lambda p: (p[0], p[1])
    if objective == "safe":
        return lambda p: (-(1.0 - p[0] - p[1]), p[0], p[1])
    raise ValueError(f"unknown objective: {objective!r}")


def best_point(data: dict, perception: str, shield: str, objective: str = "fail"):
    """(t, fail, stuck) selected per `objective`."""
    pts = sweep_points(data, perception, shield)
    if not pts:
        return None
    f, s, t = min(pts, key=_sort_key(objective))
    return (t, f, s)


def carr_result(data, perception: str):
    """(fail, stuck) for Carr shield, or None."""
    if data is None or data.get("status") == "infeasible":
        return None
    r = data.get("results", {}).get(f"{perception}/rl/carr")
    return (r["fail_rate"], r["stuck_rate"]) if r else None


def collect_best(sweep_data, obs_data, carr_data, perception: str,
                 sweep_shields: list[str], fs_data=None,
                 objective: str = "fail") -> dict:
    """Build ordered {shield: {fail, stuck, threshold}} for bar chart."""
    key = _sort_key(objective)
    raw = {}
    for s in sweep_shields:
        bp = best_point(sweep_data, perception, s, objective)
        if bp:
            raw[s] = {"fail": bp[1], "stuck": bp[2], "threshold": bp[0]}
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            f, s, t = min(pts, key=key)
            raw["observation"] = {"fail": f, "stuck": s, "threshold": t}
    cr = carr_result(carr_data, perception)
    if cr:
        raw["carr"] = {"fail": cr[0], "stuck": cr[1], "threshold": None}
    if fs_data:
        pts = sweep_points(fs_data, perception, "forward_sampling")
        if pts:
            f, s, t = min(pts, key=key)
            raw["forward_sampling"] = {"fail": f, "stuck": s, "threshold": t}
    return {s: raw[s] for s in BAR_ORDER if s in raw}


# ── pareto scatter ─────────────────────────────────────────────────────────────

def _draw_pareto(ax, curves: dict, carr_pt, title: str) -> None:
    for shield, pts in curves.items():
        c, m = COLORS[shield], MARKERS[shield]
        for f, s, t in pts:
            ax.scatter(f * 100, s * 100, color=c, marker=m, s=55, zorder=3)
            if t in LABEL_T:
                ax.annotate(f"{t:.2f}", (f * 100, s * 100),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=c)
    if carr_pt:
        f, s = carr_pt
        ax.scatter(f * 100, s * 100, color=COLORS["carr"],
                   marker="D", s=90, zorder=4)
        ax.annotate("Carr", (f * 100, s * 100),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7.5, color=COLORS["carr"], fontweight="bold")
    ax.set_xlabel("Fail rate (%)")
    ax.set_ylabel("Stuck rate (%)")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)


def make_pareto_figure(cs_name: str, cs_cfg: dict) -> str:
    sweep = load_json(cs_cfg["sweep"])
    obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
    carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
    fs_path = cs_cfg.get("fs")
    fs    = load_json(fs_path) if fs_path and fs_path.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(f"Pareto Scatter — {cs_cfg['long']}", fontsize=10)

    for ax, perc, plbl in zip(axes,
                               ["uniform", "adversarial_opt"],
                               ["Uniform perception", "Adversarial perception"]):
        curves = {s: sweep_points(sweep, perc, s) for s in cs_cfg["sweep_shields"]}
        if obs:
            curves["observation"] = sweep_points(obs, perc, "observation")
        if fs:
            curves["forward_sampling"] = sweep_points(fs, perc, "forward_sampling")
        _draw_pareto(ax, curves, carr_result(carr, perc), plbl)

    legend_shields = (
        [s for s in BAR_ORDER
         if s in cs_cfg["sweep_shields"] or s == "observation" or
         (s == "carr" and carr is not None) or
         (s == "forward_sampling" and fs is not None)]
    )
    handles = [
        plt.scatter([], [], color=COLORS[s], marker=MARKERS[s],
                    s=55, label=SHIELD_LABELS[s])
        for s in legend_shields
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = OUTDIR / f"pareto_{VERSION}_{cs_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── bar chart ──────────────────────────────────────────────────────────────────

def _piecewise_cartpole_y(y: float) -> float:
    """Display 0-5% linearly, then compress 5-100% into the same visual height."""
    if y <= 5:
        return y
    return 5 + (y - 5) / 19


def _format_piecewise_cartpole_axis(ax) -> None:
    raw_ticks = [0, 1, 2, 3, 4, 5, 20, 40, 60, 80, 100]
    ax.set_yticks([_piecewise_cartpole_y(t) for t in raw_ticks])
    ax.set_yticklabels([str(t) for t in raw_ticks])
    ax.set_ylim(0, _piecewise_cartpole_y(100) + 0.7)
    ax.axhline(_piecewise_cartpole_y(5), color="#777777", linewidth=0.8, alpha=0.75)

    y_break = _piecewise_cartpole_y(5)
    y0, y1 = ax.get_ylim()
    y_frac = (y_break - y0) / (y1 - y0)
    kwargs = dict(transform=ax.transAxes, color="#555555", clip_on=False, linewidth=0.9)
    ax.plot((-0.012, 0.012), (y_frac - 0.012, y_frac + 0.012), **kwargs)
    ax.plot((0.988, 1.012), (y_frac - 0.012, y_frac + 0.012), **kwargs)


def _draw_bars(
    ax,
    best_dict: dict,
    title: str,
    ylim: float = 115,
    piecewise_cartpole_axis: bool = False,
    include_safe_region: bool = False,
) -> None:
    shields = list(best_dict.keys())
    fails   = [best_dict[s]["fail"]  * 100 for s in shields]
    stucks  = [best_dict[s]["stuck"] * 100 for s in shields]
    safes   = [max(0.0, 100.0 - fails[i] - stucks[i]) for i in range(len(shields))]
    colors  = [COLORS.get(s, "#888") for s in shields]
    x, w    = np.arange(len(shields)), 0.52
    ymap = _piecewise_cartpole_y if piecewise_cartpole_axis else (lambda y: y)

    fail_heights = [ymap(v) for v in fails]
    totals = [fails[i] + stucks[i] for i in range(len(shields))]
    stuck_heights = [ymap(totals[i]) - fail_heights[i] for i in range(len(shields))]
    safe_heights = [ymap(totals[i] + safes[i]) - ymap(totals[i]) for i in range(len(shields))]

    ax.bar(x, fail_heights,  w, color=colors, alpha=0.90, zorder=3)
    ax.bar(x, stuck_heights, w, bottom=fail_heights, color=colors, alpha=0.32,
           hatch="///", edgecolor="white", zorder=3)
    if include_safe_region:
        for i, s in enumerate(shields):
            ax.bar(
                x[i],
                safe_heights[i],
                w,
                bottom=ymap(totals[i]),
                facecolor="white",
                edgecolor=colors[i],
                linewidth=1.3,
                zorder=4,
            )

    for i, s in enumerate(shields):
        t_val = best_dict[s].get("threshold")
        lbl   = f"t={t_val:.2f}" if t_val is not None else "—"
        label_offset = (_piecewise_cartpole_y(100) + 0.7) * 0.015 if piecewise_cartpole_axis else ylim * 0.015
        # Keep threshold labels separate from the safe percentage labels, which
        # are centered inside the outlined safe segment.
        if include_safe_region:
            label_y = ymap(100.0) + label_offset
            label_va = "bottom"
        else:
            label_y = ymap(totals[i]) + label_offset
            label_va = "bottom"
        ax.text(x[i], label_y, lbl,
                ha="center", va=label_va, fontsize=7.5, zorder=6,
                color="#455A64")
        if fails[i] >= 6:
            ax.text(x[i], ymap(fails[i] / 2), f"{fails[i]:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=5)
        if stucks[i] >= 6:
            ax.text(
                x[i],
                ymap(fails[i]) + stuck_heights[i] / 2,
                f"{stucks[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                fontweight="bold",
                zorder=5,
            )
        if include_safe_region and safes[i] >= 6:
            ax.text(
                x[i],
                ymap(totals[i]) + safe_heights[i] / 2,
                f"{safes[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#263238",
                fontweight="bold",
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SHIELD_LABELS.get(s, s) for s in shields], fontsize=9)
    ax.set_ylabel("Rate (%)")
    if piecewise_cartpole_axis:
        _format_piecewise_cartpole_axis(ax)
    else:
        ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def make_bar_figure(cs_name: str, cs_cfg: dict) -> str:
    sweep = load_json(cs_cfg["sweep"])
    obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
    carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
    fs_path = cs_cfg.get("fs")
    fs    = load_json(fs_path) if fs_path and fs_path.exists() else None

    # Compute a shared y-limit scaled to the actual data range
    all_totals = []
    for perc in ["uniform", "adversarial_opt"]:
        best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"], fs_data=fs)
        for d in best.values():
            all_totals.append((d["fail"] + d["stuck"]) * 100)
    max_total = max(all_totals) if all_totals else 10
    ylim = max(max_total * 1.18, 12)   # 18% headroom for labels; min 12

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharey=True)
    fig.suptitle(f"Best Operating Point — {cs_cfg['long']}", fontsize=10)

    for ax, perc, plbl in zip(axes,
                               ["uniform", "adversarial_opt"],
                               ["Uniform perception", "Adversarial perception"]):
        best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"], fs_data=fs)
        _draw_bars(
            ax,
            best,
            plbl,
            ylim=ylim,
            piecewise_cartpole_axis=(cs_name == "cartpole_lowacc"),
            include_safe_region=True,
        )

    fp = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    sp = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///",
                        edgecolor="white", label="Stuck %")
    # Place legend where bars are shortest to avoid overlap
    legend_loc = "upper left" if max_total > 80 else "upper right"
    axes[1].legend(handles=[fp, sp], fontsize=8, loc=legend_loc)

    plt.tight_layout()
    out = OUTDIR / f"barchart_{VERSION}_{cs_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── combined overview (2 rows × 4 cols) ───────────────────────────────────────

_SUMMARY_TITLE = {
    "fail": "Best Operating Point — All Case Studies",
    "safe": "Highest-Safe Operating Point — All Case Studies",
}
_SUMMARY_SUFFIX = {"fail": "bars", "safe": "safe_bars"}


def make_summary_figure(objective: str = "fail") -> str:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(_SUMMARY_TITLE[objective], fontsize=12, y=1.01)

    # With the safe region drawn, every bar reaches 100% so use a shared y-axis.
    col_ylim = 115

    for col, (cs_name, cs_cfg) in enumerate(CASES.items()):
        sweep = load_json(cs_cfg["sweep"])
        obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
        fs_path = cs_cfg.get("fs")
        fs    = load_json(fs_path) if fs_path and fs_path.exists() else None

        for row, (perc, plbl) in enumerate(
            zip(["uniform", "adversarial_opt"], ["Uniform", "Adversarial"])
        ):
            ax = axes[row][col]
            best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"],
                                 fs_data=fs, objective=objective)
            _draw_bars(
                ax,
                best,
                f"{cs_cfg['label']}\n({plbl})",
                ylim=col_ylim,
                piecewise_cartpole_axis=(cs_name == "cartpole_lowacc"),
                include_safe_region=True,
            )
            if col > 0:
                ax.set_ylabel("")

    fp = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    sp = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///",
                        edgecolor="white", label="Stuck %")
    safe_p = mpatches.Patch(facecolor="white", edgecolor="#455A64",
                            linewidth=1.2, label="Safe %")
    fig.legend(handles=[fp, sp, safe_p], fontsize=9,
               loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    out = OUTDIR / f"summary_{VERSION}_{_SUMMARY_SUFFIX[objective]}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── markdown helpers ───────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _best_row(sweep_data, obs_data, carr_data, cs_cfg, perception, fs_data=None) -> dict:
    """Return {shield: (t, fail, stuck)} for the markdown table."""
    row = {}
    for s in cs_cfg["sweep_shields"]:
        bp = best_point(sweep_data, perception, s)
        if bp:
            row[s] = bp
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            f, s, t = min(pts, key=lambda p: (p[0], p[1]))
            row["observation"] = (t, f, s)
    cr = carr_result(carr_data, perception)
    if cr:
        row["carr"] = (None, cr[0], cr[1])
    if fs_data:
        pts = sweep_points(fs_data, perception, "forward_sampling")
        if pts:
            f, s, t = min(pts, key=lambda p: (p[0], p[1]))
            row["forward_sampling"] = (t, f, s)
    return row


def generate_markdown(figures: dict) -> str:
    lines: list[str] = []

    lines += [
        "# Evaluation Summary — v5",
        "",
        "**Shields compared**: Envelope, Single-Belief, Observation, Carr, Fwd-Sampling  ",
        "*(where feasible — see per-case notes)*",
        "",
        "**Case studies** (most-recent variants only):",
        "TaxiNet (16 states, 16 obs) · Obstacle (50 states, 3 obs) · "
        "CartPole low-acc (82 states, P_mid=0.373) · Refuel v2 (344 states, 29 obs)",
        "",
        "**Trials**: 200 per combination. Bar charts show the best operating",
        "threshold (min fail%, then min stuck%) for each shield.",
        "Pareto plots shown only where data varies meaningfully on both axes.",
        "",
        "**Forward-Sampling shield**: Uses `ForwardSampledBelief` (budget=500 points,",
        "K_samples=100) as an inner approximation of the reachable belief set, then",
        "applies the same probability-threshold mechanism as the Envelope shield.",
        "Feasible for all four case studies. With budget=500, K=100, the shield",
        "covers all 2n+2 structured observation-likelihood extremals for models with",
        "n≤249 states (TaxiNet, Obstacle, CartPole) and 100/690 extremals for Refuel.",
        "",
        "---",
        "",
        "## Cross-Case Summary",
        "",
        f"![Overview bar charts]({figures['summary']})",
        "",
    ]

    # cross-case best-points table
    lines += [
        "### Best operating points",
        "",
        "| Case study | Shield | t | Fail% (unif) | Stuck% (unif) | Fail% (adv) | Stuck% (adv) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cs_name, cs_cfg in CASES.items():
        sweep   = load_json(cs_cfg["sweep"])
        obs     = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr    = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
        fs_path = cs_cfg.get("fs")
        fs      = load_json(fs_path) if fs_path and fs_path.exists() else None
        ru    = _best_row(sweep, obs, carr, cs_cfg, "uniform", fs_data=fs)
        ra    = _best_row(sweep, obs, carr, cs_cfg, "adversarial_opt", fs_data=fs)
        all_s = [s for s in BAR_ORDER if s in ru or s in ra]
        for i, s in enumerate(all_s):
            cs_col = cs_cfg["long"] if i == 0 else ""
            t_u  = f"{ru[s][0]:.2f}" if s in ru and ru[s][0] else "—"
            fu   = _pct(ru[s][1]) if s in ru else "N/A"
            su   = _pct(ru[s][2]) if s in ru else "N/A"
            fa   = _pct(ra[s][1]) if s in ra else "N/A"
            sa   = _pct(ra[s][2]) if s in ra else "N/A"
            lines.append(f"| {cs_col} | {SHIELD_LABELS[s]} | {t_u} | {fu} | {su} | {fa} | {sa} |")
    lines.append("")

    lines += [
        "### Key cross-case findings",
        "",
        "1. **Envelope dominates Single-Belief** (TaxiNet, Obstacle) — lower fail at every"
        " threshold, at comparable or slightly higher stuck cost. Infeasible for CartPole"
        " lowacc and Refuel v2.",
        "",
        "2. **Observation shield is bimodal**: near-optimal for CartPole (2.5% fail / 0%"
        " stuck — matches Single-Belief) and offers a unique 0%-stuck operating region for"
        " Refuel v2 (3% fail / 0% stuck at t=0.65, vs Single-Belief's minimum 38% stuck)."
        " Degrades badly for TaxiNet (>55% fail at 0% stuck) and collapses to"
        " Carr-equivalent behaviour for Obstacle (1.5% fail / 98.5% stuck).",
        "",
        "3. **Carr achieves the lowest raw fail rate** on TaxiNet (7.5%) and Obstacle"
        " (1.5%), but always at near-total stuck cost (92–99%). Competitive only for"
        " CartPole (1.5% fail / 5.5% stuck) where 82 near-unique observations make"
        " Carr non-conservative. Infeasible for Refuel v2.",
        "",
        "4. **Observation informativeness governs shield effectiveness**: CartPole"
        " (82 obs ≈ 82 states) → all memoryless methods near-optimal; Obstacle"
        " (3 obs / 50 states) → both Observation and Carr degenerate; TaxiNet"
        " (16 obs / 16 states, poor posterior) → history essential; Refuel v2"
        " (29 obs / 344 states) → intermediate, with memoryless advantage at low t.",
        "",
        "5. **CartPole lowacc is the only case with zero liveness cost** across all"
        " threshold-based shields — both Single-Belief and Observation achieve ≤2.5% fail"
        " / 0% stuck at their best threshold.",
        "",
        "6. **Forward-Sampling (budget=500, K=100) achieves near-envelope safety** on"
        " TaxiNet (37.5% fail / 28% stuck vs Envelope 35% / 34% — nearly identical,"
        " slightly better liveness) and Obstacle (8% fail / 83.5% stuck vs Envelope 3% / 85%)."
        " With sufficient budget it is **always more conservative than Single-Belief**"
        " (lower fail, higher stuck), as the inner-approximation theory predicts: taking"
        " the min over N≥1 belief points is never less conservative than the single-point"
        " midpoint posterior. On CartPole lowacc it remains near-optimal (≤0.5% fail /"
        " 0% stuck). On Refuel v2 it matches Single-Belief for uniform perception"
        " (0% fail / 78% stuck vs SB 0% / 79%) but is more conservative adversarially"
        " (0% fail / 90.5% stuck vs SB 0% / 80%).",
        "",
        "---",
        "",
    ]

    # ── per-case sections ──────────────────────────────────────────────────────
    for cs_name, cs_cfg in CASES.items():
        sweep   = load_json(cs_cfg["sweep"])
        obs     = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr    = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
        fs_path = cs_cfg.get("fs")
        fs      = load_json(fs_path) if fs_path and fs_path.exists() else None

        lines += [f"## {cs_cfg['long']}", ""]

        if cs_cfg["pareto"]:
            lines += [
                f"![Pareto scatter]({figures['pareto'][cs_name]})",
                "",
                "> *Each point is one threshold setting. Only t=0.90 and t=0.95 are"
                " labelled. No lines connect points.*",
                "",
            ]

        lines += [
            f"![Bar chart — best threshold per shield]({figures['bar'][cs_name]})",
            "",
        ]

        # per-case table
        lines += [
            "### Best operating points",
            "",
            "| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |",
            "|---|---|---|---|---|---|---|",
        ]
        ru = _best_row(sweep, obs, carr, cs_cfg, "uniform", fs_data=fs)
        ra = _best_row(sweep, obs, carr, cs_cfg, "adversarial_opt", fs_data=fs)
        for s in BAR_ORDER:
            if s not in ru and s not in ra:
                continue
            tu  = f"{ru[s][0]:.2f}" if s in ru and ru[s][0] else "—"
            fu  = _pct(ru[s][1]) if s in ru else "N/A"
            su  = _pct(ru[s][2]) if s in ru else "N/A"
            ta  = f"{ra[s][0]:.2f}" if s in ra and ra[s][0] else "—"
            fa  = _pct(ra[s][1]) if s in ra else "N/A"
            sa  = _pct(ra[s][2]) if s in ra else "N/A"
            lines.append(
                f"| {SHIELD_LABELS[s]} | {tu} | {fu} | {su} | {ta} | {fa} | {sa} |"
            )
        lines.append("")

        # per-case findings + interpretation
        if cs_name == "taxinet":
            lines += [
                "### Key findings",
                "",
                "- **Envelope** offers the best safety-liveness trade-off: 35% fail / 34%"
                " stuck (uniform) at t=0.95.",
                "- **Single-Belief** is the most liveness-friendly: 44% fail / 11% stuck —"
                " useful when being stuck matters more than the remaining fail rate.",
                "- **Observation** achieves lower fail (12.5%) only at t=0.95, carrying"
                " 87.5% stuck — a poor trade given Envelope's 35% fail / 34% stuck at the"
                " same threshold.",
                "- **Carr** reaches the lowest raw fail (7.5%) but blocks ≥92% of episodes"
                " from step 0. The midpoint POMDP has 0 winning supports, so Carr is"
                " degenerate here.",
                "- **Fwd-Sampling** at t=0.95: 37.5% fail / 28% stuck (uniform) —"
                " **near-envelope quality** (cf. Envelope 35% / 34% stuck, Single-Belief"
                " 44% / 11% stuck). With budget=500, K=100, the shield now consistently"
                " beats Single-Belief on safety (7 pp lower fail) at the cost of 17 pp"
                " more stuck. It essentially matches Envelope safety with slightly better"
                " liveness, and does so without any LP computation.",
                "",
                "### Structural interpretation",
                "",
                "TaxiNet's perception accuracy (P_mid ≈ 0.354 across 16 states) is barely"
                " above random. The 16-obs / 16-state structure looks bijective but the"
                " emission noise means any single observation is consistent with"
                " multiple conflicting true states. A single obs carries almost no reliable"
                " information about which lane the agent occupies.",
                "",
                "This is why **history is essential**: each observation individually is"
                " nearly uninformative, but a sequence of observations and actions"
                " progressively eliminates impossible states and concentrates the belief."
                " Single-Belief and Envelope exploit this accumulation; the memoryless"
                " Observation shield cannot.",
                "",
                "**Carr's degeneracy** (0 winning supports) reveals a structural property:"
                " under the midpoint POMDP, there is no support set reachable from safe"
                " initial states from which safety can be guaranteed regardless of"
                " adversarial transitions. The IPOMDP probability-based shields relax the"
                " requirement from certainty to high-probability safety, which is what"
                " allows them to act at all.",
                "",
                "**Envelope outperforms Single-Belief** because midpoint transition"
                " probabilities systematically underestimate the risk of unsafe transitions"
                " in a noisy environment. The envelope's worst-case analysis corrects this"
                " optimism at the cost of slightly higher stuck.",
                "",
                "**Forward-Sampling (budget=500, K=100) approaches Envelope quality**."
                " With budget=500 the coordinate-extremal pruner keeps all 2n=32 min/max"
                " extremal points for TaxiNet and fills the remaining 468 slots with"
                " diverse random candidates. K=100 likelihood samples cover all 34"
                " structured (2n+2) hybrid extremals plus 66 random interior points."
                " The result is an inner-approximation belief set that closely tracks the"
                " LP envelope's reachable belief polytope — achieving 37.5% fail vs"
                " Envelope's 35% fail, a gap of only 2.5 pp at the same threshold.",
                "",
                "The irreducible ~34% fail at t=0.95 (Envelope) represents the fundamental"
                " difficulty ceiling for this perception regime: even with full belief"
                " history and robust worst-case shielding, near-random position sensing"
                " cannot prevent failure in all episodes.",
                "",
                "---",
                "",
            ]
        elif cs_name == "obstacle":
            lines += [
                "### Key findings",
                "",
                "- **Envelope** Pareto-dominates Single-Belief at every threshold;"
                " 3% fail / 85% stuck at t=0.95 is the best achievable safety-liveness"
                " point for any threshold-based shield.",
                "- **Observation** and **Carr** converge to the same extreme corner:"
                " ~1.5% fail / 98.5% stuck — nearly indistinguishable.",
                "- **Single-Belief** at t=0.95 (14% fail / 50% stuck) offers the best"
                " liveness of any low-fail operating point.",
                "- **Fwd-Sampling** at t=0.95: 8% fail / 83.5% stuck (uniform),"
                " 9.5% fail / 82.5% stuck (adversarial) — significantly better than"
                " Single-Belief on safety (8% vs ~14% fail uniform) and approaching"
                " Envelope (3% fail / 85% stuck). The gap between FS and Envelope"
                " (5 pp on fail at near-identical stuck) represents the fundamental limit"
                " of inner approximation: sampling cannot guarantee the exact worst-case"
                " belief that the LP finds.",
                "",
                "### Structural interpretation",
                "",
                "Three observations for 50 states (~17 states per observation) is the most"
                " extreme observation compression in this benchmark. Every observation group"
                " contains a heterogeneous mix of states — some near the obstacle, some not"
                " — with conflicting safety requirements. This creates **irreducible stuck"
                " even at t=0.50**: the shield cannot find an action safe for all states"
                " consistent with the current observation.",
                "",
                "**Why Observation ≡ Carr**: with 17 diverse states per observation, the"
                " probability-based posterior P(s|obs) is nearly uniform. The observation"
                " shield's calculation converges to Carr's worst-case support analysis —"
                " both ask 'is action a safe for all (or almost all) states consistent"
                " with this observation?' and with a heterogeneous group the answer is"
                " almost always no, leading to the same ~98.5% stuck at high thresholds.",
                "",
                "**Belief tracking breaks the deadlock**: after several steps, the"
                " trajectory of observations and actions constrains which of the ~17 states"
                " are actually reachable — distinguishing, for example, states that were"
                " reached by moving east from those reached by moving west. The belief"
                " effectively narrows the support far below 17, allowing confident action"
                " recommendations. This is why Single-Belief, Envelope, and Forward-Sampling"
                " substantially outperform memoryless methods at mid-range thresholds.",
                "",
                "**Envelope's dominance at every threshold** (not just high thresholds as"
                " in TaxiNet) reflects the width of the Obstacle IPOMDP intervals: with"
                " 3 observations for 50 states, per-state transition probabilities are"
                " poorly determined and the intervals [P_lower, P_upper] are wide. The"
                " midpoint POMDP used by Single-Belief consistently underestimates risk,"
                " while the Envelope's worst-case analysis remains accurate.",
                "",
                "**Forward-Sampling (budget=500, K=100) now tracks the Envelope closely**:"
                " at t=0.95 it achieves 8% fail / 83.5% stuck vs Envelope's 3% / 85% stuck."
                " The 500 diverse belief points cover all 102 structured (2n+2) hybrid"
                " extremals for Obstacle's 50-state model, plus 398 random interior points."
                " This near-complete extremal coverage explains the major improvement over"
                " the previous budget=100, K=10 run (14.5% fail): the additional belief"
                " points correctly identify dangerous scenarios near the obstacle that the"
                " small-budget version missed. The residual 5 pp gap to Envelope reflects"
                " irreducible sampling uncertainty.",
                "",
                "---",
                "",
            ]
        elif cs_name == "cartpole_lowacc":
            lines += [
                "### Key findings",
                "",
                "- All shields achieve **0% stuck** at their best threshold —"
                " CartPole lowacc is the only case with zero liveness cost.",
                "- **Fwd-Sampling** achieves ≤0.5% fail / 0% stuck across nearly all"
                " thresholds (uniform) — matching or marginally beating Single-Belief"
                " (1% fail / 0% stuck at t=0.85). The near-bijective structure makes all"
                " N=500 sampled belief points converge quickly to near-singleton"
                " distributions, so the increased budget adds no conservatism overhead.",
                "- **Single-Belief** is marginally next: 1% fail / 0% stuck at t=0.85.",
                "- **Observation** matches Single-Belief's liveness (0% stuck) at 2.5%"
                " fail. The near-bijective 82-obs/82-state structure means a single"
                " observation is nearly as informative as the full belief.",
                "- **Carr** is competitive (1.5% fail / 0% stuck uniform); the lowacc"
                " support-MDP has only 2 reachable supports (1 winning), so Carr adds"
                " essentially no conservatism.",
                "- Envelope not available (LP infeasible at ~1.9 s/step for 200-trial sweep).",
                "",
                "### Structural interpretation",
                "",
                "CartPole is the easiest shielding problem in this suite due to three"
                " compounding structural properties:",
                "",
                "1. **Near-bijective observations (82 obs / 82 states)**: each observation"
                "   is associated with one canonical state, so even with 37% per-state"
                "   accuracy the posterior concentrates strongly on the correct state."
                "   History adds almost nothing — the first observation is already highly"
                "   informative. This is why Observation shield matches Single-Belief.",
                "",
                "2. **Only 2 actions**: stuck requires the shield to block *left* AND"
                "   *right* simultaneously. This can only happen if the posterior places"
                "   significant probability on the FAIL state, which rarely occurs during"
                "   normal RL trajectories that stay well within the safe region.",
                "",
                "3. **Gradual, controllable dynamics**: from any non-FAIL state, at least"
                "   one action reduces the pole angle. The physics create no dead-end"
                "   belief states where every action leads to failure.",
                "",
                "**Forward-Sampling's near-zero cost** on CartPole reflects the near-bijective"
                " structure: with 82 near-unique observations, all N=500 belief points"
                " collapse to near-singleton distributions after the first observation."
                " The K=100 likelihood samples all converge to the same posterior, so"
                " budget and K have no practical effect here. This also explains why the"
                " budget increase from 100→500 and K from 10→100 made no material"
                " difference on CartPole — the model is simple enough that any positive"
                " budget is sufficient.",
                "",
                "**The counterintuitive stuck comparison**: CartPole standard (P_mid=0.532)"
                " shows 68% stuck at t=0.95 for the Observation shield, while lowacc"
                " (P_mid=0.373) shows 0% stuck at the same threshold. Higher accuracy"
                " concentrates the posterior more strongly on individual states. If that"
                " concentration falls on a near-FAIL state, P(FAIL|obs) can exceed 5%"
                " and trigger blocking at t=0.95. Noisier observations spread the"
                " posterior more diffusely, keeping P(FAIL|obs) below the threshold —"
                " lower perception accuracy accidentally prevents over-conservatism.",
                "",
                "**Carr's trivial support structure** (2 supports, 1 winning) confirms"
                " the near-bijective property: once the agent takes one step from the"
                " safe initial support, the support collapses to a near-singleton"
                " immediately. Carr is essentially operating on the true state.",
                "",
                "---",
                "",
            ]
        elif cs_name == "refuel_v2":
            lines += [
                "### Key findings",
                "",
                "- **Single-Belief** achieves 0% fail at t=0.90 (79% stuck uniform;"
                " 80% stuck adversarial) — the lowest stuck rate among 0%-fail operating"
                " points under uniform perception.",
                "- **Observation** also achieves 0% fail but with 99% stuck at t=0.90."
                "  Its unique advantage: at t=0.65, it gives **3–4.5% fail / 0% stuck**"
                " — the only 0%-stuck operating point for Refuel v2."
                " Single-Belief has ≥38% stuck at every threshold.",
                "- **Fwd-Sampling** at t=0.50 uniform: **0% fail / 78% stuck** —"
                " matching Single-Belief's best (0% / 79%) within 1 pp. With budget=500,"
                " K=100, the shield becomes more conservative than Single-Belief (as"
                " theory predicts for a well-covered inner approximation),"
                " so it no longer offers the liveness advantage seen with budget=100."
                " Adversarial: 0% fail / 90.5% stuck at t=0.80 — more conservative"
                " than Single-Belief (0% / 80%), reflecting full-coverage sampling"
                " finding more dangerous belief states.",
                "- Carr and Envelope are both infeasible (support-MDP BFS and LP exceed"
                " memory / time budgets at 344 states × 29 obs).",
                "",
                "### Structural interpretation",
                "",
                "Refuel v2 is the only benchmark where safety predicates are genuinely"
                " **hidden from the observation**: fuel level and obstacle proximity are"
                " not encoded in any observation bit. The agent must infer danger entirely"
                " from indirect signals (relative position, time elapsed), testing whether"
                " IPOMDP shielding provides real value under genuine partial observability.",
                "",
                "**Single-Belief's liveness trap** arises because accurate belief tracking"
                " works against liveness. As the episode progresses, the belief correctly"
                " concentrates on states where fuel is critically low or the obstacle is"
                " adjacent. At t≥0.70, the shield rightly identifies that all actions have"
                " significant probability of leading to failure — but the agent is now"
                " paralysed in a belief corner with no safe exit. This is a true"
                " safety-liveness tension: accurate danger awareness leads to paralysis.",
                "",
                "**The Observation shield's 0%-stuck advantage at low t** is a consequence"
                " of its memorylessness. Without accumulating history, the posterior over"
                " 344 states via 29 observations (~12 states per obs) is too uncertain to"
                " classify all actions as dangerous simultaneously. The shield 'doesn't know"
                " enough to be paralysed.' The cost is 3–4.5% fail, but this is the only"
                " operating point with zero liveness cost in the entire benchmark.",
                "",
                "**Forward-Sampling's behaviour on Refuel v2 validates the theory**."
                " With budget=100, K=10 (only ~1.4% of Refuel's 690 structured"
                " likelihood extremals sampled), the shield was artificially under-conservative:"
                " it happened to miss dangerous belief corners, giving 0% fail / 70% stuck"
                " adversarially — better liveness than SB but for the wrong reason."
                " With budget=500, K=100 (~14.5% extremal coverage), the shield correctly"
                " becomes more conservative than SB (0% fail / 90.5% stuck adversarially)."
                " For Refuel v2 where n=344 and full coverage would require budget≥688, K≥690,"
                " the shield remains a scalable but conservative approximation.",
                "",
                "**Scalability boundary**: Carr (support-MDP BFS over 344 states × 29 obs)"
                " and Envelope (LP at ~144 s/step) both exceed practical limits. Forward"
                " Sampling (~35 s/combo) and Single-Belief are the only scalable options,"
                " with Forward Sampling providing better adversarial liveness at the cost"
                " of slightly worse safety guarantees.",
                "",
                "---",
                "",
            ]

    # ── inference-time section ──────────────────────────────────────────────
    lines += [
        "## Inference Time",
        "",
        "Per-step wall-clock latency of `shield.next_actions()`. "
        "300 steps per shield (30 for Envelope), threshold=0.90, "
        "random-walk trajectories using the IPOMDP exact transition model. "
        "Fwd-Sampling timings reflect budget=500, K_samples=100.",
        "",
    ]
    if TIMING.exists():
        from .run_timing_benchmark import generate_markdown_table
        with open(TIMING) as _f:
            _timing_data = json.load(_f)
        lines.append(generate_markdown_table(_timing_data))
        lines += [
            "",
            "### Timing hierarchy",
            "",
            "- **No Shield / Observation / Single-Belief / Carr**: sub-millisecond "
            "(0.1–225 μs/step) — all suitable for real-time control loops.",
            "- **Fwd-Sampling (budget=500, K=100)**: 7–384 ms/step — 4–10× faster than "
            "Envelope where both are feasible; feasible for Refuel v2 where Envelope (∼144 s/step) "
            "is not. Suitable for low-to-moderate frequency decisions (≥2–10 Hz for most models).",
            "- **Envelope**: 83–646 ms/step (LP-based). Suitable only for offline or "
            "very-low-frequency shielding. Infeasible for CartPole and Refuel.",
            "- **Carr** runtime is near-zero after precomputation (table lookup): "
            "1–37 μs/step, competitive with Single-Belief, but the BFS precomputation "
            "is infeasible for large state spaces.",
            "",
        ]
    else:
        lines += [
            "_Timing data not available. Run "
            "`python -m ipomdp_shielding.experiments.run_timing_benchmark` "
            "to generate._",
            "",
        ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figures: dict = {"pareto": {}, "bar": {}}

    for cs_name, cs_cfg in CASES.items():
        print(f"\n── {cs_cfg['long']} ──")

        if cs_cfg["pareto"]:
            path = make_pareto_figure(cs_name, cs_cfg)
            figures["pareto"][cs_name] = path
            print(f"  Pareto → {path}")

        path = make_bar_figure(cs_name, cs_cfg)
        figures["bar"][cs_name] = path
        print(f"  Bar    → {path}")

    figures["summary"] = make_summary_figure("fail")
    print(f"\n  Summary (lowest-fail) → {figures['summary']}")
    figures["summary_safe"] = make_summary_figure("safe")
    print(f"  Summary (highest-safe) → {figures['summary_safe']}")

    rel = lambda p: Path(p).relative_to(OUTDIR)
    rel_figures = {
        "pareto": {k: rel(v) for k, v in figures["pareto"].items()},
        "bar": {k: rel(v) for k, v in figures["bar"].items()},
        "summary": rel(figures["summary"]),
        "summary_safe": rel(figures["summary_safe"]),
    }
    md = generate_markdown(rel_figures)
    MD_PATH.write_text(md)
    print(f"\n  Markdown → {MD_PATH}")


if __name__ == "__main__":
    main()
