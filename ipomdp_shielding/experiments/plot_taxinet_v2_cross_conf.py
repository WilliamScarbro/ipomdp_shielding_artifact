"""Cross-confidence comparison plot for TaxiNetV2.

Loads results from all three confidence levels (0.95, 0.99, 0.995) and
generates:
  1. Grouped bar chart: fail rate and stuck rate per shield per conf level.
  2. Pareto scatter: fail vs. stuck for all shields/confs.
  3. Grounding plot: CP MC fail rate vs. PRISM action_filter=0.9 crash bound across conf levels.

Usage:
    python -m ipomdp_shielding.experiments.plot_taxinet_v2_cross_conf
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONF_LEVELS = ["0.95", "0.99", "0.995"]
RESULT_FILES = {
    "0.95": "results/taxinet_v2/taxinet_v2_comparison_results.json",
    "0.99": "results/taxinet_v2/taxinet_v2_comparison_conf99_results.json",
    "0.995": "results/taxinet_v2/taxinet_v2_comparison_conf995_results.json",
}
SCARBRO_FILE = "results/taxinet_v2/scarbro_baseline_import.json"
OUTPUT_DIR = "results/taxinet_v2/cross_conf_figures"

SHIELDS = ["none", "conformal_prediction", "observation", "single_belief", "envelope", "forward_sampling"]
SHIELD_LABELS = {
    "none": "No Shield",
    "conformal_prediction": "Conf. Pred.",
    "observation": "Obs.",
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Fwd-Sampling",
}
SHIELD_COLORS = {
    "none": "#888888",
    "conformal_prediction": "#e63946",
    "observation": "#f4a261",
    "single_belief": "#4a90d9",
    "envelope": "#2a9d8f",
    "forward_sampling": "#00838F",
}
SHIELD_MARKERS = {
    "none": "X",
    "conformal_prediction": "^",
    "observation": "s",
    "single_belief": "D",
    "envelope": "o",
    "forward_sampling": "P",
}


def _action_filter_label(value: float | None) -> str:
    if value is None:
        return "action_filter=?"
    return f"action_filter={value:.1f}"


def load_all_results() -> Dict[str, Dict]:
    root = _project_root()
    results = {}
    for cl, rel in RESULT_FILES.items():
        path = root / rel
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping conf={cl}")
            continue
        with path.open() as fh:
            data = json.load(fh)
        results[cl] = data["results"]
    return results


def load_scarbro() -> List[Dict]:
    path = _project_root() / SCARBRO_FILE
    if not path.exists():
        return []
    with path.open() as fh:
        d = json.load(fh)
    pts = []
    for v in d["variants"]:
        m, s = v["metadata"], v["summary"]
        if not m["default_action"] or not s:
            continue
        crash = (s.get("crash") or {}).get("value")
        stuck = (s.get("stuck_or_default") or {}).get("value")
        if crash is None or stuck is None:
            continue
        pts.append({
            "conf": m["confidence_level"],
            "af": m["action_filter_tag"],
            "action_filter": m.get("action_filter"),
            "action_filter_label": m.get("action_filter_label", _action_filter_label(m.get("action_filter"))),
            "crash": crash,
            "stuck": stuck,
        })
    return pts


def get(results: Dict, conf: str, perc: str, sel: str, shield: str, metric: str):
    key = f"{perc}/{sel}/{shield}"
    r = results.get(conf, {}).get(key)
    return r[metric] if r else None


def make_grouped_bar(all_results, scarbro_pts, perc, selector, outdir):
    """Grouped bar: for each shield, show fail+stuck bars at each conf level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confs = [c for c in CONF_LEVELS if c in all_results]
    shields_to_plot = [s for s in SHIELDS if s != "none"]

    n_shields = len(shields_to_plot)
    n_confs = len(confs)
    bar_width = 0.35
    group_gap = 0.15
    group_width = n_confs * bar_width + group_gap

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    metrics = [("fail_rate", "Fail Rate"), ("stuck_rate", "Stuck Rate")]

    conf_colors = ["#2196F3", "#FF9800", "#9C27B0"]
    conf_hatches = ["", "//", "xx"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x_positions = np.arange(n_shields) * group_width
        for ci, conf in enumerate(confs):
            vals = []
            lo_errs = []
            hi_errs = []
            for sh in shields_to_plot:
                key = f"{perc}/{selector}/{sh}"
                r = all_results[conf].get(key)
                if r:
                    v = r[metric]
                    ci_lo = r.get(f"{metric}_ci_low", v)
                    ci_hi = r.get(f"{metric}_ci_high", v)
                    vals.append(v)
                    lo_errs.append(v - ci_lo)
                    hi_errs.append(ci_hi - v)
                else:
                    vals.append(0)
                    lo_errs.append(0)
                    hi_errs.append(0)

            positions = x_positions + ci * bar_width
            bars = ax.bar(positions, vals, bar_width * 0.9,
                          label=f"conf={conf}",
                          color=conf_colors[ci], alpha=0.85, hatch=conf_hatches[ci])
            ax.errorbar(positions, vals,
                        yerr=[lo_errs, hi_errs],
                        fmt="none", ecolor="black", capsize=2, alpha=0.6)

        # PRISM action_filter=0.9 overlay (horizontal lines per conf) — fail panel only
        if metric == "fail_rate":
            prism_by_conf = {}
            for pt in scarbro_pts:
                if pt["af"] == "af9":
                    prism_by_conf[pt["conf"]] = pt["crash"]
            for ci, conf in enumerate(confs):
                if conf in prism_by_conf:
                    # Draw a horizontal reference line at x-range of each group
                    x_start = x_positions[0] + ci * bar_width - bar_width * 0.5
                    x_end = x_positions[-1] + ci * bar_width + bar_width * 0.5
                    ax.hlines(prism_by_conf[conf], x_start, x_end,
                              colors=conf_colors[ci], linestyles="--",
                              linewidth=1.5, alpha=0.9,
                              label=f"PRISM action_filter=0.9 conf={conf}" if ci == 0 else "")

        ax.set_xticks(x_positions + (n_confs - 1) * bar_width / 2)
        ax.set_xticklabels([SHIELD_LABELS[s] for s in shields_to_plot], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    p_label = "Uniform" if perc == "uniform" else "Adversarial"
    fig.suptitle(f"TaxiNetV2 — {p_label} Perception  |  {selector.upper()} selector\n"
                 f"Fail & Stuck rates across conf levels (conf=0.95/0.99/0.995)",
                 fontsize=11)
    fig.tight_layout()
    fname = f"cross_conf_bars_{perc}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def make_grounding_plot(all_results, scarbro_pts, outdir):
    """CP shield MC vs. PRISM crash bound across confidence levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confs = [c for c in CONF_LEVELS if c in all_results]
    x = np.arange(len(confs))
    conf_labels = [f"conf={c}" for c in confs]

    # PRISM action-filter crash per conf
    prism_af9 = {}
    prism_af6 = {}
    for pt in scarbro_pts:
        if pt["af"] == "af9":
            prism_af9[pt["conf"]] = pt["crash"]
        if pt["af"] == "af6":
            prism_af6[pt["conf"]] = pt["crash"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, perc in zip(axes, ["uniform", "adversarial_opt"]):
        cp_fail = [get(all_results, c, perc, "rl", "conformal_prediction", "fail_rate")
                   for c in confs]
        cp_stuck = [get(all_results, c, perc, "rl", "conformal_prediction", "stuck_rate")
                    for c in confs]
        prism_crash_af9 = [prism_af9.get(c) for c in confs]
        prism_crash_af6 = [prism_af6.get(c) for c in confs]

        # MC lines
        ax.plot(x, cp_fail, "o-", color="#e63946", linewidth=2, markersize=7,
                label="CP shield: MC fail%")
        ax.plot(x, cp_stuck, "^--", color="#e63946", linewidth=1.5, markersize=6,
                alpha=0.7, label="CP shield: MC stuck%")

        # PRISM bound lines
        if any(v is not None for v in prism_crash_af9):
            ax.plot(x, prism_crash_af9, "s--", color="purple", linewidth=1.8,
                    markersize=6, label="PRISM action_filter=0.9 crash ≤ (formal bound)")
        if any(v is not None for v in prism_crash_af6):
            ax.plot(x, prism_crash_af6, "D:", color="mediumpurple", linewidth=1.2,
                    markersize=5, alpha=0.7, label="PRISM action_filter=0.6 crash ≤ (formal bound)")

        ax.set_xticks(x)
        ax.set_xticklabels(conf_labels, fontsize=9)
        ax.set_ylabel("Rate", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"Grounding: CP shield MC vs. PRISM bounds\n"
                     f"{'Uniform' if perc == 'uniform' else 'Adversarial'} perception | RL selector",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fname = "cross_conf_grounding.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def make_pareto_scatter(all_results, scarbro_pts, outdir):
    """Pareto scatter across all conf levels and shields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confs = [c for c in CONF_LEVELS if c in all_results]
    conf_alphas = {"0.95": 1.0, "0.99": 0.65, "0.995": 0.35}
    conf_sizes = {"0.95": 140, "0.99": 100, "0.995": 60}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, perc in zip(axes, ["uniform", "adversarial_opt"]):
        for conf in confs:
            for sh in SHIELDS:
                key = f"{perc}/rl/{sh}"
                r = all_results[conf].get(key)
                if not r:
                    continue
                ax.scatter(
                    r["stuck_rate"], r["fail_rate"],
                    s=conf_sizes[conf],
                    color=SHIELD_COLORS[sh],
                    marker=SHIELD_MARKERS[sh],
                    alpha=conf_alphas[conf],
                    zorder=5,
                    label=f"{SHIELD_LABELS[sh]} conf={conf}" if conf == "0.95" else None,
                )
                if conf == "0.95":
                    ax.annotate(f"  {conf}", (r["stuck_rate"], r["fail_rate"]),
                                fontsize=6, color=SHIELD_COLORS[sh])

        p_label = "Uniform" if perc == "uniform" else "Adversarial"
        ax.set_xlabel("Stuck Rate  (conservatism →)", fontsize=10)
        ax.set_ylabel("Fail/Crash Rate  (← safety)", fontsize=10)
        ax.set_title(f"TaxiNetV2 Pareto — {p_label} Perception\n"
                     f"All conf levels (opacity: 0.95 > 0.99 > 0.995)",
                     fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        if perc == "uniform":
            ax.legend(loc="upper right", fontsize=6, ncol=2)

    fig.tight_layout()
    fname = "cross_conf_pareto.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    root = _project_root()
    outdir = str(root / OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)

    print("Loading results...")
    all_results = load_all_results()
    scarbro_pts = load_scarbro()
    print(f"  Loaded {len(all_results)} conf levels, {len(scarbro_pts)} PRISM variants")

    print("Generating grounding plot...")
    make_grounding_plot(all_results, scarbro_pts, outdir)

    print("Generating grouped bar charts...")
    for perc in ["uniform", "adversarial_opt"]:
        make_grouped_bar(all_results, scarbro_pts, perc, "rl", outdir)

    print("Generating cross-conf Pareto scatter...")
    make_pareto_scatter(all_results, scarbro_pts, outdir)

    print(f"\nAll cross-conf figures written to: {outdir}")


if __name__ == "__main__":
    main()
