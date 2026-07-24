"""Regenerate the TaxiNetV2 conformal extremal-comparison figure + supplement table.

Reproduces figures/taxinet_v2_extremal_comparison.png (paper Fig. 4) and prints the
data underlying supplement Table 3, from the bundled JSONs:
    results/taxinet_v2/taxinet_v2_comparison_results.json          (kappa=0.95)
    results/taxinet_v2/taxinet_v2_comparison_conf99_results.json   (kappa=0.99)
    results/taxinet_v2/taxinet_v2_comparison_conf995_results.json  (kappa=0.995)
    results/taxinet_v2/operating_pareto_sweep/results.json         (beta sweep)

The interval-belief shields select their per-method beta operating point that
minimizes failures (top row) or maximizes safe completions (bottom row), with the
same tiebreakers as the main summary figures. Conformal sweeps kappa in {0.95,0.99,0.995}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFORMAL_RESULT_DIR = ROOT / "results" / "taxinet_v2"
POINT_SWEEP_PATH = ROOT / "results" / "taxinet_v2" / "operating_pareto_sweep" / "results.json"
FIG_DIR = ROOT / "figures"
OUT_PATH = FIG_DIR / "taxinet_v2_extremal_comparison.png"

CONFORMAL_RESULT_FILES = {
    "0.95": "taxinet_v2_comparison_results.json",
    "0.99": "taxinet_v2_comparison_conf99_results.json",
    "0.995": "taxinet_v2_comparison_conf995_results.json",
}

POINT_METHOD_KEYS = ("envelope", "forward_sampling", "single_belief")
METHOD_ORDER = ("envelope", "forward_sampling", "single_belief", "conformal_prediction")
METHOD_LABELS = {
    "envelope": "Envelope",
    "forward_sampling": "Fwd-Sampling",
    "single_belief": "Single-Belief",
    "conformal_prediction": "Conformal",
}
METHOD_COLORS = {
    "envelope": "#e07b3a",
    "forward_sampling": "#2aa3a3",
    "single_belief": "#3a73b8",
    "conformal_prediction": "#8b5fbf",
}


@dataclass(frozen=True)
class Row:
    method: str
    setting_kind: str
    setting_value: str
    fail_rate: float
    stuck_rate: float
    safe_rate: float
    num_trials: int


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta as beta_dist

    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lower, upper)


def cp_ci(rate: float, n: int) -> tuple[float, float]:
    return _clopper_pearson(int(round(rate * n)), n)


def load_conformal_rows() -> dict[str, list[Row]]:
    rows: dict[str, list[Row]] = {"uniform": [], "adversarial_opt": []}
    for confidence, filename in CONFORMAL_RESULT_FILES.items():
        with (CONFORMAL_RESULT_DIR / filename).open() as handle:
            data = json.load(handle)["results"]
        for perception in rows:
            metrics = data[f"{perception}/rl/conformal_prediction"]
            rows[perception].append(
                Row("conformal_prediction", "kappa", confidence,
                    metrics["fail_rate"], metrics["stuck_rate"], metrics["safe_rate"],
                    metrics["num_trials"])
            )
    return rows


def load_point_rows() -> dict[str, dict[str, list[Row]]]:
    with POINT_SWEEP_PATH.open() as handle:
        data = json.load(handle)
    out: dict[str, dict[str, list[Row]]] = {"uniform": {}, "adversarial_opt": {}}
    for perception in out:
        point_sweep = data["point_sweep"][perception]
        for method in POINT_METHOD_KEYS:
            out[perception][method] = [
                Row(method, "beta", beta_str,
                    metrics["fail_rate"], metrics["stuck_rate"], metrics["safe_rate"],
                    metrics["num_trials"])
                for beta_str, metrics in point_sweep[method].items()
            ]
    return out


def _selection_key(objective: str) -> Callable[[Row], tuple[float, float, float]]:
    if objective == "lowest_fail":
        return lambda row: (row.fail_rate, row.stuck_rate, -row.safe_rate)
    if objective == "highest_safe":
        return lambda row: (-row.safe_rate, row.fail_rate, row.stuck_rate)
    raise ValueError(f"unknown objective: {objective}")


def select_rows(point_rows, conformal_rows, objective) -> dict[str, Row]:
    key = _selection_key(objective)
    selected = {m: min(point_rows[m], key=key) for m in POINT_METHOD_KEYS}
    selected["conformal_prediction"] = min(conformal_rows, key=key)
    return selected


def setting_label(row: Row) -> str:
    val = float(row.setting_value)
    if abs(val * 100 - round(val * 100)) < 1e-9 and (val * 100) % 10 == 0:
        text = f"{val:.1f}"
    else:
        text = f"{val:g}"
    return (rf"$\kappa$={text}" if row.setting_kind == "kappa" else rf"$\beta$={text}")


def pct(value: float) -> str:
    scaled = value * 100
    return f"{scaled:.0f}%" if abs(scaled - round(scaled)) < 1e-9 else f"{scaled:.1f}%"


def draw_panel(ax, selected: dict[str, Row], title: str) -> None:
    x = np.arange(len(METHOD_ORDER))
    width = 0.58
    fails = np.array([selected[m].fail_rate * 100 for m in METHOD_ORDER])
    stucks = np.array([selected[m].stuck_rate * 100 for m in METHOD_ORDER])
    safes = np.array([selected[m].safe_rate * 100 for m in METHOD_ORDER])
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    ax.bar(x, fails, width, color=colors, alpha=0.92, zorder=3)
    ax.bar(x, stucks, width, bottom=fails, color=colors, alpha=0.34,
           hatch="///", edgecolor="white", zorder=3)
    ax.bar(x, safes, width, bottom=fails + stucks, color="white",
           edgecolor=colors, linewidth=1.4, zorder=3)

    for idx, method in enumerate(METHOD_ORDER):
        row = selected[method]
        ax.text(idx, 102.0, setting_label(row), ha="center", va="bottom", fontsize=7.4)
        if fails[idx] >= 6:
            ax.text(idx, fails[idx] / 2, pct(row.fail_rate), ha="center", va="center",
                    fontsize=7.2, color="white", fontweight="bold")
        if stucks[idx] >= 6:
            ax.text(idx, fails[idx] + stucks[idx] / 2, pct(row.stuck_rate), ha="center",
                    va="center", fontsize=7.0, color="white", fontweight="bold")
        if safes[idx] >= 6:
            ax.text(idx, fails[idx] + stucks[idx] + safes[idx] / 2, pct(row.safe_rate),
                    ha="center", va="center", fontsize=7.0, color="#263238", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8.3)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Rate (%)", fontsize=8.8)
    ax.set_title(title, fontsize=9.3)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def _fmt_pct(rate: float) -> str:
    scaled = rate * 100.0
    return f"{scaled:.0f}" if abs(scaled - round(scaled)) < 1e-9 else f"{scaled:.1f}"


def _fmt_ci_int(lo: float, hi: float) -> str:
    return f"[{int(round(lo*100))},{int(round(hi*100))}]"


def emit_table_summary(point_rows, conformal_rows) -> None:
    print()
    print("=" * 78)
    print("Selected operating points (data underlying supplement Table 3 / Fig. 4)")
    print("=" * 78)
    for objective in ("lowest_fail", "highest_safe"):
        for perception in ("uniform", "adversarial_opt"):
            sel = select_rows(point_rows[perception], conformal_rows[perception], objective)
            print()
            print(f"-- {objective} / {perception} --")
            for method in METHOD_ORDER:
                r = sel[method]
                n = r.num_trials
                fl, fh = cp_ci(r.fail_rate, n)
                sl, sh = cp_ci(r.stuck_rate, n)
                ssl, ssh = cp_ci(r.safe_rate, n)
                print(f"  {METHOD_LABELS[method]:<14} {r.setting_kind}={r.setting_value} "
                      f"n={n}  fail={_fmt_pct(r.fail_rate)} {_fmt_ci_int(fl, fh)}  "
                      f"stuck={_fmt_pct(r.stuck_rate)} {_fmt_ci_int(sl, sh)}  "
                      f"safe={_fmt_pct(r.safe_rate)} {_fmt_ci_int(ssl, ssh)}")


def main() -> None:
    conformal_rows = load_conformal_rows()
    point_rows = load_point_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.1), sharey=True)
    panels = [
        ("lowest_fail", "uniform", "Lowest Failure / Uniform"),
        ("lowest_fail", "adversarial_opt", "Lowest Failure / Adversarial"),
        ("highest_safe", "uniform", "Highest Safe / Uniform"),
        ("highest_safe", "adversarial_opt", "Highest Safe / Adversarial"),
    ]
    for ax, (objective, perception, title) in zip(axes.flat, panels):
        selected = select_rows(point_rows[perception], conformal_rows[perception], objective)
        draw_panel(ax, selected, title)

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.92, label="Fail %")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.34, hatch="///",
                                 edgecolor="white", label="Stuck %")
    safe_patch = mpatches.Patch(facecolor="white", edgecolor="#455A64",
                                linewidth=1.2, label="Safe %")
    fig.legend(handles=[fail_patch, stuck_patch, safe_patch], fontsize=8,
               loc="lower center", ncol=3, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")

    emit_table_summary(point_rows, conformal_rows)


if __name__ == "__main__":
    main()
