"""Regenerate the TaxiNet alpha-sweep figures from the bundled sweep summary.

Reproduces the supplement figures
    figures/alpha_sweep_taxinet_fail.png
    figures/alpha_sweep_taxinet_stuck.png
    figures/alpha_sweep_taxinet_safe.png
from data/sweep/rl_alpha_taxinet_v2/sweep_summary.json (bundled with the artifact,
regenerable via `run_experiments.sh reproduce-alpha`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "data" / "sweep" / "rl_alpha_taxinet_v2" / "sweep_summary.json"
DEFAULT_OUT_DIR = ROOT / "figures"

PERCEPTION_LABELS = {
    "uniform": "Uniform Random",
    "adversarial_opt": "Adversarial Optimized",
}

SHIELD_STYLES = {
    "single_belief": {"color": "steelblue", "marker": "o", "label": "Single-Belief"},
    "envelope": {"color": "seagreen", "marker": "s", "label": "Envelope"},
    "forward_sampling": {"color": "darkorange", "marker": "^", "label": "Fwd-Sampling"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_aggregate(path: Path) -> tuple[list[dict], dict]:
    with path.open() as handle:
        payload = json.load(handle)
    return payload["results"]["aggregated"], payload["metadata"]["config"]


def filter_rows(agg: list[dict], perception: str, shield: str, beta: float) -> list[dict]:
    return [
        row for row in agg
        if row["perception"] == perception
        and row["shield"] == shield
        and abs(row["beta"] - beta) < 1e-9
    ]


def plot_metric(agg, config, out_dir, metric_key, metric_label, suffix) -> Path:
    betas = sorted(config["betas"])
    perceptions = [p for p in config["perceptions"] if any(r["perception"] == p for r in agg)]
    shields = [s for s in config["shields"] if any(r["shield"] == s for r in agg)]

    fig, axes = plt.subplots(
        len(perceptions), len(betas),
        figsize=(5 * len(betas), 4 * len(perceptions)), squeeze=False,
    )
    for r, perception in enumerate(perceptions):
        for c, beta in enumerate(betas):
            ax = axes[r, c]
            for shield in shields:
                rows = filter_rows(agg, perception, shield, beta)
                if not rows:
                    continue
                rows.sort(key=lambda row: row["alpha"])
                xs = [row["alpha"] for row in rows]
                ys = [row[metric_key] for row in rows]
                lo_key = metric_key.replace("_mean", "_ci_low")
                hi_key = metric_key.replace("_mean", "_ci_high")
                yerr_lo = [max(y - row[lo_key], 0.0) for row, y in zip(rows, ys)]
                yerr_hi = [max(row[hi_key] - y, 0.0) for row, y in zip(rows, ys)]
                style = SHIELD_STYLES[shield]
                ax.errorbar(
                    xs, ys, yerr=[yerr_lo, yerr_hi],
                    marker=style["marker"], color=style["color"], label=style["label"],
                    linewidth=1.4, markersize=5, capsize=2, elinewidth=0.8, alpha=0.9,
                )
            ax.set_ylim(-0.05, 1.05)
            if r == len(perceptions) - 1:
                ax.set_xlabel(r"$\alpha$", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{PERCEPTION_LABELS.get(perception, perception)}\n{metric_label}", fontsize=9)
            if r == 0:
                ax.set_title(rf"$\beta={beta:g}$", fontsize=10)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{metric_label} vs " + r"$\alpha$" + f" -- {config['case_study_name'].upper()} "
        + r"($\beta$ values " + f"{betas})", fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alpha_sweep_taxinet_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    agg, config = load_aggregate(args.summary_json)
    for key, label, suffix in [
        ("fail_rate_mean", "Fail rate", "fail"),
        ("stuck_rate_mean", "Stuck rate", "stuck"),
        ("safe_rate_mean", "Safe rate", "safe"),
    ]:
        print(f"Wrote {plot_metric(agg, config, args.out_dir, key, label, suffix)}")


if __name__ == "__main__":
    main()
