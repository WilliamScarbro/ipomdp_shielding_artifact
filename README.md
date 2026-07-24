# IPOMDP Shielding — Paper Artifact

Reproducible artifact for the paper *Interval-POMDP Shielding for Agents with
Imperfect Perception*. It contains the experiment code, the bundled trained
controllers and result data needed to reproduce every figure and table, and a
Docker recipe for a self-contained environment.

The artifact supports two modes:

* **`figures`** — regenerate every paper figure/table from the *bundled* result
  JSONs. Fast (seconds), fully deterministic, and requires no GPU. This is the
  recommended way to verify the paper's plots and numbers.
* **`reproduce-*`** — rerun the underlying Monte Carlo experiments from scratch
  using the bundled trained agents and optimized perception realizations (no
  retraining needed). These are long-running; see the runtimes below.

## Artifact evaluation documents

For the EMSOFT Artifact Evaluation Committee:

* `INSTALL.md` — install + minimal working example.
* `REQUIREMENTS.md` — hardware/software requirements and constraints.
* `STATUS.md` — badges requested (Available, Reviewed, Reproducible) and justification.
* `LICENSE` — MIT (open source; required for the *Available* badge).

## Quick start (Docker)

```bash
docker build -t ipomdp-shielding .
# Regenerate all paper figures into the host's ./figures (default CMD):
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .

./run_experiments.sh figures     # regenerate all figures + tables -> ./figures/
./run_experiments.sh smoke       # fast end-to-end sanity check
```

Python 3.11 is recommended. The reproduction path depends only on
`numpy, scipy, statsmodels, matplotlib, torch` (CPU build, pinned in
`requirements.txt`).

## What `figures` produces

`./run_experiments.sh figures` (i.e. `scripts/plot_paper_figures.py`) writes the
following into `figures/` and prints the supplement tables to stdout:

| Paper item | Output file | Bundled source data |
|---|---|---|
| Fig. 1 — main summary, lowest-failure | `summary_v7_bars.png` | `results/sweep_v7/{threshold,obs,fs,carr}/*.json` |
| Fig. 2 — main summary, highest-safe | `summary_v7_safe_bars.png` | same |
| Fig. 3 — TaxiNet / Obstacle Pareto | `pareto_v7_taxinet.png`, `pareto_v7_obstacle.png` | `results/sweep_v7/*` |
| per-case bars | `barchart_v7_*.png` | same |
| Fig. 4 — conformal baseline comparison | `taxinet_v2_extremal_comparison.png` | `results/taxinet_v2/*.json`, `.../operating_pareto_sweep/results.json` |
| Shield inference timing | `inference_timing_summary.pdf` | `results/timing_benchmark/shield_timing.json` |
| Envelope coarseness | `coarse_taxinet_results.png` | `results/final/coarse_taxinet_results.json` |
| Perception variability | `perception_variability_taxinet.png` | `results/final/perception_variability/*.json` |
| Alpha sensitivity sweep | `alpha_sweep_taxinet_{fail,stuck,safe}.png` | `data/sweep/rl_alpha_taxinet_v2/sweep_summary.json` |
| Tables 1 & 2 (main summary data) | `results/sweep_v7/evaluation_summary.md` (CIs in the sweep JSONs) | `results/sweep_v7/*` |
| Table 3 (conformal comparison) | printed to stdout by the Fig. 4 step | `results/taxinet_v2/*` |
| Two-state LP worked example (supplement) | printed to stdout | self-contained |

The five shields evaluated are **Envelope** (LP over-approximation), **Single-Belief**
(point-estimate history), **Observation** (memoryless), **Carr** (support-based,
Carr et al.), and **Fwd-Sampling** (sampled under-approximation), on the four
benchmarks **TaxiNet**, **Obstacle**, **CartPole**, and **Refuel**.

## Reproducing experiments from scratch

Each command reruns the Monte Carlo experiments and overwrites the corresponding
JSONs under `results/`; rerun `figures` afterward to replot. All commands reuse
the bundled trained agents (`results/cache/*_agent.pt`) and optimized adversarial
realizations (`results/cache/*_opt_realization*.json`), so nothing is retrained.

| Command | Reproduces | Approx. runtime |
|---|---|---|
| `./run_experiments.sh reproduce-main` | Fig. 1/2/3, Tables 1/2 (4 benchmarks × 5 shields × 2 regimes, 200 rollouts) | ~8 h |
| `./run_experiments.sh reproduce-timing` | timing figure | ~10 min |
| `./run_experiments.sh reproduce-coarse` | coarseness figure | ~20 min |
| `./run_experiments.sh reproduce-perception` | perception-variability figure | ~30 min |
| `./run_experiments.sh reproduce-alpha` | alpha-sweep figures | ~2 h |
| `./run_experiments.sh reproduce-conformal` | Fig. 4, Table 3 | ~1 h |
| `./run_experiments.sh reproduce-all` | everything above, then `figures` | ~12 h |

## Layout

```
ipomdp_shielding/           # experiment package
  Models/                   # (I)POMDP / (I)MDP models, confidence intervals
  Propagators/              # belief propagation (LFP envelope, exact HMM, sampling)
  Evaluation/               # runtime shields (envelope, single-belief, observation,
                            #   Carr support-based, conformal), metrics
  MonteCarlo/               # closed-loop simulation, DQN controller, adversarial
                            #   realization optimizer (Cross-Entropy Method)
  CaseStudies/              # TaxiNet, TaxiNetV2, Obstacle/Refuel gridworlds, CartPole
  experiments/              # sweep runners, plotters, and configs/
scripts/                    # figure regenerators + smoke test
results/                    # bundled paper result JSONs + trained-agent/realization caches
data/sweep/                 # bundled alpha-sweep data
figures/                    # regenerated figures land here
```

## Retraining controllers (optional)

The controllers and perception artifacts are bundled, so retraining is not needed
to reproduce the paper. To rebuild them from scratch, install the extra deps and
use the training scripts under `CaseStudies/`:

```bash
pip install -r requirements-train.txt
python -m ipomdp_shielding.CaseStudies.CartPole.train_lowacc   # example
```
