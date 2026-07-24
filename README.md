# IPOMDP Shielding — Paper Artifact

Reproducible artifact for the paper *Interval POMDP Shielding for
Imperfect-Perception Agents*. It bundles the experiment code, the trained
controllers, and the result data needed to reproduce every figure and table in
the paper, together with a Docker recipe for a self-contained environment.

Everything runs through **two commands**:

| Command | What it does | Time |
|---|---|---|
| `smoke_test` | Regenerate every paper figure/table from the bundled result data, then run a tiny live shielding sweep to exercise the simulation pipeline. | ~2 min |
| `reproduce_results` | Rerun every Monte Carlo experiment from scratch (using the bundled trained agents — no retraining), then regenerate every figure/table. | ~12 h |

The artifact is **CPU-only** and needs no network access at run time. Start with
`smoke_test`; run `reproduce_results` only if you want to recompute the underlying
numbers.

## Build

```bash
docker build -t ipomdp-shielding .
```

## Run

Both commands write the paper figures into `figures/` inside the container; mount
a host directory there to collect them.

```bash
# Fast check + regenerate all figures/tables (this is also the default command):
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding

# Equivalent explicit form:
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding \
    ./run_experiments.sh smoke_test

# Full reproduction from scratch (~12 h):
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding \
    ./run_experiments.sh reproduce_results
```

After either command, `figures/` on the host holds the regenerated plots and the
supplement tables are printed to stdout.

## What gets produced

Both commands produce the same set of outputs (`reproduce_results` first
recomputes the underlying result JSONs; `smoke_test` plots the bundled ones).

| Paper item | Output file | Source data |
|---|---|---|
| Fig. 1 — main summary, lowest-failure | `figures/summary_v7_bars.png` | `results/sweep_v7/{threshold,obs,fs,carr}/*.json` |
| Fig. 2 — main summary, highest-safe | `figures/summary_v7_safe_bars.png` | same |
| Fig. 3 — TaxiNet / Obstacle Pareto | `figures/pareto_v7_taxinet.png`, `figures/pareto_v7_obstacle.png` | `results/sweep_v7/*` |
| per-case bar charts | `figures/barchart_v7_*.png` | same |
| Fig. 4 — conformal baseline comparison | `figures/taxinet_v2_extremal_comparison.png` | `results/taxinet_v2/*.json` |
| Shield inference timing | `figures/inference_timing_summary.pdf` | `results/timing_benchmark/shield_timing.json` |
| Envelope coarseness | `figures/coarse_taxinet_results.png` | `results/final/coarse_taxinet_results.json` |
| Perception variability | `figures/perception_variability_taxinet.png` | `results/final/perception_variability/*.json` |
| Alpha sensitivity sweep | `figures/alpha_sweep_taxinet_{fail,stuck,safe}.png` | `data/sweep/rl_alpha_taxinet_v2/sweep_summary.json` |
| Tables 1 & 2 (main summary) | printed to stdout | `results/sweep_v7/*` |
| Table 3 (conformal comparison) | printed to stdout | `results/taxinet_v2/*` |
| Two-state LP worked example (supplement) | printed to stdout | self-contained |

The five shields evaluated are **Envelope** (LP over-approximation),
**Single-Belief** (point-estimate history), **Observation** (memoryless),
**Carr** (support-based, Carr et al.), and **Fwd-Sampling** (sampled
under-approximation), on the four benchmarks **TaxiNet**, **Obstacle**,
**CartPole**, and **Refuel**.

## Artifact evaluation documents

* `INSTALL.md` — build/run and a minimal working example.
* `REQUIREMENTS.md` — hardware/software requirements and constraints.
* `STATUS.md` — badges requested (Available, Reviewed, Reproducible) and justification.
* `LICENSE` — MIT (required for the *Available* badge).

## Layout

```
ipomdp_shielding/           # experiment package
  Models/                   # (I)POMDP / (I)MDP models, confidence intervals
  Propagators/              # belief propagation (LFP envelope, exact HMM, sampling)
  Evaluation/               # runtime shields + metrics
  MonteCarlo/               # closed-loop simulation, DQN controller, CEM realization optimizer
  CaseStudies/              # TaxiNet, TaxiNetV2, Obstacle/Refuel gridworlds, CartPole
  experiments/              # sweep runners, plotters, and configs/
scripts/                    # figure regenerators + smoke test
results/                    # bundled paper result JSONs + trained-agent/realization caches
data/sweep/                 # bundled alpha-sweep data
figures/                    # regenerated figures land here
```
