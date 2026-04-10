# IPOMDP Shielding Artifact

This repository is a clean paper artifact reconstructed from the experiment codebase. It contains the single bundled experiment, the cached models needed to rerun it, the bundled result set in `results/experiment/`, and a smoke test that exercises the artifact end to end.

## Layout

- `ipomdp_shielding/`: experiment package and case-study code
- `results/cache/`: required trained-agent and optimized-realization caches
- `results/experiment/`: bundled experiment outputs
- `evaluation_summary.md`: generated paper-facing summary
- `scripts/smoke_test.py`: minimal end-to-end verification

## Installation

```bash
pip install -r requirements.txt
pip install --no-deps -e .
```

## Commands

```bash
./run_experiments.sh smoke
./run_experiments.sh plot
./run_experiments.sh reproduce
```

`smoke` runs a tiny real threshold sweep for `cartpole_lowacc` in a temporary copy of `results/`, regenerates the summary, and checks that the expected outputs are non-empty.

`plot` rebuilds `evaluation_summary.md` and the artifact figures from the bundled JSON results.

`reproduce` reruns the experiment bundle and then regenerates the summary.
