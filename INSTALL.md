# INSTALL

The artifact runs via Docker. Nothing else needs to be installed — the image
pins the entire environment (base image `python:3.11-slim`, CPU-only).

## Build

From the artifact root (the directory containing this file):

```bash
docker build -t ipomdp-shielding .
```

## Minimal working example (~2 min)

```bash
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding
```

This is the default command (`./run_experiments.sh smoke_test`): it regenerates
every paper figure/table from the bundled data into the mounted `figures/`
directory and runs a tiny (2-trial) live shielding sweep. Expected final line:

```
[smoke] ALL CHECKS PASSED
```

## Full reproduction (~12 h)

```bash
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding \
    ./run_experiments.sh reproduce_results
```

Reruns every Monte Carlo experiment from the bundled trained agents (no
retraining), then regenerates every figure/table.

See `README.md` for the command-to-figure mapping and `REQUIREMENTS.md` for
environment details.
