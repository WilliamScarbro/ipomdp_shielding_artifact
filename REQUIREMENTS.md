# REQUIREMENTS

Hardware and software needed to evaluate this artifact.

## Hardware
- **CPU:** any x86-64 machine (the artifact is CPU-only; no GPU required).
- **Memory:** 8 GB RAM is sufficient. The figure-regeneration path uses well under 2 GB.
- **Disk:** ~1 GB free (artifact is ~28 MB; the remainder is the Docker image /
  Python environment). No network access is required at run time — all trained
  controllers, optimized perception realizations, and result JSONs are bundled.

## Software
Two supported paths; either is sufficient.

### Docker (recommended, self-contained)
- Docker Engine (any recent version). Nothing else — the image pins all
  dependencies. Base image: `python:3.11-slim`.

### Local Python
- **Python 3.11** (3.9+ works; 3.11 is what the artifact was vetted on).
- The reproduction path depends only on the packages pinned in
  `requirements.txt`: `numpy 1.26.4`, `scipy 1.13.1`, `statsmodels 0.14.2`,
  `matplotlib 3.9.2`, `torch 2.4.1` (CPU build).
- Optional retraining of controllers additionally needs `requirements-train.txt`
  (`gymnasium`, `torchvision`, `pillow`, `tqdm`). **Not needed** to reproduce any
  paper result — controllers are bundled.

## Time budget
- `figures` / `smoke`: **seconds to a couple of minutes**.
- Full experiment reproduction (`reproduce-all`): **~12 hours** on a single core.
  Per-experiment runtimes are listed in `README.md`.

## Constraints that could cause evaluation to fail
- Do **not** run under Python < 3.9.
- Install `torch` from the CPU wheel index as pinned; the default CUDA wheel is
  unnecessary and much larger.
- Use the bundled `results/cache/*` — deleting it forces retraining, which is out
  of scope for reproduction.
