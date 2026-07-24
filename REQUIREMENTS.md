# REQUIREMENTS

Hardware and software needed to evaluate this artifact.

## Hardware
- **CPU:** any x86-64 machine (the artifact is CPU-only; no GPU required).
- **Memory:** 8 GB RAM is sufficient. The figure-regeneration path uses well under 2 GB.
- **Disk:** ~1 GB free (artifact is ~28 MB; the remainder is the Docker image /
  Python environment). No network access is required at run time — all trained
  controllers, optimized perception realizations, and result JSONs are bundled.

## Technical skills required to review
Only basic command-line and Docker proficiency: `docker build` and `docker run`
with a volume mount. No GPU, cluster, licensed software, or special hardware.
Interpreting the figures/tables benefits from — but does not require —
familiarity with POMDPs and runtime shielding; `README.md` maps every command to
its paper figure/table so outputs can be checked against the paper directly.

## Software
- **Docker Engine** (any recent version). Nothing else is required — the image
  pins the entire environment. Base image: `python:3.11-slim`.
- For reference, the reproduction path inside the image depends only on the
  packages pinned in `requirements.txt`: `numpy 1.26.4`, `scipy 1.13.1`,
  `statsmodels 0.14.2`, `matplotlib 3.9.2`, `torch 2.4.1` (CPU build). Optional
  controller retraining (not needed — controllers are bundled) uses
  `requirements-train.txt`.

## Time budget
- `smoke_test`: **~2 minutes**.
- `reproduce_results` (full experiment reproduction): **~12 hours** on a single core.

## Constraints that could cause evaluation to fail
- Do **not** run under Python < 3.9.
- Install `torch` from the CPU wheel index as pinned; the default CUDA wheel is
  unnecessary and much larger.
- Use the bundled `results/cache/*` — deleting it forces retraining, which is out
  of scope for reproduction.
