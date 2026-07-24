# STATUS

We are applying for the following EMSOFT artifact badges.

## Available
The artifact is archived in a public, persistent repository with a DOI (Zenodo,
**[10.5281/zenodo.21539082](https://doi.org/10.5281/zenodo.21539082)**) and
released under the MIT License (see `LICENSE`). It is self-contained: all
code, trained controllers, optimized perception realizations, and result data
needed to reproduce every figure and table in the paper are included.

## Reviewed (Functional)
The artifact runs and produces the outputs described in the paper:

- `./run_experiments.sh smoke_test` (the default Docker command) regenerates
  every paper figure and supplement table from the bundled result data,
  deterministically, in ~2 minutes with no GPU or network access, and
  additionally runs a tiny live Monte Carlo shielding sweep to exercise the
  simulation pipeline.
- Every regenerated figure and printed table matches the paper. For example, the
  conformal-baseline comparison (Fig. 4 / Table 3) reproduces exactly, and the
  main summary/Pareto figures are byte-for-byte equivalent to the submission.

A mapping from each output to the specific paper figure/table is given in
`README.md`.

## Reproducible
`./run_experiments.sh reproduce_results` re-executes every underlying Monte Carlo
experiment from scratch, reusing the bundled trained agents and optimized
adversarial realizations (no retraining), then regenerates every figure/table
from the freshly computed results. Note the full run takes **~12 hours** on a
single core.

## Notes for evaluators
- The artifact is CPU-only. See `REQUIREMENTS.md` for the (modest) hardware and
  software constraints.
- The recommended entry point is Docker (`INSTALL.md`), which pins the entire
  environment; a local Python 3.11 path is also supported.
