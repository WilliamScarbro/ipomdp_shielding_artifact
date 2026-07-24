# STATUS

We are applying for the following EMSOFT artifact badges.

## Available
The artifact is archived in a public, persistent repository with a DOI (Zenodo)
and released under the MIT License (see `LICENSE`). It is self-contained: all
code, trained controllers, optimized perception realizations, and result data
needed to reproduce every figure and table in the paper are included.

## Reviewed (Functional)
The artifact runs and produces the outputs described in the paper:

- `./run_experiments.sh figures` regenerates every paper figure and supplement
  table from the bundled result JSONs, deterministically, in under a couple of
  minutes and with no GPU or network access.
- `./run_experiments.sh smoke` performs a fast end-to-end sanity check that
  additionally exercises the live Monte Carlo simulation / shield pipeline on a
  bundled trained controller.
- Every regenerated figure and printed table matches the paper. For example, the
  conformal-baseline comparison (Fig. 4 / Table 3) reproduces exactly, and the
  main summary/Pareto figures are byte-for-byte equivalent to the submission.

A mapping from each command to the specific paper figure/table it produces is
given in `README.md`.

## Reproducible
The underlying Monte Carlo experiments can be re-executed from scratch via the
`reproduce-*` subcommands, which reuse the bundled trained agents and optimized
adversarial realizations (no retraining). These regenerate the result JSONs that
the `figures` step then plots. Note the full run (`reproduce-all`) takes
**~12 hours** on a single core; individual `reproduce-*` runtimes are documented
in `README.md`, so reviewers can select a subset within their time budget.

## Notes for evaluators
- The artifact is CPU-only. See `REQUIREMENTS.md` for the (modest) hardware and
  software constraints.
- The recommended entry point is Docker (`INSTALL.md`), which pins the entire
  environment; a local Python 3.11 path is also supported.
