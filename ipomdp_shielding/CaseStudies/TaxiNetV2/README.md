# TaxiNetV2 implementation notes

## Benchmark spec

`TaxiNetV2` preserves the current TaxiNet control task and only swaps the
perception source.

- Shared with the existing TaxiNet case study:
  - same signed state space: `cte in {-2,-1,0,1,2}`, `he in {-1,0,1}`, plus `FAIL`
  - same action space: `{-1,0,1}`
  - same stochastic taxiing dynamics and same perfect-perception dynamic shield
  - same safety objective and rollout-style evaluation contract
- Replaced by vendored conformal artifacts:
  - observation data comes from vendored conformal prediction outputs
  - train/cal/test split metadata comes from vendored `train/models/*.pt`
  - the observation alphabet is the set of conformal prediction-set pairs that
    appear in the vendored TaxiNetV2 artifact

## Data products

The loader consumes the following vendored files directly:

- `compiler/lib/acc90/real_cte_pred_acc90_conf{95,99,995}.csv`
- `compiler/lib/acc90/real_he_pred_acc90_conf{95,99,995}.csv`
- `train/models/{train,cal,val,test}_indices.pt`

The resulting normalized products inside this module are:

- empirical `(true_state, conformal_observation)` samples for interval learning
- split metadata exposed through `get_scarbro_split_indices(...)`
- compatibility-only projected concrete test observations derived from the
  conformal sets for legacy callers that expect per-state sample lists

## Current limitation

The repository does not yet vendor point-estimate classifier outputs.
`TaxiNetV2` therefore uses conformal-set observations as the primary
observation model, and only derives projected concrete test samples as a
compatibility adapter for older helper code.
