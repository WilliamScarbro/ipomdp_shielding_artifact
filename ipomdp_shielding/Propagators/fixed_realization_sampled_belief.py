"""Forward-sampled belief with fixed per-trajectory perception realizations.

Companion to ForwardSampledBelief. The varying sampler re-samples likelihood
vectors at every timestep (matching the IPOMDP semantics where nature picks
freely inside the bounds each step). This propagator instead samples K full
perception tables P(o|s) at trajectory start and propagates one belief per
realization for the entire trajectory.

Each row of self.points is a single belief evolving under one fixed,
self-consistent perception model. The K rows together form an
under-approximation of the reachable belief set under the "consistent
realization" semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..Models.ipomdp import IPOMDP
from .belief_base import IPOMDP_Belief


@dataclass
class FixedRealizationSampledBelief(IPOMDP_Belief):
    """Belief propagator with K fixed per-trajectory perception realizations.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model.
    num_realizations : int
        Number of parallel realizations K. Each holds one belief throughout
        a trajectory under one fixed perception model P(o|s).
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.
    """

    ipomdp: IPOMDP = field(repr=False)
    num_realizations: int = 100
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    points: np.ndarray = field(init=False, repr=False)
    _L: np.ndarray = field(init=False, repr=False)
    _obs_index: dict = field(init=False, repr=False)
    _parameterizer: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Imported lazily to avoid a circular dependency:
        # Propagators -> MonteCarlo -> Evaluation -> MonteCarlo.
        from ..MonteCarlo.fixed_realization_model import IntervalRealizationParameterizer

        if self.rng is None:
            self.rng = np.random.default_rng()
        self._parameterizer = IntervalRealizationParameterizer(self.ipomdp)
        self._obs_index = {o: j for j, o in enumerate(self.ipomdp.observations)}
        self._reset_beliefs()
        self._resample_realizations()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_beliefs(self):
        n = len(self.ipomdp.states)
        uniform = np.ones(n) / n
        self.points = np.tile(uniform, (self.num_realizations, 1))

    def _resample_realizations(self):
        """Sample K full perception tables P(o|s) and cache as (K, n_states, n_obs)."""
        K = self.num_realizations
        n_states = len(self.ipomdp.states)
        n_obs = len(self.ipomdp.observations)
        n_params_states, n_params_obs = self._parameterizer.param_shape

        alphas = self.rng.uniform(0.0, 1.0, size=(K, n_params_states, n_params_obs))

        L = np.zeros((K, n_states, n_obs), dtype=float)
        for r in range(K):
            realization = self._parameterizer.params_to_realization(alphas[r])
            for i, s in enumerate(self.ipomdp.states):
                row = realization.get(s, {})
                for o, p in row.items():
                    j = self._obs_index.get(o)
                    if j is not None:
                        L[r, i, j] = float(p)
        self._L = L

    def restart(self):
        """Reset beliefs to uniform and resample K fresh perception realizations."""
        self._reset_beliefs()
        self._resample_realizations()

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, action, obs) -> bool:
        """Propagate K beliefs through (action, obs) using each row's fixed P(o|s).

        1. Predict: points @ T_a
        2. Look up obs column for every realization: W[r, j] = L[r, j, obs]
        3. Posterior: predicted * W (elementwise per realization), normalised
        """
        T_mat = self.ipomdp._build_T_matrix(action)
        predicted = self.points @ T_mat  # (K, n_states)

        obs_idx = self._obs_index.get(obs)
        if obs_idx is None:
            # Unknown observation — keep predicted prior unchanged.
            self.points = predicted
            return True

        W = self._L[:, :, obs_idx]  # (K, n_states)
        unnorm = predicted * W
        sums = unnorm.sum(axis=1, keepdims=True)

        out = predicted.copy()
        valid = sums.ravel() > 1e-15
        if np.any(valid):
            out[valid] = unnorm[valid] / sums[valid]
        self.points = out
        return True

    # ------------------------------------------------------------------
    # Set-containment queries — same semantics as ForwardSampledBelief
    # ------------------------------------------------------------------

    def minimum_allowed_probability(self, allowed: Iterable) -> float:
        allowed = list(allowed)
        if not allowed:
            return 0.0
        idx = np.asarray(allowed)
        probs = self.points[:, idx].sum(axis=1)
        return float(np.min(probs))

    def maximum_disallowed_probability(self, disallowed: Iterable) -> float:
        disallowed = list(disallowed)
        if not disallowed:
            return 0.0
        idx = np.asarray(disallowed)
        probs = self.points[:, idx].sum(axis=1)
        return float(np.max(probs))
