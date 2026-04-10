"""Forward-sampled belief under-approximation for coarseness measurement.

Maintains a finite set of N concrete belief points as an inner
approximation of the true reachable belief set. Provides set-containment
probability queries that are O(N) (no LP required).

Mathematical basis:
    P_sampled (under) ⊆ P_true ⊆ P_lfp (over)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from ..Models.ipomdp import IPOMDP
from .belief_base import IPOMDP_Belief
from .utils import tightened_likelihood_bounds


class LikelihoodSamplingStrategy(Enum):
    """Strategy for sampling observation likelihood vectors."""
    EXTREME_POINTS = auto()
    UNIFORM_RANDOM = auto()
    HYBRID = auto()


class PruningStrategy(Enum):
    """Strategy for pruning candidate belief points back to budget."""
    COORDINATE_EXTREMAL = auto()
    FARTHEST_POINT = auto()
    RANDOM = auto()


@dataclass
class ForwardSampledBelief(IPOMDP_Belief):
    """Belief propagator using forward-sampled concrete belief points.

    Maintains a set of N belief points (numpy array of shape (N, n))
    as an inner approximation of the reachable belief set.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model.
    budget : int
        Number of belief points to maintain (N).
    K_samples : int
        Number of observation likelihood samples per propagation step.
    likelihood_strategy : LikelihoodSamplingStrategy
        How to sample observation likelihood vectors.
    pruning_strategy : PruningStrategy
        How to prune candidates back to budget.
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.
    """
    ipomdp: IPOMDP = field(repr=False)
    budget: int = 100
    K_samples: int = 10
    likelihood_strategy: LikelihoodSamplingStrategy = LikelihoodSamplingStrategy.HYBRID
    pruning_strategy: PruningStrategy = PruningStrategy.COORDINATE_EXTREMAL
    rng: Optional[np.random.Generator] = field(default=None, repr=False)
    points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        n = len(self.ipomdp.states)
        # Start with uniform prior as a single point, then fill budget
        uniform = np.ones(n) / n
        self.points = np.tile(uniform, (self.budget, 1))

    def restart(self):
        """Reset belief points to uniform prior."""
        n = len(self.ipomdp.states)
        uniform = np.ones(n) / n
        self.points = np.tile(uniform, (self.budget, 1))

    # ------------------------------------------------------------------
    # Tightened observation bounds (vectorized over states)
    # ------------------------------------------------------------------

    def _tightened_bounds_vectors(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tightened observation likelihood bounds for all states.

        Returns
        -------
        L_eff : np.ndarray of shape (n,)
            Tightened lower bounds.
        U_eff : np.ndarray of shape (n,)
            Tightened upper bounds.
        """
        n = len(self.ipomdp.states)
        L_eff = np.zeros(n)
        U_eff = np.zeros(n)
        for j, s in enumerate(self.ipomdp.states):
            lo, hi = tightened_likelihood_bounds(self.ipomdp, s, obs)
            L_eff[j] = lo
            U_eff[j] = hi
        return L_eff, U_eff

    # ------------------------------------------------------------------
    # Likelihood sampling strategies
    # ------------------------------------------------------------------

    def _sample_likelihoods(self, L_eff: np.ndarray, U_eff: np.ndarray) -> np.ndarray:
        """Sample K observation likelihood vectors w in [L_eff, U_eff].

        Returns
        -------
        W : np.ndarray of shape (K, n)
            Sampled likelihood vectors.
        """
        n = L_eff.shape[0]
        K = self.K_samples

        if self.likelihood_strategy == LikelihoodSamplingStrategy.EXTREME_POINTS:
            return self._sample_extreme_points(L_eff, U_eff, K, n)
        elif self.likelihood_strategy == LikelihoodSamplingStrategy.UNIFORM_RANDOM:
            return self._sample_uniform_random(L_eff, U_eff, K, n)
        else:  # HYBRID
            return self._sample_hybrid(L_eff, U_eff, K, n)

    def _sample_extreme_points(self, L_eff, U_eff, K, n):
        """Each w_j in {L_eff, U_eff}: enumerate if 2^n small, else random binary masks."""
        if n <= 15 and 2**n <= K:
            # Enumerate all 2^n extreme points
            num_corners = 2**n
            W = np.zeros((num_corners, n))
            for i in range(num_corners):
                for j in range(n):
                    if (i >> j) & 1:
                        W[i, j] = U_eff[j]
                    else:
                        W[i, j] = L_eff[j]
            # If we have room, add random fill
            if num_corners < K:
                extra = self._random_binary_masks(L_eff, U_eff, K - num_corners, n)
                W = np.vstack([W, extra])
            return W
        else:
            return self._random_binary_masks(L_eff, U_eff, K, n)

    def _random_binary_masks(self, L_eff, U_eff, K, n):
        """Random binary masks: each coordinate independently at L or U."""
        masks = self.rng.integers(0, 2, size=(K, n))
        W = np.where(masks, U_eff[np.newaxis, :], L_eff[np.newaxis, :])
        return W

    def _sample_uniform_random(self, L_eff, U_eff, K, n):
        """w_j ~ Uniform(L_eff, U_eff) independently per coordinate."""
        u = self.rng.uniform(size=(K, n))
        span = U_eff - L_eff
        W = L_eff[np.newaxis, :] + u * span[np.newaxis, :]
        return W

    def _sample_hybrid(self, L_eff, U_eff, K, n):
        """2n coordinate-extremal + all-low + all-high + random fill."""
        parts = []
        mid = 0.5 * (L_eff + U_eff)

        # 2n coordinate-extremal: push one coordinate low/high, rest at midpoint
        for j in range(n):
            # Push j low
            w_low = mid.copy()
            w_low[j] = L_eff[j]
            parts.append(w_low)
            # Push j high
            w_high = mid.copy()
            w_high[j] = U_eff[j]
            parts.append(w_high)

        # All-low and all-high corners
        parts.append(L_eff.copy())
        parts.append(U_eff.copy())

        W_structured = np.array(parts)
        n_structured = len(parts)

        if n_structured >= K:
            # More structured points than budget: subsample
            idx = self.rng.choice(n_structured, size=K, replace=False)
            return W_structured[idx]

        # Fill remainder with uniform random
        n_random = K - n_structured
        W_random = self._sample_uniform_random(L_eff, U_eff, n_random, n)
        return np.vstack([W_structured, W_random])

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------

    def _compute_posteriors(
        self, predicted: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        """Compute normalized posteriors from predicted beliefs and likelihoods.

        Parameters
        ----------
        predicted : np.ndarray of shape (N, n)
            Predicted belief points after transition.
        W : np.ndarray of shape (K, n)
            Sampled observation likelihood vectors.

        Returns
        -------
        posteriors : np.ndarray of shape (N*K, n)
            Normalized posterior belief points.
        """
        N = predicted.shape[0]
        K = W.shape[0]
        n = predicted.shape[1]

        # Outer product: each of N predicted beliefs x K likelihood vectors
        # predicted: (N,1,n), W: (1,K,n) -> unnorm: (N,K,n)
        unnorm = predicted[:, np.newaxis, :] * W[np.newaxis, :, :]
        unnorm = unnorm.reshape(N * K, n)

        # Normalize: b' = x / sum(x)
        sums = unnorm.sum(axis=1, keepdims=True)
        # Filter out points with zero or near-zero normalizing constant
        valid_mask = sums.ravel() > 1e-15
        if not valid_mask.any():
            # All zero — return current points unchanged
            return predicted.copy()

        posteriors = np.zeros_like(unnorm)
        posteriors[valid_mask] = unnorm[valid_mask] / sums[valid_mask]
        # Only keep valid posteriors
        return posteriors[valid_mask]

    # ------------------------------------------------------------------
    # Pruning strategies
    # ------------------------------------------------------------------

    def _prune(self, candidates: np.ndarray) -> np.ndarray:
        """Prune candidate set down to budget N.

        Parameters
        ----------
        candidates : np.ndarray of shape (M, n) where M >= budget

        Returns
        -------
        pruned : np.ndarray of shape (budget, n)
        """
        if candidates.shape[0] <= self.budget:
            # Pad with copies if needed
            if candidates.shape[0] < self.budget:
                deficit = self.budget - candidates.shape[0]
                idx = self.rng.choice(candidates.shape[0], size=deficit, replace=True)
                return np.vstack([candidates, candidates[idx]])
            return candidates

        if self.pruning_strategy == PruningStrategy.COORDINATE_EXTREMAL:
            return self._prune_coordinate_extremal(candidates)
        elif self.pruning_strategy == PruningStrategy.FARTHEST_POINT:
            return self._prune_farthest_point(candidates)
        else:  # RANDOM
            return self._prune_random(candidates)

    def _prune_coordinate_extremal(self, candidates: np.ndarray) -> np.ndarray:
        """Keep min/max per coordinate, random-fill remainder."""
        n = candidates.shape[1]
        selected_idx = set()

        # Min and max for each coordinate
        for j in range(n):
            selected_idx.add(int(np.argmin(candidates[:, j])))
            selected_idx.add(int(np.argmax(candidates[:, j])))

        selected = list(selected_idx)
        remaining = self.budget - len(selected)

        if remaining > 0:
            all_idx = np.arange(candidates.shape[0])
            mask = np.ones(candidates.shape[0], dtype=bool)
            mask[selected] = False
            pool = all_idx[mask]
            if len(pool) > 0:
                fill = self.rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                selected.extend(fill.tolist())

        # If still short, duplicate
        while len(selected) < self.budget:
            selected.append(selected[len(selected) % len(selected)])

        return candidates[selected[:self.budget]]

    def _prune_farthest_point(self, candidates: np.ndarray) -> np.ndarray:
        """Greedy farthest-point sampling for coverage."""
        M = candidates.shape[0]
        selected = [self.rng.integers(M)]
        min_dists = np.full(M, np.inf)

        for _ in range(self.budget - 1):
            last = candidates[selected[-1]]
            dists = np.linalg.norm(candidates - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            # Zero out already-selected so they aren't re-picked
            for s in selected:
                min_dists[s] = -1.0
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        return candidates[selected]

    def _prune_random(self, candidates: np.ndarray) -> np.ndarray:
        """Simple random subsample."""
        idx = self.rng.choice(candidates.shape[0], size=self.budget, replace=False)
        return candidates[idx]

    # ------------------------------------------------------------------
    # Main propagation
    # ------------------------------------------------------------------

    def propagate(self, action, obs) -> bool:
        """Propagate belief points through action and observation.

        1. Predict: points @ T_a (vectorized b_pred = T_a^T @ b)
        2. Sample K observation likelihood vectors
        3. Compute posteriors: x = w * y, normalize
        4. Prune N*K candidates back to budget N

        Parameters
        ----------
        action : Action
            The action taken.
        obs : Observation
            The observation received.

        Returns
        -------
        bool
            True (always succeeds for sampling-based approach).
        """
        # 1. Predict: b_pred = b @ T_a  (each row b is a belief, T[i,j]=P(s_j|s_i,a))
        T_mat = self.ipomdp._build_T_matrix(action)
        predicted = self.points @ T_mat  # (N, n)

        # 2. Sample observation likelihood vectors
        L_eff, U_eff = self._tightened_bounds_vectors(obs)
        W = self._sample_likelihoods(L_eff, U_eff)  # (K, n)

        # 3. Compute posteriors
        posteriors = self._compute_posteriors(predicted, W)  # (M, n), M <= N*K

        if posteriors.shape[0] == 0:
            # Degenerate case: no valid posteriors
            return True

        # 4. Prune back to budget
        self.points = self._prune(posteriors)
        return True

    # ------------------------------------------------------------------
    # Set-containment probability queries — O(N), no LP
    # ------------------------------------------------------------------

    def minimum_allowed_probability(self, allowed) -> float:
        """Minimum probability of being in allowed states across all points.

        Parameters
        ----------
        allowed : list of int
            State indices considered allowed/safe.

        Returns
        -------
        float
            min over points of sum(b[allowed])
        """
        if not allowed:
            return 0.0
        allowed_arr = np.array(allowed)
        probs = self.points[:, allowed_arr].sum(axis=1)
        return float(np.min(probs))

    def maximum_disallowed_probability(self, disallowed) -> float:
        """Maximum probability of being in disallowed states across all points.

        Parameters
        ----------
        disallowed : list of int
            State indices considered disallowed/unsafe.

        Returns
        -------
        float
            max over points of sum(b[disallowed])
        """
        if not disallowed:
            return 0.0
        disallowed_arr = np.array(disallowed)
        probs = self.points[:, disallowed_arr].sum(axis=1)
        return float(np.max(probs))
