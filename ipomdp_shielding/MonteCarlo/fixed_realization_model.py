"""Fixed realization perception model for IPOMDP.

This module implements a perception model that uses a single fixed realization
of the interval observation probabilities throughout all trials. Unlike the
AdversarialPerceptionModel (which adapts per timestep), this model commits to
a strategy upfront.

The fixed realization can be learned via optimization to maximize failure rate.
"""

import json
from typing import Any, Dict, Optional
import numpy as np

from .perception_models import PerceptionModel
from ..Models.ipomdp import IPOMDP


class FixedRealizationPerceptionModel(PerceptionModel):
    """Perception model using a fixed realization of interval probabilities.

    Stores a complete realization P(obs|state) for all (state, obs) pairs,
    where each probability is within the IPOMDP intervals. The realization
    is fixed across all trials and timesteps.

    Attributes
    ----------
    realization : dict
        Nested dict mapping state -> observation -> probability
        Must satisfy: P_lower[s][o] <= realization[s][o] <= P_upper[s][o]
        and sum_o realization[s][o] = 1 for each state s
    metadata : dict, optional
        Additional metadata (training config, objective score, etc.)
    """

    def __init__(
        self,
        realization: Dict[Any, Dict[Any, float]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize fixed realization perception model.

        Parameters
        ----------
        realization : dict
            Nested dict: state -> observation -> probability
        metadata : dict, optional
            Additional information about this realization
        """
        self.realization = realization
        self.metadata = metadata or {}
        self._validate_realization()

    def _validate_realization(self):
        """Validate that realization is a valid probability distribution.

        Checks:
        1. All probabilities are non-negative
        2. Probabilities sum to 1 for each state (simplex constraint)

        Note: Interval constraint checking requires access to IPOMDP,
        so it's done separately in IntervalRealizationParameterizer.
        """
        for state, obs_probs in self.realization.items():
            if state == "FAIL":
                continue

            # Check non-negative
            for obs, prob in obs_probs.items():
                if prob < 0:
                    raise ValueError(
                        f"Negative probability {prob} for state={state}, obs={obs}"
                    )

            # Check simplex constraint
            total = sum(obs_probs.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Probabilities for state={state} sum to {total}, expected 1.0"
                )

    def sample_observation(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sample observation from fixed distribution for given state.

        Parameters
        ----------
        state : any
            The true state
        ipomdp : IPOMDP
            The interval POMDP model (not used, kept for interface)
        context : dict, optional
            Additional context (not used)

        Returns
        -------
        observation
            Sampled observation from fixed distribution
        """
        if state == "FAIL":
            return "FAIL"

        dist = self.realization[state]
        observations = list(dist.keys())
        probs = [dist[o] for o in observations]

        import random
        return random.choices(observations, weights=probs, k=1)[0]

    def sample_distribution(
        self,
        state: Any,
        ipomdp: IPOMDP,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[Any, float]:
        """Return the fixed distribution P(obs|state).

        Parameters
        ----------
        state : any
            The true state
        ipomdp : IPOMDP
            The interval POMDP model (not used)
        context : dict, optional
            Additional context (not used)

        Returns
        -------
        dict
            Fixed distribution over observations
        """
        if state == "FAIL":
            return {"FAIL": 1.0}

        return self.realization[state].copy()

    def save(self, filepath: str):
        """Save realization to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save JSON file
        """
        # Convert to serializable format using repr() for keys
        data = {
            "realization": {
                repr(state): {repr(obs): float(prob) for obs, prob in obs_probs.items()}
                for state, obs_probs in self.realization.items()
            },
            "metadata": self.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "FixedRealizationPerceptionModel":
        """Load realization from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file

        Returns
        -------
        FixedRealizationPerceptionModel
            Loaded perception model
        """
        import ast

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct realization with proper key types using ast.literal_eval
        raw_realization = data["realization"]
        realization = {}

        for state_key, obs_probs in raw_realization.items():
            try:
                state = ast.literal_eval(state_key)
            except:
                state = state_key  # Keep as string if not parseable

            realization[state] = {}

            for obs_key, prob in obs_probs.items():
                try:
                    obs = ast.literal_eval(obs_key)
                except:
                    obs = obs_key  # Keep as string if not parseable

                realization[state][obs] = prob

        metadata = data.get("metadata", {})

        return cls(realization=realization, metadata=metadata)


class IntervalRealizationParameterizer:
    """Converts between alpha parameters and interval realizations.

    Uses linear interpolation to parameterize realizations:
        P(obs|state) = (1-alpha) * P_lower + alpha * P_upper

    Then normalizes to ensure probabilities sum to 1 per state.

    This parameterization:
    - Ensures all probabilities stay within intervals automatically
    - Uses alpha in [0,1], which is easy to optimize
    - Reduces parameter space (one alpha per (state, obs) pair)

    Attributes
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    states : list
        Non-FAIL states
    observations : list
        All observations
    param_shape : tuple
        Shape of alpha parameter array (num_states, num_observations)
    """

    def __init__(self, ipomdp: IPOMDP):
        """Initialize parameterizer.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        """
        self.ipomdp = ipomdp

        # Exclude FAIL state from parameterization
        self.states = [s for s in ipomdp.states if s != "FAIL"]
        self.observations = list(ipomdp.observations)

        self._param_shape = (len(self.states), len(self.observations))

    @property
    def param_shape(self) -> tuple:
        """Shape of alpha parameter array.

        Returns
        -------
        tuple
            (num_states, num_observations)
        """
        return self._param_shape

    def params_to_realization(
        self,
        alphas: np.ndarray
    ) -> Dict[Any, Dict[Any, float]]:
        """Convert alpha parameters to realization dict.

        For each (state, obs) pair:
            P_raw(obs|state) = (1-alpha) * P_lower + alpha * P_upper

        Then normalize per state to ensure sum = 1.

        Parameters
        ----------
        alphas : np.ndarray
            Alpha parameters of shape (num_states, num_observations)
            Values should be in [0, 1]

        Returns
        -------
        dict
            Realization: state -> observation -> probability
        """
        if alphas.shape != self._param_shape:
            raise ValueError(
                f"Expected alphas shape {self._param_shape}, got {alphas.shape}"
            )

        realization: Dict[Any, Dict[Any, float]] = {}

        # Add FAIL state
        realization["FAIL"] = {"FAIL": 1.0}

        def project_to_bounded_simplex(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
            """Project x onto {p: lo <= p <= hi, sum(p)=1}.

            Uses iterative proportional redistribution of surplus/deficit mass across
            dimensions with remaining slack. Assumes the bounds are feasible:
              sum(lo) <= 1 <= sum(hi).
            """
            x = np.asarray(x, dtype=float)
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)

            if lo.shape != x.shape or hi.shape != x.shape:
                raise ValueError("Shape mismatch in bounded-simplex projection")
            if lo.sum() > 1.0 + 1e-8 or hi.sum() < 1.0 - 1e-8:
                raise ValueError("Infeasible bounds: cannot satisfy sum=1 within [lo, hi]")

            p = np.clip(x, lo, hi)
            # Iteratively redistribute until sum close to 1
            for _ in range(500):
                s = float(p.sum())
                if abs(s - 1.0) < 1e-10:
                    break
                if s < 1.0:
                    slack = hi - p
                    slack_sum = float(slack.sum())
                    if slack_sum <= 1e-15:
                        break
                    p = p + slack * ((1.0 - s) / slack_sum)
                else:
                    slack = p - lo
                    slack_sum = float(slack.sum())
                    if slack_sum <= 1e-15:
                        break
                    p = p - slack * ((s - 1.0) / slack_sum)
                p = np.clip(p, lo, hi)

            # Final small correction (if needed) without breaking bounds materially
            residual = 1.0 - float(p.sum())
            if abs(residual) > 1e-8:
                if residual > 0:
                    slack = hi - p
                    slack_sum = float(slack.sum())
                    if slack_sum > 0:
                        p = p + slack * (residual / slack_sum)
                else:
                    slack = p - lo
                    slack_sum = float(slack.sum())
                    if slack_sum > 0:
                        p = p + slack * (residual / slack_sum)  # residual is negative
                p = np.clip(p, lo, hi)

            return p

        # For each non-FAIL state
        for i, state in enumerate(self.states):
            lo = np.array([self.ipomdp.P_lower[state].get(obs, 0.0) for obs in self.observations], dtype=float)
            hi = np.array([self.ipomdp.P_upper[state].get(obs, 0.0) for obs in self.observations], dtype=float)
            x = np.zeros(len(self.observations), dtype=float)

            # Interpolate for each observation
            for j, obs in enumerate(self.observations):
                alpha = float(np.clip(alphas[i, j], 0.0, 1.0))
                x[j] = (1.0 - alpha) * lo[j] + alpha * hi[j]

            p = project_to_bounded_simplex(x, lo, hi)
            realization[state] = {obs: float(p[j]) for j, obs in enumerate(self.observations)}

        return realization

    def realization_to_params(
        self,
        realization: Dict[Any, Dict[Any, float]]
    ) -> np.ndarray:
        """Convert realization dict to alpha parameters (inverse operation).

        For each (state, obs) pair, solves:
            P(obs|state) ≈ (1-alpha) * P_lower + alpha * P_upper
        for alpha.

        Note: Due to normalization in params_to_realization, this may not
        be a perfect inverse. Used primarily for initialization.

        Parameters
        ----------
        realization : dict
            Nested dict: state -> observation -> probability

        Returns
        -------
        np.ndarray
            Alpha parameters of shape (num_states, num_observations)
        """
        alphas = np.zeros(self._param_shape)

        for i, state in enumerate(self.states):
            for j, obs in enumerate(self.observations):
                p = realization[state].get(obs, 0.0)
                p_lower = self.ipomdp.P_lower[state].get(obs, 0.0)
                p_upper = self.ipomdp.P_upper[state].get(obs, 0.0)

                # Solve: p = (1-alpha) * p_lower + alpha * p_upper
                # => alpha = (p - p_lower) / (p_upper - p_lower)

                if abs(p_upper - p_lower) > 1e-9:
                    alpha = (p - p_lower) / (p_upper - p_lower)
                    alpha = np.clip(alpha, 0.0, 1.0)
                else:
                    # Interval is degenerate, use midpoint
                    alpha = 0.5

                alphas[i, j] = alpha

        return alphas

    def validate_realization(
        self,
        realization: Dict[Any, Dict[Any, float]]
    ) -> bool:
        """Check if realization satisfies interval constraints.

        Parameters
        ----------
        realization : dict
            Nested dict: state -> observation -> probability

        Returns
        -------
        bool
            True if all probabilities are within intervals
        """
        for state in self.states:
            for obs in self.observations:
                p = realization[state].get(obs, 0.0)
                p_lower = self.ipomdp.P_lower[state].get(obs, 0.0)
                p_upper = self.ipomdp.P_upper[state].get(obs, 0.0)

                if p < p_lower - 1e-6 or p > p_upper + 1e-6:
                    return False

        return True
