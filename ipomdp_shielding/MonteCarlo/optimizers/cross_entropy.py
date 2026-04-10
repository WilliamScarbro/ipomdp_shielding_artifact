"""Cross-Entropy Method optimizer for black-box optimization.

The Cross-Entropy Method (CEM) is a simple yet effective derivative-free
optimization algorithm that works well with noisy objectives.

Algorithm:
1. Maintain a distribution (mean, std) over parameters
2. Sample candidates from this distribution
3. Evaluate candidates with objective function
4. Select elite candidates (top performers)
5. Update distribution to match elite set
6. Decay exploration (reduce std) over time

Reference: Rubinstein & Kroese (2004), "The Cross-Entropy Method"
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class CrossEntropyOptimizer:
    """Cross-Entropy Method optimizer for continuous parameters.

    Optimizes parameters in [0, 1]^n to maximize an objective function.

    Attributes
    ----------
    param_shape : tuple
        Shape of parameter array (e.g., (num_states, num_observations))
    num_params : int
        Total number of parameters (product of param_shape)
    config : dict
        Optimization configuration
    mean : np.ndarray
        Current distribution mean
    std : np.ndarray
        Current distribution standard deviation
    history : dict
        Training history (scores, means, stds per iteration)
    """

    def __init__(
        self,
        param_shape: Tuple[int, ...],
        num_candidates: int = 50,
        num_elite: int = 10,
        initial_std: float = 0.3,
        std_decay: float = 0.95,
        min_std: float = 0.01,
        seed: Optional[int] = None
    ):
        """Initialize Cross-Entropy optimizer.

        Parameters
        ----------
        param_shape : tuple
            Shape of parameter array
        num_candidates : int
            Number of candidates to sample per iteration
        num_elite : int
            Number of top candidates to use for distribution update
        initial_std : float
            Initial standard deviation for exploration
        std_decay : float
            Factor to decay std each iteration (0 < decay < 1)
        min_std : float
            Minimum standard deviation (prevents premature convergence)
        seed : int, optional
            Random seed for reproducibility
        """
        self.param_shape = param_shape
        self.num_params = int(np.prod(param_shape))

        self.config = {
            "num_candidates": num_candidates,
            "num_elite": num_elite,
            "initial_std": initial_std,
            "std_decay": std_decay,
            "min_std": min_std,
            "seed": seed
        }

        # Initialize distribution
        self.mean = None
        self.std = None
        self.history = {
            "best_scores": [],
            "mean_scores": [],
            "elite_mean_scores": [],
            "mean_params": [],
            "std_params": []
        }

        if seed is not None:
            np.random.seed(seed)

        self._initialize()

    def _initialize(self):
        """Initialize mean and std for optimization."""
        # Start at midpoint (0.5) with high exploration
        self.mean = 0.5 * np.ones(self.num_params)
        self.std = self.config["initial_std"] * np.ones(self.num_params)

    def _sample_candidates(self) -> np.ndarray:
        """Sample candidate parameters from current distribution.

        Samples from N(mean, std^2) and clips to [0, 1].

        Returns
        -------
        np.ndarray
            Candidates of shape (num_candidates, num_params)
        """
        num_candidates = self.config["num_candidates"]

        # Sample from Gaussian
        candidates = np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=(num_candidates, self.num_params)
        )

        # Clip to [0, 1]
        candidates = np.clip(candidates, 0.0, 1.0)

        return candidates

    def _update_distribution(
        self,
        candidates: np.ndarray,
        scores: np.ndarray
    ):
        """Update distribution based on elite candidates.

        Selects top-K candidates by score and updates mean/std to match
        their distribution.

        Parameters
        ----------
        candidates : np.ndarray
            Candidate parameters of shape (num_candidates, num_params)
        scores : np.ndarray
            Objective scores of shape (num_candidates,)
        """
        num_elite = self.config["num_elite"]

        # Select elite candidates (top scores)
        elite_indices = np.argsort(-scores)[:num_elite]
        elite_candidates = candidates[elite_indices]
        elite_scores = scores[elite_indices]

        # Update mean to elite mean
        self.mean = np.mean(elite_candidates, axis=0)

        # Update std to elite std
        self.std = np.std(elite_candidates, axis=0)

        # Apply decay to prevent premature convergence
        self.std = np.maximum(
            self.config["min_std"],
            self.config["std_decay"] * self.std
        )

        # Log history
        self.history["elite_mean_scores"].append(float(np.mean(elite_scores)))

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        max_iterations: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run Cross-Entropy optimization.

        Parameters
        ----------
        objective_fn : callable
            Function mapping parameter array (shape param_shape) to scalar score
            Higher scores are better (maximization)
        max_iterations : int
            Maximum number of optimization iterations
        verbose : bool
            Whether to print progress

        Returns
        -------
        best_params : np.ndarray
            Best parameters found (shape param_shape)
        best_score : float
            Best objective score achieved
        history : dict
            Training history (scores, parameters per iteration)
        """
        best_params = None
        best_score = -np.inf

        for iteration in range(max_iterations):
            # Sample candidates
            candidates = self._sample_candidates()

            # Evaluate candidates
            scores = np.array([
                objective_fn(c.reshape(self.param_shape))
                for c in candidates
            ])

            # Track best
            iter_best_idx = np.argmax(scores)
            iter_best_score = scores[iter_best_idx]

            if iter_best_score > best_score:
                best_score = iter_best_score
                best_params = candidates[iter_best_idx].reshape(self.param_shape)

            # Update distribution
            self._update_distribution(candidates, scores)

            # Log history
            self.history["best_scores"].append(float(best_score))
            self.history["mean_scores"].append(float(np.mean(scores)))
            self.history["mean_params"].append(self.mean.copy())
            self.history["std_params"].append(self.std.copy())

            # Print progress
            if verbose:
                print(f"Iteration {iteration + 1}/{max_iterations}: "
                      f"best={best_score:.4f}, "
                      f"mean={np.mean(scores):.4f}, "
                      f"elite_mean={self.history['elite_mean_scores'][-1]:.4f}, "
                      f"std={np.mean(self.std):.4f}")

        return best_params, best_score, self.history

    def get_history(self) -> Dict:
        """Get optimization history.

        Returns
        -------
        dict
            Training history including scores and parameters per iteration
        """
        return self.history.copy()
