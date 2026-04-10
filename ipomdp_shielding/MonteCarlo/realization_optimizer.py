"""Training infrastructure for optimal interval realization.

This module provides the OptimizedRealizationTrainer class for learning
fixed interval realizations that maximize failure probability via black-box
optimization.
"""

from typing import Any, Callable, Dict, Optional, Set
import json
import numpy as np

from ..Models.ipomdp import IPOMDP
from ..Evaluation.runtime_shield import RuntimeImpShield

from .fixed_realization_model import (
    FixedRealizationPerceptionModel,
    IntervalRealizationParameterizer
)
from .optimizers import CrossEntropyOptimizer
from .action_selectors import ActionSelector, RandomActionSelector
from .initial_states import InitialStateGenerator, RandomInitialState
from .simulation import run_monte_carlo_trials, compute_safety_metrics


class OptimizedRealizationTrainer:
    """Trainer for learning optimal fixed interval realizations.

    Uses Cross-Entropy Method to search the space of interval realizations
    and find one that maximizes failure probability via Monte Carlo evaluation.

    Attributes
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield
    rt_shield_factory : callable
        Factory for runtime shields
    action_selector : ActionSelector
        Agent's action selection strategy
    initial_generator : InitialStateGenerator
        Strategy for sampling initial states
    parameterizer : IntervalRealizationParameterizer
        Converts between alpha parameters and realizations
    optimizer : CrossEntropyOptimizer
        Black-box optimizer
    config : dict
        Training configuration
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        rt_shield_factory: Callable[[], RuntimeImpShield],
        action_selector: ActionSelector,
        initial_generator: InitialStateGenerator,
        num_candidates: int = 50,
        num_trials_per_candidate: int = 20,
        num_elite: int = 10,
        initial_std: float = 0.3,
        std_decay: float = 0.95,
        min_std: float = 0.01,
        seed: Optional[int] = None
    ):
        """Initialize realization trainer.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield
        rt_shield_factory : callable
            Factory returning fresh RuntimeImpShield instances
        action_selector : ActionSelector
            Agent's action selection strategy
        initial_generator : InitialStateGenerator
            Strategy for sampling initial states
        num_candidates : int
            Number of candidates to sample per CEM iteration
        num_trials_per_candidate : int
            Number of Monte Carlo trials for evaluating each candidate
        num_elite : int
            Number of top candidates for CEM distribution update
        initial_std : float
            Initial exploration standard deviation
        std_decay : float
            Standard deviation decay factor per iteration
        min_std : float
            Minimum standard deviation (prevents premature convergence)
        seed : int, optional
            Random seed for reproducibility
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.rt_shield_factory = rt_shield_factory
        self.action_selector = action_selector
        self.initial_generator = initial_generator

        # Create parameterizer
        self.parameterizer = IntervalRealizationParameterizer(ipomdp)

        # Create optimizer
        self.optimizer = CrossEntropyOptimizer(
            param_shape=self.parameterizer.param_shape,
            num_candidates=num_candidates,
            num_elite=num_elite,
            initial_std=initial_std,
            std_decay=std_decay,
            min_std=min_std,
            seed=seed
        )

        # Store configuration
        self.config = {
            "num_candidates": num_candidates,
            "num_trials_per_candidate": num_trials_per_candidate,
            "num_elite": num_elite,
            "initial_std": initial_std,
            "std_decay": std_decay,
            "min_std": min_std,
            "seed": seed
        }

    def train(
        self,
        trial_length: int = 20,
        max_iterations: int = 100,
        verbose: bool = True
    ) -> FixedRealizationPerceptionModel:
        """Train optimal realization via Cross-Entropy Method.

        Parameters
        ----------
        trial_length : int
            Maximum steps per Monte Carlo trial
        max_iterations : int
            Maximum CEM iterations
        verbose : bool
            Whether to print progress

        Returns
        -------
        FixedRealizationPerceptionModel
            Trained perception model with optimal realization
        """
        # Store trial length for objective function
        self._trial_length = trial_length

        if verbose:
            print("=" * 60)
            print("Training Optimal Fixed Realization")
            print("=" * 60)
            print(f"Parameter shape: {self.parameterizer.param_shape}")
            print(f"Candidates per iteration: {self.config['num_candidates']}")
            print(f"Trials per candidate: {self.config['num_trials_per_candidate']}")
            print(f"Max iterations: {max_iterations}")
            print("=" * 60)

        # Define objective function
        def objective(alphas: np.ndarray) -> float:
            return self._evaluate_realization(alphas)

        # Run optimization
        best_alphas, best_score, history = self.optimizer.optimize(
            objective_fn=objective,
            max_iterations=max_iterations,
            verbose=verbose
        )

        # Convert best alphas to realization
        best_realization = self.parameterizer.params_to_realization(best_alphas)

        # Validate
        if not self.parameterizer.validate_realization(best_realization):
            print("WARNING: Best realization violates interval constraints!")

        # Create metadata
        metadata = {
            "objective_score": float(best_score),
            "training_config": self.config,
            "trial_length": trial_length,
            "max_iterations": max_iterations,
            "action_selector": str(self.action_selector.__class__.__name__),
            "initial_generator": str(self.initial_generator.__class__.__name__),
            "training_history": {
                "best_scores": [float(s) for s in history["best_scores"]],
                "mean_scores": [float(s) for s in history["mean_scores"]],
                "elite_mean_scores": [float(s) for s in history["elite_mean_scores"]]
            }
        }

        if verbose:
            print("=" * 60)
            print(f"Training complete! Best score: {best_score:.4f}")
            print("=" * 60)

        return FixedRealizationPerceptionModel(
            realization=best_realization,
            metadata=metadata
        )

    def _evaluate_realization(self, alphas: np.ndarray) -> float:
        """Evaluate a realization candidate via Monte Carlo trials.

        Parameters
        ----------
        alphas : np.ndarray
            Alpha parameters defining the realization

        Returns
        -------
        float
            Failure rate (objective score to maximize)
        """
        # Convert to realization
        realization = self.parameterizer.params_to_realization(alphas)

        # Create temporary perception model
        perception = FixedRealizationPerceptionModel(realization)

        # Run Monte Carlo trials
        results = run_monte_carlo_trials(
            ipomdp=self.ipomdp,
            pp_shield=self.pp_shield,
            perception=perception,
            rt_shield_factory=self.rt_shield_factory,
            action_selector=self.action_selector,
            initial_generator=self.initial_generator,
            num_trials=self.config["num_trials_per_candidate"],
            trial_length=self._trial_length,
            store_trajectories=False,
            seed=None  # Don't reset seed for each candidate
        )

        # Compute metrics
        metrics = compute_safety_metrics(results)

        # Return failure rate as objective score
        return metrics.fail_rate

    def save_training_history(self, filepath: str):
        """Export training history to JSON.

        Parameters
        ----------
        filepath : str
            Path to save JSON file
        """
        history = self.optimizer.get_history()

        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {
            "best_scores": [float(s) for s in history["best_scores"]],
            "mean_scores": [float(s) for s in history["mean_scores"]],
            "elite_mean_scores": [float(s) for s in history["elite_mean_scores"]],
            "config": self.config
        }

        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)


def train_optimal_realization(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, Set[Any]],
    rt_shield_factory: Callable[[], RuntimeImpShield],
    action_selector: Optional[ActionSelector] = None,
    initial_generator: Optional[InitialStateGenerator] = None,
    num_candidates: int = 50,
    num_trials_per_candidate: int = 20,
    max_iterations: int = 100,
    trial_length: int = 20,
    num_elite: int = 10,
    initial_std: float = 0.3,
    std_decay: float = 0.95,
    min_std: float = 0.01,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> FixedRealizationPerceptionModel:
    """High-level API for training optimal fixed realization.

    Convenience function with sensible defaults for training a fixed
    realization that maximizes failure probability.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield
    rt_shield_factory : callable
        Factory returning fresh RuntimeImpShield instances
    action_selector : ActionSelector, optional
        Agent's action selection strategy (default: RandomActionSelector)
    initial_generator : InitialStateGenerator, optional
        Initial state sampling strategy (default: RandomInitialState)
    num_candidates : int
        Number of candidates per CEM iteration
    num_trials_per_candidate : int
        Number of Monte Carlo trials per candidate evaluation
    max_iterations : int
        Maximum CEM iterations
    trial_length : int
        Maximum steps per trial
    num_elite : int
        Number of elite candidates for CEM update
    initial_std : float
        Initial exploration standard deviation
    std_decay : float
        Std decay factor per iteration
    min_std : float
        Minimum standard deviation
    seed : int, optional
        Random seed for reproducibility
    save_path : str, optional
        If provided, save trained model to this path
    verbose : bool
        Whether to print progress

    Returns
    -------
    FixedRealizationPerceptionModel
        Trained perception model with optimal realization
    """
    # Use defaults if not provided
    if action_selector is None:
        action_selector = RandomActionSelector()
    if initial_generator is None:
        initial_generator = RandomInitialState()

    # Create trainer
    trainer = OptimizedRealizationTrainer(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        rt_shield_factory=rt_shield_factory,
        action_selector=action_selector,
        initial_generator=initial_generator,
        num_candidates=num_candidates,
        num_trials_per_candidate=num_trials_per_candidate,
        num_elite=num_elite,
        initial_std=initial_std,
        std_decay=std_decay,
        min_std=min_std,
        seed=seed
    )

    # Train
    optimal_perception = trainer.train(
        trial_length=trial_length,
        max_iterations=max_iterations,
        verbose=verbose
    )

    # Save if requested
    if save_path is not None:
        optimal_perception.save(save_path)
        if verbose:
            print(f"\nSaved optimal realization to: {save_path}")

    return optimal_perception
