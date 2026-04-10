"""Experiment runner helper class for common experiment setup and evaluation."""

from typing import Callable, Dict, Optional
from dataclasses import dataclass

from ..Models.ipomdp import IPOMDP
from ..Evaluation.runtime_shield import RuntimeImpShield
from .evaluator import MonteCarloSafetyEvaluator
from .perception_models import PerceptionModel


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Attributes
    ----------
    num_trials : int
        Number of Monte Carlo trials to run
    trial_length : int
        Maximum steps per trial
    seed : int, optional
        Random seed for reproducibility
    save_path : str, optional
        Path to save visualization
    track_timesteps : bool
        Whether to compute timestep-level metrics
    action_shield_threshold : float
        Threshold for action shield (probability cutoff)
    """
    num_trials: int = 100
    trial_length: int = 20
    seed: Optional[int] = 42
    save_path: Optional[str] = None
    track_timesteps: bool = False
    action_shield_threshold: float = 0.8


class ExperimentRunner:
    """Helper class for running Monte Carlo safety experiments.

    Encapsulates common setup logic:
    - Creating runtime shield factory
    - Creating evaluator
    - Printing and plotting results

    This class uses composition over inheritance, providing a helper
    that experiments instantiate and use rather than inheriting from.

    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration
    ipomdp : IPOMDP
        The interval POMDP model
    dyn_shield : dict
        Perfect perception shield (dynamics shield)
    evaluator : MonteCarloSafetyEvaluator
        Evaluator instance (set after create_evaluator)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        ipomdp: IPOMDP,
        dyn_shield: Dict
    ):
        """Initialize experiment runner.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration
        ipomdp : IPOMDP
            The interval POMDP model to use
        dyn_shield : dict
            The dynamics shield (perfect perception shield)
        """
        self.config = config
        self.ipomdp = ipomdp
        self.dyn_shield = dyn_shield
        self.evaluator = None

    def create_rt_shield_factory(self) -> Callable[[], RuntimeImpShield]:
        """Create runtime shield factory function.

        Returns
        -------
        callable
            Factory function that returns fresh RuntimeImpShield instances
        """
        from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
        from ..Propagators.lfp_propagator import default_solver

        ipomdp = self.ipomdp
        dyn_shield = self.dyn_shield
        threshold = self.config.action_shield_threshold

        def factory():
            n = len(ipomdp.states)
            template = TemplateFactory.canonical(n)
            polytope = BeliefPolytope.uniform_prior(n)
            propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
            return RuntimeImpShield(dyn_shield, propagator, action_shield=threshold)

        return factory

    def create_evaluator(
        self,
        perception: PerceptionModel,
        rt_shield_factory: Callable[[], RuntimeImpShield]
    ) -> MonteCarloSafetyEvaluator:
        """Create Monte Carlo safety evaluator.

        Parameters
        ----------
        perception : PerceptionModel
            Perception model to use
        rt_shield_factory : callable
            Factory for runtime shield instances

        Returns
        -------
        MonteCarloSafetyEvaluator
            Configured evaluator instance
        """
        self.evaluator = MonteCarloSafetyEvaluator(
            ipomdp=self.ipomdp,
            pp_shield=self.dyn_shield,
            perception=perception,
            rt_shield_factory=rt_shield_factory
        )
        return self.evaluator

    def print_experiment_header(self, title: str, **details):
        """Print formatted experiment header.

        Parameters
        ----------
        title : str
            Experiment title
        **details
            Additional key-value pairs to display
        """
        print("=" * 60)
        print(title)
        print(f"Trials: {self.config.num_trials}, Length: {self.config.trial_length}")
        if self.config.track_timesteps:
            print("Timestep tracking: ENABLED")
        for key, value in details.items():
            print(f"{key}: {value}")
        print("=" * 60)

    def print_results(self, results: Dict, title: str = "RESULTS"):
        """Print formatted results.

        Parameters
        ----------
        results : dict
            Results dictionary (can be flat or nested)
        title : str
            Section title
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        # Handle nested results (2-player game)
        if results and isinstance(next(iter(results.values())), dict):
            for outer_key, inner_dict in results.items():
                print(f"\n{outer_key.upper()}:")
                for inner_key, metrics in inner_dict.items():
                    print(f"  {inner_key:8s}: "
                          f"fail={metrics.fail_rate:.1%}, "
                          f"stuck={metrics.stuck_rate:.1%}, "
                          f"safe={metrics.safe_rate:.1%}")
        else:
            for mode, metrics in results.items():
                print(f"\n{mode.upper()}:")
                print(metrics)
