"""High-level Monte Carlo safety evaluation API.

This module provides the MonteCarloSafetyEvaluator class for running
comprehensive safety evaluations across different scenarios.
"""

from typing import Any, Callable, Dict, List, Optional, Set

from ..Models.ipomdp import IPOMDP
from ..Evaluation.runtime_shield import RuntimeImpShield

from .data_structures import MCSafetyMetrics
from .action_selectors import (
    ActionSelector,
    RandomActionSelector,
    BeliefSelector,
)
from .perception_models import (
    PerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    LegacyPerceptionAdapter,
)
from .initial_states import (
    InitialStateGenerator,
    RandomInitialState,
    SafeInitialState,
    BoundaryInitialState,
)
from .simulation import run_monte_carlo_trials, compute_safety_metrics, compute_timestep_metrics


class MonteCarloSafetyEvaluator:
    """High-level interface for Monte Carlo safety evaluation.

    Evaluates shielding strategies across best/worst/average case scenarios
    by varying action selection strategies and perception models.

    This implements a 2-player game framework:
    - Player 1 (Agent): Chooses actions from the shield
    - Player 2 (Nature): Chooses perception probabilities within intervals

    Nature can be cooperative (random perception) or adversarial (maximizing failure).
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        perception: "PerceptionModel | Callable[[Any], Any]",
        rt_shield_factory: Callable[[], RuntimeImpShield]
    ):
        """Initialize evaluator.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        perception : PerceptionModel or callable
            Default perception model or legacy function
        rt_shield_factory : callable
            Factory function returning fresh RuntimeImpShield instance
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.rt_shield_factory = rt_shield_factory

        # Convert legacy perception to PerceptionModel
        if callable(perception) and not isinstance(perception, PerceptionModel):
            self.perception = LegacyPerceptionAdapter(perception)
        else:
            self.perception = perception

    def evaluate(
        self,
        action_selector: ActionSelector,
        num_trials: int = 100,
        trial_length: int = 20,
        sampling_modes: Optional[List[str]] = None,
        store_trajectories: bool = False,
        seed: Optional[int] = None,
        perception_model: Optional[PerceptionModel] = None,
        compute_timestep_metrics_flag: bool = False
    ) -> tuple:
        """Run Monte Carlo evaluation across sampling modes.

        Parameters
        ----------
        action_selector : ActionSelector
            Strategy for selecting actions
        num_trials : int
            Number of trials per sampling mode
        trial_length : int
            Maximum steps per trial
        sampling_modes : list of str, optional
            Sampling modes to evaluate. Defaults to ["random", "best_case", "worst_case"]
        store_trajectories : bool
            Whether to store full trajectories (memory intensive)
        seed : int, optional
            Random seed for reproducibility
        perception_model : PerceptionModel, optional
            Override the default perception model for this evaluation
        compute_timestep_metrics_flag : bool
            Whether to compute timestep-level cumulative metrics (default: False)

        Returns
        -------
        tuple
            (results_by_mode, timestep_metrics_by_mode)
            - results_by_mode: Dict[str, MCSafetyMetrics]
            - timestep_metrics_by_mode: Dict[str, TimestepMetrics] or None
        """
        if sampling_modes is None:
            sampling_modes = ["random", "best_case", "worst_case"]

        # Use provided perception model or default
        perception = perception_model if perception_model is not None else self.perception

        # Map mode names to generator classes
        generator_map = {
            "random": RandomInitialState(),
            "best_case": SafeInitialState(),
            "worst_case": BoundaryInitialState()
        }
        # generator_map = {
        #     "random": SafeInitialState(),
        #     "best_case": SafeInitialState(),
        #     "worst_case": SafeInitialState()
        # }

        results_by_mode = {}
        timestep_metrics_by_mode = {} if compute_timestep_metrics_flag else None

        for mode in sampling_modes:
            if mode not in generator_map:
                raise ValueError(f"Unknown sampling mode: {mode}")

            generator = generator_map[mode]

            # Run trials for this mode
            results = run_monte_carlo_trials(
                ipomdp=self.ipomdp,
                pp_shield=self.pp_shield,
                perception=perception,
                rt_shield_factory=self.rt_shield_factory,
                action_selector=action_selector,
                initial_generator=generator,
                num_trials=num_trials,
                trial_length=trial_length,
                store_trajectories=store_trajectories,
                seed=seed
            )

            # Compute metrics
            metrics = compute_safety_metrics(results)
            results_by_mode[mode] = metrics

            # Compute timestep metrics if requested
            if compute_timestep_metrics_flag:
                timestep_metrics_by_mode[mode] = compute_timestep_metrics(results, trial_length)

        return results_by_mode, timestep_metrics_by_mode

    def evaluate_two_player_game(
        self,
        num_trials: int = 100,
        trial_length: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, MCSafetyMetrics]]:
        """Evaluate all combinations of agent and nature strategies.

        This runs the full 2-player game analysis:
        - Agent strategies: random, best (highest belief prob), worst (lowest belief prob)
        - Nature strategies: cooperative (uniform), adversarial

        Parameters
        ----------
        num_trials : int
            Number of trials per combination
        trial_length : int
            Maximum steps per trial
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Nested dict: nature_strategy -> agent_strategy -> MCSafetyMetrics
        """
        results = {}

        # Define perception models (Nature's strategies)
        perception_models = {
            "cooperative": UniformPerceptionModel(),
            "adversarial": AdversarialPerceptionModel(self.pp_shield)
        }

        # Define action selectors (Agent's strategies)
        action_selectors = {
            "random": RandomActionSelector(),
            "best": BeliefSelector(mode="best"),
            "worst": BeliefSelector(mode="worst"),
        }

        # Use random initial state for all evaluations
        initial_generator = RandomInitialState()

        for nature_name, perception_model in perception_models.items():
            print(f"\nEvaluating nature strategy: {nature_name}")
            results[nature_name] = {}

            for agent_name, action_selector in action_selectors.items():
                print(f"  Agent strategy: {agent_name}...")

                # Reset RL selector state if needed
                if hasattr(action_selector, 'reset'):
                    action_selector.reset()

                trial_results = run_monte_carlo_trials(
                    ipomdp=self.ipomdp,
                    pp_shield=self.pp_shield,
                    perception=perception_model,
                    rt_shield_factory=self.rt_shield_factory,
                    action_selector=action_selector,
                    initial_generator=initial_generator,
                    num_trials=num_trials,
                    trial_length=trial_length,
                    store_trajectories=False,
                    seed=seed
                )

                metrics = compute_safety_metrics(trial_results)
                results[nature_name][agent_name] = metrics

                print(f"    fail={metrics.fail_rate:.2%}, "
                      f"stuck={metrics.stuck_rate:.2%}, "
                      f"safe={metrics.safe_rate:.2%}")

        return results

