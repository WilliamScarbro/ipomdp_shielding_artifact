"""Monte Carlo Safety Evaluation for Shielding Strategies.

This package provides tools for evaluating the safety of shielding strategies
through Monte Carlo simulation. It implements a 2-player game framework:

- Player 1 (Agent): Chooses actions from the shield (best/worst/random strategies)
- Player 2 (Nature): Chooses perception probabilities within IPOMDP intervals

The package measures safety through simulation, tracking three outcomes:
- fail: State reached "FAIL"
- stuck: No allowed actions available
- safe: Completed trial without failure

Supports best/worst/average case analysis via modular initial state sampling.

Features:
- Timestep-level tracking of cumulative safety probabilities
- ExperimentRunner helper class for common experiment setup
- Visualization of safety metrics and timestep evolution

Modules
-------
action_selectors
    Action selection strategies (random, safest, riskiest, RL-based)
perception_models
    Perception models (uniform, adversarial, legacy adapter)
initial_states
    Initial state sampling strategies (random, safe interior, boundary)
simulation
    Core Monte Carlo simulation functions and timestep metrics
evaluator
    High-level MonteCarloSafetyEvaluator class
experiment_runner
    ExperimentConfig and ExperimentRunner helper class
visualization
    Plotting functions for results and timestep evolution
data_structures
    SafetyTrialResult, MCSafetyMetrics, and TimestepMetrics dataclasses
experiments
    Experiment functions for Taxinet case study
"""

# Data structures
from .data_structures import SafetyTrialResult, MCSafetyMetrics, TimestepMetrics

# Action selectors
from .action_selectors import (
    ActionSelector,
    RandomActionSelector,
    UniformFallbackSelector,
    BeliefSelector,
    SingleBeliefSelector,
    ShieldedActionSelector,
    ObservationShieldedActionSelector,
    BeliefShieldedActionSelector,
    SingleBeliefShieldedActionSelector,
    RLActionSelector,
)

# Neural network action selectors
from .neural_action_selector import (
    NeuralActionSelector,
    QNetwork,
    ReplayBuffer,
    ObservationActionEncoder,
)

# Perception models
from .perception_models import (
    PerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    LegacyPerceptionAdapter,
)

# Fixed realization and optimization
from .fixed_realization_model import (
    FixedRealizationPerceptionModel,
    IntervalRealizationParameterizer,
)
from .realization_optimizer import (
    OptimizedRealizationTrainer,
    train_optimal_realization,
)

# Initial state generators
from .initial_states import (
    InitialStateGenerator,
    RandomInitialState,
    SafeInitialState,
    BoundaryInitialState,
)

# Core simulation
from .simulation import (
    run_single_trial,
    run_monte_carlo_trials,
    compute_safety_metrics,
    compute_timestep_metrics,
)

# High-level API
from .evaluator import MonteCarloSafetyEvaluator

# Experiment runner
from .experiment_runner import ExperimentConfig, ExperimentRunner

# Visualization
from .visualization import (
    plot_safety_metrics,
    plot_rl_training_curves,
    plot_two_player_game_results,
    plot_timestep_evolution,
    plot_timestep_comparison,
)

# Experiments
from .experiments import (
    taxinet_monte_carlo_safety_experiment,
    belief_selector_experiment,
    two_player_game_experiment,
    rl_two_player_game_experiment,
)

__all__ = [
    # Data structures
    "SafetyTrialResult",
    "MCSafetyMetrics",
    "TimestepMetrics",
    # Action selectors
    "ActionSelector",
    "RandomActionSelector",
    "UniformFallbackSelector",
    "BeliefSelector",
    "SingleBeliefSelector",
    "ShieldedActionSelector",
    "ObservationShieldedActionSelector",
    "BeliefShieldedActionSelector",
    "SingleBeliefShieldedActionSelector",
    "RLActionSelector",
    # Neural network action selectors
    "NeuralActionSelector",
    "QNetwork",
    "ReplayBuffer",
    "ObservationActionEncoder",
    # Perception models
    "PerceptionModel",
    "UniformPerceptionModel",
    "AdversarialPerceptionModel",
    "LegacyPerceptionAdapter",
    # Fixed realization and optimization
    "FixedRealizationPerceptionModel",
    "IntervalRealizationParameterizer",
    "OptimizedRealizationTrainer",
    "train_optimal_realization",
    # Initial state generators
    "InitialStateGenerator",
    "RandomInitialState",
    "SafeInitialState",
    "BoundaryInitialState",
    # Core simulation
    "run_single_trial",
    "run_monte_carlo_trials",
    "compute_safety_metrics",
    "compute_timestep_metrics",
    # High-level API
    "MonteCarloSafetyEvaluator",
    # Experiment runner
    "ExperimentConfig",
    "ExperimentRunner",
    # Visualization
    "plot_safety_metrics",
    "plot_rl_training_curves",
    "plot_two_player_game_results",
    "plot_timestep_evolution",
    "plot_timestep_comparison",
    # Experiments
    "taxinet_monte_carlo_safety_experiment",
    "belief_selector_experiment",
    "two_player_game_experiment",
    "rl_two_player_game_experiment",
]
