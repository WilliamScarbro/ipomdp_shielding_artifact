"""Shield evaluation and runtime shield construction."""

from .runtime_shield import RuntimeImpShield
from .shield_evaluator import evaluate_runtime_shield
from .support_mdp_builder import SupportMDPBuilder
from .carr_shield import CarrShield
from .propagator_comparison import compare_propagators, run_simple_test
from .template_comparison import (
    compare_templates,
    compare_templates_scripted,
    run_scripted_comparisons,
    run_taxinet_scripted_comparison,
    run_toy_comparison,
    run_toy_comparison_averaged,
    run_taxinet_comparison,
    run_taxinet_comparison_averaged,
    run_multiple_comparisons,
    run_all_comparisons,
    plot_comparison,
    plot_averaged_comparison,
    plot_metrics_individually,
    plot_averaged_metrics_individually,
    create_toy_model,
    GroundTruthComparisonResult,
    compute_ground_truth_comparison,
    plot_ground_truth_comparison,
    run_taxinet_ground_truth_comparison,
)
from .report_runner import ReportRunner, ScriptedReportRunner
from .script_library import RunScript, ScriptLibrary, generate_script
from .metrics import (
    MetricValue,
    MetricsCollector,
    StepMetrics,
    ApproximationMetrics_1,
    GroundTruthComparisonMetrics,
    StepPredictions,
    compute_template_spread,
    compute_volume_proxy,
    compute_safest_action_prob,
)
from .lfp_reporters import (
    ComparisonResult,
    LFPReporter,
    ScriptedLFPReporter,
)
from .single_run import (
    DebugConfig,
    RunStatistics,
    run_single_debug,
    run_taxinet_debug,
    compute_top_k_states,
)
# Re-export from MonteCarlo package for backward compatibility
from ..MonteCarlo import (
    # Action selectors (Agent's strategy)
    ActionSelector,
    RandomActionSelector,
    UniformFallbackSelector,
    BeliefSelector,
    ShieldedActionSelector,
    ObservationShieldedActionSelector,
    BeliefShieldedActionSelector,
    SingleBeliefShieldedActionSelector,
    RLActionSelector,
    # Perception models (2-player game: Nature's strategy)
    PerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    LegacyPerceptionAdapter,
    # Initial state generators
    InitialStateGenerator,
    RandomInitialState,
    SafeInitialState,
    BoundaryInitialState,
    # Data structures
    SafetyTrialResult,
    MCSafetyMetrics,
    TimestepMetrics,
    # Core functions
    run_single_trial,
    run_monte_carlo_trials,
    compute_safety_metrics,
    compute_timestep_metrics,
    MonteCarloSafetyEvaluator,
    # Experiment runner
    ExperimentConfig,
    ExperimentRunner,
    # Visualization
    plot_safety_metrics,
    plot_two_player_game_results,
    plot_rl_training_curves,
    plot_timestep_evolution,
    plot_timestep_comparison,
    # Experiment functions
    taxinet_monte_carlo_safety_experiment,
    belief_selector_experiment,
    two_player_game_experiment,
    rl_two_player_game_experiment,
)

__all__ = [
    'RuntimeImpShield',
    'evaluate_runtime_shield',
    'SupportMDPBuilder',
    'CarrShield',
    'compare_propagators',
    'run_simple_test',
    'compare_templates',
    'compare_templates_scripted',
    'run_scripted_comparisons',
    'run_taxinet_scripted_comparison',
    'run_toy_comparison',
    'run_toy_comparison_averaged',
    'run_taxinet_comparison',
    'run_taxinet_comparison_averaged',
    'run_multiple_comparisons',
    'run_all_comparisons',
    'plot_comparison',
    'plot_averaged_comparison',
    'plot_metrics_individually',
    'plot_averaged_metrics_individually',
    'create_toy_model',
    'ReportRunner',
    'ScriptedReportRunner',
    'RunScript',
    'ScriptLibrary',
    'generate_script',
    'MetricValue',
    'MetricsCollector',
    'StepMetrics',
    'ApproximationMetrics_1',
    'GroundTruthComparisonMetrics',
    'StepPredictions',
    'compute_template_spread',
    'compute_volume_proxy',
    'compute_safest_action_prob',
    'GroundTruthComparisonResult',
    'compute_ground_truth_comparison',
    'plot_ground_truth_comparison',
    'run_taxinet_ground_truth_comparison',
    'ComparisonResult',
    'LFPReporter',
    'ScriptedLFPReporter',
    'DebugConfig',
    'RunStatistics',
    'run_single_debug',
    'run_taxinet_debug',
    'compute_top_k_states',
    # Action selectors (Agent's strategy)
    'ActionSelector',
    'RandomActionSelector',
    'UniformFallbackSelector',
    'BeliefSelector',
    'ShieldedActionSelector',
    'ObservationShieldedActionSelector',
    'BeliefShieldedActionSelector',
    'SingleBeliefShieldedActionSelector',
    'RLActionSelector',
    # Perception models (Nature's strategy)
    'PerceptionModel',
    'UniformPerceptionModel',
    'AdversarialPerceptionModel',
    'LegacyPerceptionAdapter',
    # Initial state generators
    'InitialStateGenerator',
    'RandomInitialState',
    'SafeInitialState',
    'BoundaryInitialState',
    # Data structures
    'SafetyTrialResult',
    'MCSafetyMetrics',
    'TimestepMetrics',
    # Core functions
    'run_single_trial',
    'run_monte_carlo_trials',
    'compute_safety_metrics',
    'compute_timestep_metrics',
    'MonteCarloSafetyEvaluator',
    # Experiment runner
    'ExperimentConfig',
    'ExperimentRunner',
    # Visualization
    'plot_safety_metrics',
    'plot_two_player_game_results',
    'plot_rl_training_curves',
    'plot_timestep_evolution',
    'plot_timestep_comparison',
    # Experiment functions
    'taxinet_monte_carlo_safety_experiment',
    'belief_selector_experiment',
    'two_player_game_experiment',
    'rl_two_player_game_experiment',
]
