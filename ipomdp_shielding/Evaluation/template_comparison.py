"""
Comparative analysis of template functions for LFP belief propagation.

Compares different template strategies by measuring:
1. Approximation decay: growth of BeliefPolytope bounds over time
2. Shield permissivity: how many actions remain allowed under the approximation

Templates compared:
- Canonical: standard basis vectors (bound each state probability)
- Safe-set indicators: bound probability mass in action-safe sets
- Hybrid: combination of canonical and safe-set indicators
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import random
import os

from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Models import IPOMDP
from ..Propagators import (
    LFPPropagator,
    BeliefPolytope,
    Template,
    TemplateFactory,
)
from ..Propagators.lfp_propagator import default_solver

from .runtime_shield import RuntimeImpShield
from .script_library import RunScript, ScriptLibrary
from .metrics import (
    MetricValue,
    MetricsCollector,
    StepMetrics,
    ApproximationMetrics_1,
    GroundTruthComparisonMetrics,
)
from .lfp_reporters import (
    ComparisonResult,
    LFPReporter,
    ScriptedLFPReporter,
)




# ============================================================
# Run functions
# ============================================================

def run_lfp_propagation(
        ipomdp: IPOMDP,
        pp_shield : "Dict State -> {Action}",
        template : Template,
        perception : "State -> Obs",
        initial : "State x Action",
        verbose : bool = False,
        action_shield : float = 0.8,
        length : int = 20
):

    n = len(ipomdp.states)
    polytope = BeliefPolytope.uniform_prior(n)
    propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
    rt_shield = RuntimeImpShield(pp_shield, propagator, action_shield)

    all_actions = list(ipomdp.actions)
    action_selector = lambda actions: random.choice(actions) if actions else random.choice(all_actions)
    lfp_reporter = LFPReporter(ipomdp, perception, rt_shield, action_selector, length, initial)

    return lfp_reporter.run()


def run_scripted_lfp_propagation(
        ipomdp: IPOMDP,
        pp_shield: Dict,
        template: Template,
        script: RunScript,
        action_shield: float = 0.8,
        metrics_collector: Optional[MetricsCollector] = None
) -> ComparisonResult:
    """Run LFP propagation on a pre-recorded script.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    template : Template
        Template for LFP propagation
    script : RunScript
        Pre-recorded trajectory to replay
    action_shield : float
        Minimum required probability for an action to be allowed
    metrics_collector : MetricsCollector, optional
        Metrics collector to use. Defaults to ApproximationMetrics_1.

    Returns
    -------
    ComparisonResult
        Results including metrics at each step
    """
    n = len(ipomdp.states)
    polytope = BeliefPolytope.uniform_prior(n)
    propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
    rt_shield = RuntimeImpShield(pp_shield, propagator, action_shield)

    reporter = ScriptedLFPReporter(script, rt_shield, metrics_collector)
    return reporter.run()

    
                
# def run_lfp_propagation(
#     ipomdp: IPOMDP,
#     template: Template,
#     initial: State,
#     # history: List[Tuple],
#     verbose: bool = False
# ) -> ComparisonResult:
#     """
#     Run LFP propagation with a specific template and collect metrics.
#     """
#     n = len(ipomdp.states)
#     solver = default_solver()
#     propagator = LFPPropagator(model=ipomdp, template=template, solver=solver)

#     polytope = BeliefPolytope.uniform_prior(n)
#     metrics = []

#     # Initial metrics
#     spread, min_b, max_b = compute_template_spread(polytope, template)
#     vol = compute_volume_proxy(polytope, template)
#     safest_prob, action_probs = compute_shield_permissivity(polytope, ipomdp)

#     metrics.append(ApproximationMetrics(
#         step=0,
#         template_spread=spread,
#         volume_proxy=vol,
#         min_state_bounds=min_b,
#         max_state_bounds=max_b,
#         safest_action_prob=safest_prob,
#         action_safe_probs=action_probs
#     ))

#     if verbose:
#         print(f"Step 0: spread={spread:.4f}, volume={vol:.6f}, safest_prob={safest_prob:.4f}")

#     current_state = initial
#     for step in range(length):

        
#         polytope = propagator.propagate(polytope, action, obs)

#         spread, min_b, max_b = compute_template_spread(polytope, template)
#         vol = compute_volume_proxy(polytope, template)
#         safest_prob, action_probs = compute_shield_permissivity(polytope, ipomdp)

#         metrics.append(ApproximationMetrics(
#             step=step + 1,
#             template_spread=spread,
#             volume_proxy=vol,
#             min_state_bounds=min_b,
#             max_state_bounds=max_b,
#             safest_action_prob=safest_prob,
#             action_safe_probs=action_probs
#         ))

#         if verbose:
#             print(f"Step {step+1}: spread={spread:.4f}, volume={vol:.6f}, safest_prob={safest_prob:.4f}")

#     return ComparisonResult(
#         template_name=getattr(template, 'name', 'unnamed'),
#         template=template,
#         history=history,
#         metrics=metrics,
#         final_polytope=polytope
#     )


def create_templates_for_ipomdp(
    ipomdp: IPOMDP,
    include_canonical: bool = True,
    include_safe_sets: bool = True,
    include_hybrid: bool = True
) -> Dict[str, Template]:
    """
    Create different template types for comparison.
    """
    n = len(ipomdp.states)
    states = list(ipomdp.states)
    actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)

    templates = {}

    if include_canonical:
        canonical = TemplateFactory.canonical(n)
        canonical.name = "canonical"
        templates["canonical"] = canonical

    if include_safe_sets:
        # Build safe sets for each action
        safe_sets = {}
        for action in actions:
            safe_indices = []
            for i, s in enumerate(states):
                next_states = ipomdp.T.get((s, action), {})
                if "FAIL" not in next_states or next_states.get("FAIL", 0) < 0.5:
                    safe_indices.append(i)
            if safe_indices:
                safe_sets[f"safe_{action}"] = safe_indices

        if safe_sets:
            safe_template = TemplateFactory.safe_set_indicators(n, safe_sets)
            safe_template.name = "safe_sets"
            templates["safe_sets"] = safe_template

    if include_hybrid and "canonical" in templates and "safe_sets" in templates:
        hybrid = TemplateFactory.hybrid([templates["canonical"], templates["safe_sets"]])
        hybrid.name = "hybrid"
        templates["hybrid"] = hybrid

    return templates


def compare_templates(
        ipomdp: IPOMDP,
        pp_shield : "Dict (State -> Col Action)",
        perception : "State -> Obs",
        initial : "State x Action",
        templates: Optional[Dict[str, Template]] = None,
        verbose: bool = False
) -> Dict[str, ComparisonResult]:
    """
    Compare multiple template strategies on the same IPOMDP and history.
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp, include_safe_sets=False)

    results = {}
    for name, template in templates.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running template: {name}")
            print(f"{'='*60}")

        result = run_lfp_propagation(
            ipomdp,
            pp_shield,
            template,
            perception,
            initial,
        )
        result.template_name = name
        results[name] = result

    return results


def compare_templates_scripted(
        ipomdp: IPOMDP,
        pp_shield: Dict,
        script: RunScript,
        templates: Optional[Dict[str, Template]] = None,
        action_shield: float = 0.8,
        verbose: bool = False
) -> Dict[str, ComparisonResult]:
    """Compare multiple template strategies on the same scripted trajectory.

    Unlike compare_templates which generates new trajectories for each template,
    this uses a pre-recorded script ensuring identical conditions for all templates.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    script : RunScript
        Pre-recorded trajectory to replay
    templates : dict, optional
        Templates to compare. If None, creates default templates.
    action_shield : float
        Minimum required probability for an action to be allowed
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping from template name to ComparisonResult
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp, include_safe_sets=False)

    results = {}
    for name, template in templates.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running template: {name}")
            print(f"{'='*60}")

        result = run_scripted_lfp_propagation(
            ipomdp,
            pp_shield,
            template,
            script,
            action_shield
        )
        result.template_name = name
        results[name] = result

    return results


def run_scripted_comparisons(
        ipomdp: IPOMDP,
        pp_shield: Dict,
        library: ScriptLibrary,
        templates: Optional[Dict[str, Template]] = None,
        action_shield: float = 0.8,
        verbose: bool = False
) -> "Dict[str, AveragedComparisonResult]":
    """Run template comparison on all scripts in a library and average results.
 
    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    library : ScriptLibrary
        Library of pre-recorded scripts
    templates : dict, optional
        Templates to compare. If None, creates default templates.
    action_shield : float
        Minimum required probability for an action to be allowed
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping from template name to AveragedComparisonResult
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp)

    all_results = {name: [] for name in templates}

    for i, script in enumerate(library):
        if verbose:
            print(f"Script {i + 1}/{len(library)}")

        results = compare_templates_scripted(
            ipomdp,
            pp_shield,
            script,
            templates,
            action_shield,
            verbose=False
        )

        for name, result in results.items():
            all_results[name].append(result)

    # Average the results
    averaged_results = {}
    for name, runs in all_results.items():
        if not runs:
            continue

        num_steps = max(len(r.metrics) for r in runs)
        if num_steps == 0:
            continue

        averaged_metrics = []
        for step in range(num_steps):
            # Only include runs that have data at this step
            spreads = [r.metrics[step].template_spread for r in runs if step < len(r.metrics)]
            volumes = [r.metrics[step].volume_proxy for r in runs if step < len(r.metrics)]
            probs = [r.metrics[step].safest_action_prob for r in runs if step < len(r.metrics)]

            if not spreads:  # Skip if no runs have data at this step
                continue

            averaged_metrics.append(AveragedMetrics(
                step=step,
                template_spread_mean=np.mean(spreads),
                template_spread_std=np.std(spreads),
                volume_proxy_mean=np.mean(volumes),
                volume_proxy_std=np.std(volumes),
                safest_action_prob_mean=np.mean(probs),
                safest_action_prob_std=np.std(probs)
            ))

        averaged_results[name] = AveragedComparisonResult(
            template_name=name,
            num_runs=len(runs),
            num_steps=len(averaged_metrics),
            metrics=averaged_metrics
        )

    return averaged_results


def run_taxinet_scripted_comparison(
        num_scripts: int = 10,
        script_length: int = 20,
        seed: Optional[int] = None,
        verbose: bool = True,
        save_dir: Optional[str] = None
):
    """Run scripted template comparison on the Taxinet model.

    Generates a library of scripts using perfect perception shielding,
    then compares templates on identical trajectories.

    Parameters
    ----------
    num_scripts : int
        Number of scripts to generate
    script_length : int
        Length of each script
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress
    save_dir : str, optional
        Directory to save metric plots (one plot per metric)
    """
    print("=" * 60)
    print(f"TAXINET SCRIPTED COMPARISON ({num_scripts} scripts, length {script_length})")
    print("=" * 60)

    initial = ((0, 0), 0)
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    # Generate script library
    if verbose:
        print(f"\nGenerating {num_scripts} scripts...")

    library = ScriptLibrary.generate(
        ipomdp,
        dyn_shield,
        perception,
        initial,
        num_scripts,
        script_length,
        seed=seed
    )

    if verbose:
        print(f"Generated {len(library)} scripts")
        avg_len = np.mean([s.length for s in library])
        print(f"Average script length: {avg_len:.1f}")

    templates = create_templates_for_ipomdp(ipomdp, include_safe_sets=False)

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")

    results = run_scripted_comparisons(
        ipomdp,
        dyn_shield,
        library,
        templates,
        verbose=verbose
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (SCRIPTED)")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread_mean:.4f} +/- {final.template_spread_std:.4f}")
        print(f"  Final volume: {final.volume_proxy_mean:.6e} +/- {final.volume_proxy_std:.6e}")
        print(f"  Safest prob:  {final.safest_action_prob_mean:.4f} +/- {final.safest_action_prob_std:.4f}")

    # Save plots to subdirectory (one per metric)
    if save_dir:
        metrics_collector = ApproximationMetrics_1()
        plot_averaged_metrics_individually(
            results,
            metrics_collector,
            save_dir,
            title_prefix="Taxinet Scripted"
        )

    return results, library


def plot_comparison(
    results: Dict[str, ComparisonResult],
    title: str = "Template Comparison",
    save_path: Optional[str] = None
):
    """
    Plot comparison of template strategies.

    Creates a figure with:
    - Template spread over time
    - Volume proxy over time (log scale)
    - Safest action probability over time
    - Spread normalized by initial
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: Template spread
    ax1 = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        spreads = [m.template_spread for m in result.metrics]
        ax1.plot(steps, spreads, 'o-', label=name, color=color)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Template Spread')
    ax1.set_title('Approximation Growth (Template Spread)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume proxy (log scale)
    ax2 = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        volumes = [max(m.volume_proxy, 1e-20) for m in result.metrics]  # Avoid log(0)
        ax2.semilogy(steps, volumes, 's-', label=name, color=color)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Volume Proxy (log scale)')
    ax2.set_title('Approximation Growth (Volume Proxy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Safest action probability
    ax3 = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        probs = [m.safest_action_prob for m in result.metrics]
        ax3.plot(steps, probs, '^-', label=name, color=color, markersize=8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Safest Action Probability')
    ax3.set_title('Shield Permissivity (higher = more permissive)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spread normalized by step 1 (skip step 0 which is often 0)
    ax4 = axes[1, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        spreads = [m.template_spread for m in result.metrics]
        # Normalize by first non-zero spread
        initial = next((s for s in spreads if s > 0), 1.0)
        normalized = [s / initial for s in spreads]
        ax4.plot(steps, normalized, 'd-', label=name, color=color)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Spread')
    ax4.set_title('Relative Approximation Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_metrics_individually(
    results: Dict[str, ComparisonResult],
    metrics_collector: MetricsCollector,
    save_dir: str,
    title_prefix: str = "Template Comparison"
):
    """Plot each metric to a separate file in a subdirectory.

    Parameters
    ----------
    results : dict
        Mapping from template name to ComparisonResult
    metrics_collector : MetricsCollector
        The metrics collector that defines which metrics to plot
    save_dir : str
        Directory to save plots (will be created if needed)
    title_prefix : str
        Prefix for plot titles
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for metric_name in metrics_collector.metric_names():
        config = metrics_collector.get_plot_config(metric_name)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{title_prefix}: {config['display_name']}")

        for (name, result), color in zip(results.items(), colors):
            steps = [m.step for m in result.metrics]
            values = [m.get(metric_name, 0.0) for m in result.metrics]

            if config.get('use_log_scale', False):
                values = [max(v, 1e-20) for v in values]
                ax.semilogy(steps, values, 'o-', label=name, color=color)
            else:
                ax.plot(steps, values, 'o-', label=name, color=color)

        ax.set_xlabel('Step')
        ax.set_ylabel(config.get('ylabel', metric_name))

        if config.get('ylim'):
            ax.set_ylim(config['ylim'])

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{metric_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(metrics_collector.metric_names())} metric plots to {save_dir}")


def plot_averaged_metrics_individually(
    results: Dict[str, "AveragedComparisonResult"],
    metrics_collector: MetricsCollector,
    save_dir: str,
    title_prefix: str = "Template Comparison (Averaged)"
):
    """Plot each averaged metric to a separate file with error bands.

    Parameters
    ----------
    results : dict
        Mapping from template name to AveragedComparisonResult
    metrics_collector : MetricsCollector
        The metrics collector that defines which metrics to plot
    save_dir : str
        Directory to save plots (will be created if needed)
    title_prefix : str
        Prefix for plot titles
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for metric_name in metrics_collector.metric_names():
        config = metrics_collector.get_plot_config(metric_name)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{title_prefix}: {config['display_name']}")

        for (name, result), color in zip(results.items(), colors):
            steps = [m.step for m in result.metrics]

            # Get mean and std for this metric
            mean_attr = f"{metric_name}_mean"
            std_attr = f"{metric_name}_std"

            means = [getattr(m, mean_attr, 0.0) for m in result.metrics]
            stds = [getattr(m, std_attr, 0.0) for m in result.metrics]

            if config.get('use_log_scale', False):
                means = [max(v, 1e-20) for v in means]
                ax.semilogy(steps, means, 'o-', label=name, color=color)
            else:
                ax.plot(steps, means, 'o-', label=name, color=color)
                ax.fill_between(
                    steps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color=color
                )

        ax.set_xlabel('Step')
        ax.set_ylabel(config.get('ylabel', metric_name))

        if config.get('ylim'):
            ax.set_ylim(config['ylim'])

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{metric_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(metrics_collector.metric_names())} averaged metric plots to {save_dir}")


@dataclass
class AveragedMetrics:
    """Averaged metrics across multiple runs."""
    step: int
    template_spread_mean: float
    template_spread_std: float
    volume_proxy_mean: float
    volume_proxy_std: float
    safest_action_prob_mean: float
    safest_action_prob_std: float


@dataclass
class AveragedComparisonResult:
    """Averaged results from multiple comparison runs."""
    template_name: str
    num_runs: int
    num_steps: int
    metrics: List[AveragedMetrics]


def run_multiple_comparisons(
        ipomdp: IPOMDP,
        pp_shield,
        perceptor,
        initial_state,
        templates: Optional[Dict[str, Template]] = None,
        num_runs: int = 10,
        verbose: bool = False
) -> Dict[str, AveragedComparisonResult]:
    """
    Run template comparison multiple times and average results.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    history_generator : callable
        Function that takes ipomdp and returns a history (list of (obs, action) tuples)
    templates : dict, optional
        Templates to compare. If None, creates default templates.
    num_runs : int
        Number of runs to average over
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping from template name to AveragedComparisonResult
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp)

    # Collect results from all runs
    all_results = {name: [] for name in templates}

    for run in range(num_runs):
        if verbose:
            print(f"Run {run + 1}/{num_runs}")

        results = compare_templates(
            ipomdp,
            pp_shield,
            perceptor,
            initial_state,
            templates,
            verbose=False)

        for name, result in results.items():
            all_results[name].append(result)

    # Average the results
    averaged_results = {}
    for name, runs in all_results.items():
        if not runs:
            continue

        # Use maximum length across all runs; average over available data at each step
        num_steps = max(len(r.metrics) for r in runs)
        if num_steps == 0:
            continue

        averaged_metrics = []

        for step in range(num_steps):
            # Only include runs that have data at this step
            spreads = [r.metrics[step].template_spread for r in runs if step < len(r.metrics)]
            volumes = [r.metrics[step].volume_proxy for r in runs if step < len(r.metrics)]
            probs = [r.metrics[step].safest_action_prob for r in runs if step < len(r.metrics)]

            if not spreads:  # Skip if no runs have data at this step
                continue

            averaged_metrics.append(AveragedMetrics(
                step=step,
                template_spread_mean=np.mean(spreads),
                template_spread_std=np.std(spreads),
                volume_proxy_mean=np.mean(volumes),
                volume_proxy_std=np.std(volumes),
                safest_action_prob_mean=np.mean(probs),
                safest_action_prob_std=np.std(probs)
            ))

        averaged_results[name] = AveragedComparisonResult(
            template_name=name,
            num_runs=num_runs,
            num_steps=len(averaged_metrics),
            metrics=averaged_metrics
        )

    return averaged_results


def plot_averaged_comparison(
    results: Dict[str, AveragedComparisonResult],
    title: str = "Template Comparison (Averaged)",
    save_path: Optional[str] = None
):
    """
    Plot averaged comparison with error bands.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: Template spread with std bands
    ax1 = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.template_spread_mean for m in result.metrics]
        stds = [m.template_spread_std for m in result.metrics]
        ax1.plot(steps, means, 'o-', label=name, color=color)
        ax1.fill_between(steps,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color=color)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Template Spread')
    ax1.set_title('Approximation Growth (Template Spread)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume proxy with std bands (log scale)
    ax2 = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [max(m.volume_proxy_mean, 1e-20) for m in result.metrics]
        ax2.semilogy(steps, means, 's-', label=name, color=color)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Volume Proxy (log scale)')
    ax2.set_title('Approximation Growth (Volume Proxy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Safest action probability with std bands
    ax3 = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.safest_action_prob_mean for m in result.metrics]
        stds = [m.safest_action_prob_std for m in result.metrics]
        ax3.plot(steps, means, '^-', label=name, color=color, markersize=8)
        ax3.fill_between(steps,
                         [max(0, m - s) for m, s in zip(means, stds)],
                         [min(1, m + s) for m, s in zip(means, stds)],
                         alpha=0.2, color=color)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Safest Action Probability')
    ax3.set_title('Shield Permissivity (higher = more permissive)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Normalized spread with std bands
    ax4 = axes[1, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.template_spread_mean for m in result.metrics]
        stds = [m.template_spread_std for m in result.metrics]
        # Normalize by first non-zero mean
        initial = next((m for m in means if m > 0), 1.0)
        norm_means = [m / initial for m in means]
        norm_stds = [s / initial for s in stds]
        ax4.plot(steps, norm_means, 'd-', label=name, color=color)
        ax4.fill_between(steps,
                         [m - s for m, s in zip(norm_means, norm_stds)],
                         [m + s for m, s in zip(norm_means, norm_stds)],
                         alpha=0.2, color=color)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Spread')
    ax4.set_title('Relative Approximation Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# ============================================================
# Test Models
# ============================================================

def create_toy_model() -> Tuple[IPOMDP, List[Tuple]]:
    """
    Create a simple 3-state toy IPOMDP for testing.
    """
    states = [0, 1, 2]
    observations = [0, 1, 2]
    actions = [0, 1]

    # Dynamics: action 0 stays, action 1 moves right
    dynamics = {}
    for s in states:
        dynamics[(s, 0)] = {s: 1.0}
        next_s = min(s + 1, 2)
        if next_s == s:
            dynamics[(s, 1)] = {s: 1.0}
        else:
            dynamics[(s, 1)] = {s: 0.2, next_s: 0.8}

    # Interval observation model
    P_low = {
        0: {0: 0.6, 1: 0.1, 2: 0.0},
        1: {0: 0.1, 1: 0.5, 2: 0.1},
        2: {0: 0.0, 1: 0.1, 2: 0.6},
    }
    P_high = {
        0: {0: 0.9, 1: 0.3, 2: 0.1},
        1: {0: 0.3, 1: 0.8, 2: 0.3},
        2: {0: 0.1, 1: 0.3, 2: 0.9},
    }

    ipomdp = IPOMDP(states, observations, actions, dynamics, P_low, P_high)

    # Sample history
    history = [(0, 0), (1, 1), (1, 0), (2, 1), (1, 1), (2, 0)]

    return ipomdp, history



# ============================================================
# Main test runners
# ============================================================

def run_toy_comparison(verbose: bool = True, save_path: Optional[str] = None):
    """Run template comparison on the toy model."""
    print("=" * 60)
    print("TOY MODEL TEMPLATE COMPARISON")
    print("=" * 60)

    ipomdp, history = create_toy_model()
    templates = create_templates_for_ipomdp(ipomdp)

    print(f"\nStates: {ipomdp.states}")
    print(f"Actions: {ipomdp.actions}")
    print(f"History length: {len(history)}")
    print(f"Templates: {list(templates.keys())}")

    results = compare_templates(ipomdp, history, templates, verbose=verbose)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread:.4f}")
        print(f"  Final volume proxy: {final.volume_proxy:.6f}")
        print(f"  Safest action prob: {final.safest_action_prob:.4f}")

    plot_comparison(results, "Toy Model: Template Comparison", save_path)
    return results
 

def run_toy_comparison_averaged(
    num_runs: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = None
):
    """Run averaged template comparison on the toy model."""
    print("=" * 60)
    print(f"TOY MODEL TEMPLATE COMPARISON (AVERAGED, {num_runs} runs)")
    print("=" * 60)

    ipomdp, _ = create_toy_model()
    templates = create_templates_for_ipomdp(ipomdp)

    def history_generator(ipomdp):
        import random
        # Generate random history
        actions = list(ipomdp.actions)
        obs_list = list(ipomdp.observations)
        return [(random.choice(obs_list), random.choice(actions)) for _ in range(6)]

    print(f"\nStates: {ipomdp.states}")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")

    results = run_multiple_comparisons(
        ipomdp, templates, num_runs=num_runs, verbose=verbose
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (AVERAGED)")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread_mean:.4f} +/- {final.template_spread_std:.4f}")
        print(f"  Final volume: {final.volume_proxy_mean:.6f} +/- {final.volume_proxy_std:.6f}")
        print(f"  Safest prob:  {final.safest_action_prob_mean:.4f} +/- {final.safest_action_prob_std:.4f}")

    plot_averaged_comparison(results, "Toy Model: Template Comparison (Averaged)", save_path)
    return results


def run_taxinet_comparison(verbose: bool = True, save_path: Optional[str] = None):
    """Run template comparison on the Taxinet model."""
    print("=" * 60)
    print("TAXINET MODEL TEMPLATE COMPARISON")
    print("=" * 60)

    initial = (0,0), 0
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    templates = create_templates_for_ipomdp(ipomdp)

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")
    
    results = compare_templates(
        ipomdp,
        dyn_shield,
        perception,
        initial,
        templates,
        verbose=verbose)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread:.4f}")
        print(f"  Final volume proxy: {final.volume_proxy:.6e}")
        print(f"  Safest action prob: {final.safest_action_prob:.4f}")

    plot_comparison(results, "Taxinet: Template Comparison", save_path)
    return results


def run_taxinet_comparison_averaged(
    num_runs: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = None
):
    """Run averaged template comparison on the Taxinet model."""
    print("=" * 60)
    print(f"TAXINET MODEL TEMPLATE COMPARISON (AVERAGED, {num_runs} runs)")
    print("=" * 60)

    initial_state = ((0,0), 0)
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    templates = create_templates_for_ipomdp(ipomdp)

    # ipomdp, _ = create_taxinet_model()
    # templates = create_templates_for_ipomdp(ipomdp)

    # # Get non-FAIL states and observations for random history generation
    # states_no_fail = [s for s in ipomdp.states if s != "FAIL"]
    # obs_no_fail = [o for o in ipomdp.observations if o != "FAIL"]

    # def history_generator(ipomdp):
    #     import random
    #     actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)
    #     return [(random.choice(obs_no_fail), random.choice(actions)) for _ in range(6)]

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")

    results = run_multiple_comparisons(
        ipomdp,
        dyn_shield,
        perception,
        initial_state,
        templates,
        verbose=verbose,
        num_runs=num_runs
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (AVERAGED)")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread_mean:.4f} +/- {final.template_spread_std:.4f}")
        print(f"  Final volume: {final.volume_proxy_mean:.6e} +/- {final.volume_proxy_std:.6e}")
        print(f"  Safest prob:  {final.safest_action_prob_mean:.4f} +/- {final.safest_action_prob_std:.4f}")

    plot_averaged_comparison(results, "Taxinet: Template Comparison (Averaged)", save_path)
    return results


def run_all_comparisons(save_dir: Optional[str] = None, num_runs: int = 10):
    """Run all template comparisons (single and averaged)."""
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        toy_path = os.path.join(save_dir, "toy_comparison.png")
        toy_avg_path = os.path.join(save_dir, "toy_comparison_averaged.png")
        taxinet_path = os.path.join(save_dir, "taxinet_comparison.png")
        taxinet_avg_path = os.path.join(save_dir, "taxinet_comparison_averaged.png")
    else:
        toy_path = toy_avg_path = taxinet_path = taxinet_avg_path = None

    all_results = {}

    # print("\n" + "#" * 60)
    # print("# RUNNING TOY MODEL COMPARISON (SINGLE RUN)")
    # print("#" * 60 + "\n")
    # all_results["toy_single"] = run_toy_comparison(verbose=True, save_path=toy_path)

    # print("\n" + "#" * 60)
    # print(f"# RUNNING TOY MODEL COMPARISON (AVERAGED, {num_runs} runs)")
    # print("#" * 60 + "\n")
    # all_results["toy_averaged"] = run_toy_comparison_averaged(
    #     num_runs=num_runs, verbose=True, save_path=toy_avg_path
    # )

    print("\n" + "#" * 60)
    print(f"# RUNNING TAXINET SCRIPTED COMPARISON ({num_runs} scripts)")
    print("#" * 60 + "\n")

    taxinet_scripted_dir = os.path.join(save_dir, "taxinet_scripted") if save_dir else None
    results, library = run_taxinet_scripted_comparison(
        num_scripts=num_runs,
        script_length=20,
        verbose=True,
        save_dir=taxinet_scripted_dir
    )
    all_results["taxinet_scripted"] = results

    return all_results


# ============================================================
# Ground Truth Comparison (Monte Carlo)
# ============================================================

@dataclass
class GroundTruthComparisonResult:
    """Result of comparing LFP predictions against Monte Carlo ground truth.

    Aggregates predictions from many scripted runs and compares against
    empirical frequencies computed from true states.
    """
    template_name: str
    num_runs: int
    # Per step, per action: predicted min P(safe)
    action_safety_predicted: List[Dict]  # List[Dict[action, float]]
    # Per step, per action: empirical fraction of runs where true state is safe
    action_safety_empirical: List[Dict]  # List[Dict[action, float]]
    # Per step, per state: predicted [lower, upper] bounds (averaged across runs)
    state_bounds_predicted: List[Dict[int, Tuple[float, float]]]
    # Per step, per state: empirical fraction of runs in that state
    state_occupancy_empirical: List[Dict[int, float]]
    # Summary metrics
    action_coverage_rate: float  # fraction of (step,action) pairs where empirical >= predicted_min
    mean_conservatism_gap: float  # average (empirical - predicted_min)


def compute_ground_truth_comparison(
    ipomdp: IPOMDP,
    pp_shield: Dict,
    library: ScriptLibrary,
    templates: Optional[Dict[str, Template]] = None,
    action_shield: float = 0.8,
    verbose: bool = False
) -> Dict[str, GroundTruthComparisonResult]:
    """Compare LFP predictions against Monte Carlo empirical ground truth.

    Runs each script through each template using GroundTruthComparisonMetrics,
    then aggregates predictions and compares against empirical frequencies
    from the true states in the scripts.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    library : ScriptLibrary
        Library of pre-recorded scripts (provides Monte Carlo samples)
    templates : dict, optional
        Templates to compare. If None, creates default templates.
    action_shield : float
        Minimum required probability for an action to be allowed
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping from template name to GroundTruthComparisonResult
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp, include_safe_sets=False)

    states = list(ipomdp.states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Build inv_shield for empirical safety checks
    inv_shield = {
        a: set(i for i, s in enumerate(states) if a in pp_shield.get(s, set()))
        for a in ipomdp.actions
    }

    gt_results = {}

    for template_name, template in templates.items():
        if verbose:
            print(f"\nTemplate: {template_name}")

        # Collect results from all scripts
        all_run_results: List[ComparisonResult] = []

        for i, script in enumerate(library):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Script {i + 1}/{len(library)}")

            collector = GroundTruthComparisonMetrics()
            result = run_scripted_lfp_propagation(
                ipomdp, pp_shield, template, script,
                action_shield, metrics_collector=collector
            )
            all_run_results.append(result)

        # Determine max number of steps across runs
        num_steps = max(
            len(r.true_states) for r in all_run_results
            if r.true_states is not None
        )

        # Aggregate per-step predictions and empirical data
        action_safety_predicted = []
        action_safety_empirical = []
        state_bounds_predicted = []
        state_occupancy_empirical = []

        coverage_checks = 0
        coverage_hits = 0
        conservatism_gaps = []

        actions = list(ipomdp.actions)
        n = len(states)

        for step in range(num_steps):
            # Collect predictions and true states across runs at this step
            step_action_preds = {a: [] for a in actions}
            step_state_bounds_lower = {i: [] for i in range(n)}
            step_state_bounds_upper = {i: [] for i in range(n)}
            step_true_states = []

            for result in all_run_results:
                if result.true_states is None or step >= len(result.true_states):
                    continue
                if result.action_predictions is None or step >= len(result.action_predictions):
                    continue
                if result.state_bound_predictions is None or step >= len(result.state_bound_predictions):
                    continue

                true_state = result.true_states[step]
                step_true_states.append(true_state)

                # Collect action predictions
                for a in actions:
                    if a in result.action_predictions[step]:
                        step_action_preds[a].append(result.action_predictions[step][a])

                # Collect state bound predictions
                for i in range(n):
                    if i in result.state_bound_predictions[step]:
                        lb, ub = result.state_bound_predictions[step][i]
                        step_state_bounds_lower[i].append(lb)
                        step_state_bounds_upper[i].append(ub)

            if not step_true_states:
                continue

            num_runs_at_step = len(step_true_states)

            # Compute empirical action safety: fraction of runs where true state is safe for action
            empirical_action_safety = {}
            for a in actions:
                safe_count = sum(
                    1 for s in step_true_states
                    if s != "FAIL" and state_to_idx.get(s, -1) in inv_shield[a]
                )
                empirical_action_safety[a] = safe_count / num_runs_at_step

            # Compute predicted action safety: mean of predictions across runs
            predicted_action_safety = {}
            for a in actions:
                if step_action_preds[a]:
                    predicted_action_safety[a] = float(np.mean(step_action_preds[a]))
                else:
                    predicted_action_safety[a] = 0.0

            # Compute empirical state occupancy
            empirical_state_occ = {}
            for i in range(n):
                occ_count = sum(
                    1 for s in step_true_states
                    if s != "FAIL" and state_to_idx.get(s, -1) == i
                )
                empirical_state_occ[i] = occ_count / num_runs_at_step

            # Compute predicted state bounds: mean of lower/upper across runs
            predicted_state_bounds = {}
            for i in range(n):
                if step_state_bounds_lower[i] and step_state_bounds_upper[i]:
                    mean_lb = float(np.mean(step_state_bounds_lower[i]))
                    mean_ub = float(np.mean(step_state_bounds_upper[i]))
                    predicted_state_bounds[i] = (mean_lb, mean_ub)
                else:
                    predicted_state_bounds[i] = (0.0, 1.0)

            action_safety_predicted.append(predicted_action_safety)
            action_safety_empirical.append(empirical_action_safety)
            state_bounds_predicted.append(predicted_state_bounds)
            state_occupancy_empirical.append(empirical_state_occ)

            # Coverage and conservatism for action safety
            for a in actions:
                pred_min = predicted_action_safety[a]
                emp = empirical_action_safety[a]
                coverage_checks += 1
                if emp >= pred_min - 1e-9:  # small tolerance for numerical errors
                    coverage_hits += 1
                conservatism_gaps.append(emp - pred_min)

        action_coverage_rate = coverage_hits / coverage_checks if coverage_checks > 0 else 0.0
        mean_conservatism_gap = float(np.mean(conservatism_gaps)) if conservatism_gaps else 0.0

        gt_results[template_name] = GroundTruthComparisonResult(
            template_name=template_name,
            num_runs=len(all_run_results),
            action_safety_predicted=action_safety_predicted,
            action_safety_empirical=action_safety_empirical,
            state_bounds_predicted=state_bounds_predicted,
            state_occupancy_empirical=state_occupancy_empirical,
            action_coverage_rate=action_coverage_rate,
            mean_conservatism_gap=mean_conservatism_gap,
        )

        if verbose:
            print(f"  Coverage rate: {action_coverage_rate:.3f}")
            print(f"  Mean conservatism gap: {mean_conservatism_gap:.4f}")

    return gt_results


def plot_ground_truth_comparison(
    results: Dict[str, GroundTruthComparisonResult],
    save_dir: Optional[str] = None,
    title_prefix: str = "Ground Truth Comparison",
    top_k_states: int = 5
):
    """Plot ground truth comparison results.

    Creates three types of plots:
    1. Action safety: predicted min P(safe) vs empirical P(safe) per action
    2. State occupancy: predicted bounds vs empirical frequency per state
    3. Summary: conservatism gap and coverage over time

    Parameters
    ----------
    results : dict
        Mapping from template name to GroundTruthComparisonResult
    save_dir : str, optional
        Directory to save plots. If None, displays interactively.
    title_prefix : str
        Prefix for plot titles
    top_k_states : int
        Number of most-occupied states to plot in state occupancy plot
    """
    import matplotlib.pyplot as plt

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for template_name, result in results.items():
        num_steps = len(result.action_safety_predicted)
        if num_steps == 0:
            continue

        steps = list(range(num_steps))
        actions = list(result.action_safety_predicted[0].keys()) if result.action_safety_predicted else []

        # --- Plot 1: Action Safety ---
        n_actions = len(actions)
        if n_actions > 0:
            fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 4), squeeze=False)
            fig.suptitle(f"{title_prefix}: Action Safety ({template_name})", fontsize=12)

            for j, action in enumerate(actions):
                ax = axes[0, j]
                predicted = [result.action_safety_predicted[t].get(action, 0.0) for t in range(num_steps)]
                empirical = [result.action_safety_empirical[t].get(action, 0.0) for t in range(num_steps)]

                ax.plot(steps, predicted, 'o-', color=colors[0], label='Predicted min P(safe)', markersize=4)
                ax.plot(steps, empirical, 's--', color=colors[1], label='Empirical P(safe)', markersize=4)
                ax.fill_between(steps, predicted, empirical, alpha=0.15, color=colors[2],
                                label='Conservatism gap')
                ax.set_xlabel('Step')
                ax.set_ylabel('P(safe)')
                ax.set_title(f'Action: {action}')
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_dir:
                path = os.path.join(save_dir, f"action_safety_{template_name}.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

        # --- Plot 2: State Occupancy ---
        # Find top-k most occupied states across all steps
        n_states = len(result.state_occupancy_empirical[0]) if result.state_occupancy_empirical else 0
        if n_states > 0:
            total_occ = np.zeros(n_states)
            for step_occ in result.state_occupancy_empirical:
                for i, freq in step_occ.items():
                    total_occ[i] += freq
            top_states = np.argsort(total_occ)[-top_k_states:][::-1]
            # Filter to states with nonzero occupancy
            top_states = [s for s in top_states if total_occ[s] > 0]

            if top_states:
                n_plot = len(top_states)
                fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), squeeze=False)
                fig.suptitle(f"{title_prefix}: State Occupancy ({template_name})", fontsize=12)

                for j, state_idx in enumerate(top_states):
                    ax = axes[0, j]
                    # Predicted bounds
                    lowers = [result.state_bounds_predicted[t].get(state_idx, (0, 1))[0] for t in range(num_steps)]
                    uppers = [result.state_bounds_predicted[t].get(state_idx, (0, 1))[1] for t in range(num_steps)]
                    empirical = [result.state_occupancy_empirical[t].get(state_idx, 0.0) for t in range(num_steps)]

                    ax.fill_between(steps, lowers, uppers, alpha=0.25, color=colors[0],
                                    label='Predicted bounds')
                    ax.plot(steps, empirical, 'o-', color=colors[1], label='Empirical freq', markersize=4)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Probability')
                    ax.set_title(f'State {state_idx}')
                    ax.set_ylim(-0.05, 1.05)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                if save_dir:
                    path = os.path.join(save_dir, f"state_occupancy_{template_name}.png")
                    plt.savefig(path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()

        # --- Plot 3: Summary (conservatism gap + coverage over time) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(f"{title_prefix}: Summary ({template_name})", fontsize=12)

        # Conservatism gap per step (averaged over actions)
        gap_per_step = []
        coverage_per_step = []
        for t in range(num_steps):
            step_gaps = []
            step_covered = 0
            step_total = 0
            for a in actions:
                pred = result.action_safety_predicted[t].get(a, 0.0)
                emp = result.action_safety_empirical[t].get(a, 0.0)
                step_gaps.append(emp - pred)
                step_total += 1
                if emp >= pred - 1e-9:
                    step_covered += 1
            gap_per_step.append(np.mean(step_gaps) if step_gaps else 0.0)
            coverage_per_step.append(step_covered / step_total if step_total > 0 else 0.0)

        ax1.plot(steps, gap_per_step, 'o-', color=colors[0], markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Mean Conservatism Gap')
        ax1.set_title('Conservatism Gap (empirical - predicted min)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, coverage_per_step, 's-', color=colors[2], markersize=4)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Coverage Rate')
        ax2.set_title('Coverage (frac. where empirical >= predicted min)')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"summary_{template_name}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    if save_dir:
        print(f"Saved ground truth comparison plots to {save_dir}")


def run_taxinet_ground_truth_comparison(
    num_scripts: int = 20,
    script_length: int = 20,
    seed: Optional[int] = 42,
    verbose: bool = True,
    save_dir: Optional[str] = None
):
    """Run ground truth comparison on the Taxinet model.

    Generates a library of Monte Carlo scripts, runs LFP propagation with
    GroundTruthComparisonMetrics on each, and compares predictions against
    empirical frequencies from the true states.

    Parameters
    ----------
    num_scripts : int
        Number of Monte Carlo scripts to generate
    script_length : int
        Length of each script
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress
    save_dir : str, optional
        Directory to save plots

    Returns
    -------
    tuple
        (results dict, script library)
    """
    print("=" * 60)
    print(f"TAXINET GROUND TRUTH COMPARISON ({num_scripts} scripts, length {script_length})")
    print("=" * 60)

    initial = ((0, 0), 0)
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    # Generate script library
    if verbose:
        print(f"\nGenerating {num_scripts} scripts...")

    library = ScriptLibrary.generate(
        ipomdp, dyn_shield, perception, initial,
        num_scripts, script_length, seed=seed
    )

    if verbose:
        print(f"Generated {len(library)} scripts")
        avg_len = np.mean([s.length for s in library])
        print(f"Average script length: {avg_len:.1f}")

    templates = create_templates_for_ipomdp(ipomdp, include_safe_sets=False)

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {list(ipomdp.actions)}")
    print(f"Templates: {list(templates.keys())}")

    results = compute_ground_truth_comparison(
        ipomdp, dyn_shield, library, templates,
        action_shield=0.8, verbose=verbose
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Num runs: {result.num_runs}")
        print(f"  Action coverage rate: {result.action_coverage_rate:.4f}")
        print(f"  Mean conservatism gap: {result.mean_conservatism_gap:.4f}")

    # Generate plots
    if save_dir:
        plot_ground_truth_comparison(results, save_dir=save_dir)

    return results, library


if __name__ == "__main__":
    # run_all_comparisons(save_dir="images",num_runs=3)

    run_taxinet_ground_truth_comparison(
        num_scripts= 20,
        script_length = 20,
        seed = 42,
        verbose = True,
        save_dir = "images/ground_truth"
    )
