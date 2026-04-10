"""Visualization functions for Monte Carlo safety evaluation results."""

from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .data_structures import MCSafetyMetrics, TimestepMetrics


def plot_safety_metrics(
    metrics_by_mode: Dict[str, MCSafetyMetrics],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Visualize safety metrics across sampling modes.

    Creates a 3-panel figure:
    1. Bar chart: fail/stuck/safe rates per sampling mode
    2. Histogram: failure step distribution across modes
    3. Bar chart: mean trajectory length per mode

    Parameters
    ----------
    metrics_by_mode : dict
        Mapping from sampling mode to MCSafetyMetrics
    save_path : str, optional
        Path to save figure (e.g., "images/safety_comparison.png")
    show : bool
        Whether to display the figure
    """
    if not metrics_by_mode:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    modes = list(metrics_by_mode.keys())

    # Panel 1: Outcome rates
    ax = axes[0]
    x = np.arange(len(modes))
    width = 0.25

    fail_rates = [metrics_by_mode[m].fail_rate for m in modes]
    stuck_rates = [metrics_by_mode[m].stuck_rate for m in modes]
    safe_rates = [metrics_by_mode[m].safe_rate for m in modes]

    ax.bar(x - width, fail_rates, width, label='Fail', color='red', alpha=0.7)
    ax.bar(x, stuck_rates, width, label='Stuck', color='orange', alpha=0.7)
    ax.bar(x + width, safe_rates, width, label='Safe', color='green', alpha=0.7)

    ax.set_xlabel('Sampling Mode')
    ax.set_ylabel('Rate')
    ax.set_title('Safety Outcome Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Failure step distribution
    ax = axes[1]
    for mode in modes:
        fail_steps = metrics_by_mode[mode].fail_step_distribution
        if fail_steps:
            ax.hist(fail_steps, bins=20, alpha=0.5, label=mode)

    ax.set_xlabel('Step')
    ax.set_ylabel('Failure Count')
    ax.set_title('Failure Step Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Mean trajectory length
    ax = axes[2]
    mean_steps = [metrics_by_mode[m].mean_steps for m in modes]

    ax.bar(modes, mean_steps, color='steelblue', alpha=0.7)
    ax.set_xlabel('Sampling Mode')
    ax.set_ylabel('Mean Steps')
    ax.set_title('Mean Trajectory Length')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_rl_training_curves(
    training_metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot RL training curves for best and worst selectors.

    Parameters
    ----------
    training_metrics : dict
        Training metrics from evaluate_with_trained_rl
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Episode rewards over training
    ax = axes[0]
    for selector_name, metrics in training_metrics.items():
        rewards = metrics["episode_rewards"]
        # Smooth with moving average
        window = min(50, len(rewards) // 5)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(len(smoothed)), smoothed, label=selector_name)
        else:
            ax.plot(rewards, label=selector_name)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Training Reward (smoothed)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Safe rate over training (in windows)
    ax = axes[1]
    window = 50
    for selector_name, metrics in training_metrics.items():
        outcomes = metrics["episode_outcomes"]
        safe_rates = []
        for i in range(0, len(outcomes) - window + 1, window // 2):
            chunk = outcomes[i:i+window]
            safe_rate = sum(1 for o in chunk if o == "safe") / len(chunk)
            safe_rates.append(safe_rate)

        x = np.arange(len(safe_rates)) * (window // 2)
        ax.plot(x, safe_rates, label=selector_name, marker='o', markersize=3)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Safe Rate')
    ax.set_title(f'Training Safe Rate (window={window})')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_two_player_game_results(
    results: Dict[str, Dict[str, MCSafetyMetrics]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Visualize 2-player game results comparing nature strategies.

    Parameters
    ----------
    results : dict
        Nested dict from evaluate_two_player_game
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    nature_strategies = list(results.keys())
    agent_modes = list(results[nature_strategies[0]].keys())

    x = np.arange(len(agent_modes))
    width = 0.35

    # Panel 1: Failure rates by nature strategy
    ax = axes[0]
    for i, nature in enumerate(nature_strategies):
        fail_rates = [results[nature][m].fail_rate for m in agent_modes]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, fail_rates, width, label=f"Nature: {nature}")
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)

    ax.set_xlabel('Agent Strategy (Action Selection)')
    ax.set_ylabel('Failure Rate')
    ax.set_title('Failure Rate: Cooperative vs Adversarial Nature')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_modes)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Safe rates comparison
    ax = axes[1]
    for i, nature in enumerate(nature_strategies):
        safe_rates = [results[nature][m].safe_rate for m in agent_modes]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, safe_rates, width, label=f"Nature: {nature}")
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)

    ax.set_xlabel('Agent Strategy (Action Selection)')
    ax.set_ylabel('Safe Rate')
    ax.set_title('Safe Rate: Cooperative vs Adversarial Nature')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_modes)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_timestep_evolution(
    timestep_metrics: TimestepMetrics,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Cumulative Safety Outcomes Over Time"
):
    """Plot cumulative fail/stuck/safe probabilities over timesteps.

    Shows how the fraction of trials in each outcome state evolves
    as trials progress through timesteps.

    Parameters
    ----------
    timestep_metrics : TimestepMetrics
        Aggregated timestep metrics with cumulative probabilities
    save_path : str, optional
        Path to save figure (e.g., "images/timestep_evolution.png")
    show : bool
        Whether to display the figure
    title : str
        Plot title
    """
    if timestep_metrics.num_trials == 0:
        print("No timestep data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = list(range(timestep_metrics.trial_length))

    # Plot cumulative probabilities
    ax.plot(timesteps, timestep_metrics.fail_prob_by_timestep,
            'r-', label='P(Failed by t)', linewidth=2)
    ax.plot(timesteps, timestep_metrics.stuck_prob_by_timestep,
            color='orange', label='P(Stuck by t)', linewidth=2)
    ax.plot(timesteps, timestep_metrics.safe_prob_by_timestep,
            'g-', label='P(Still Safe at t)', linewidth=2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{title}\n(n={timestep_metrics.num_trials} trials)')
    ax.legend(loc='center right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, timestep_metrics.trial_length - 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timestep evolution figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_timestep_comparison(
    metrics_dict: Dict[str, TimestepMetrics],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot timestep evolution for multiple experiment configurations.

    Creates a 3-panel figure comparing fail/stuck/safe across configurations.

    Parameters
    ----------
    metrics_dict : dict
        Mapping from config name to TimestepMetrics
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    """
    if not metrics_dict:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    config_names = list(metrics_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))

    # Panel 1: Fail probabilities
    ax = axes[0]
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        timesteps = list(range(metrics.trial_length))
        ax.plot(timesteps, metrics.fail_prob_by_timestep,
                color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('P(Failed by t)')
    ax.set_title('Cumulative Failure Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Panel 2: Stuck probabilities
    ax = axes[1]
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        timesteps = list(range(metrics.trial_length))
        ax.plot(timesteps, metrics.stuck_prob_by_timestep,
                color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('P(Stuck by t)')
    ax.set_title('Cumulative Stuck Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Panel 3: Safe probabilities
    ax = axes[2]
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        timesteps = list(range(metrics.trial_length))
        ax.plot(timesteps, metrics.safe_prob_by_timestep,
                color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('P(Still Safe at t)')
    ax.set_title('Survival Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timestep comparison figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
