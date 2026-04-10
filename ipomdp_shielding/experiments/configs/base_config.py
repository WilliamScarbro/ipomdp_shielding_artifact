"""Base configuration classes for experiments."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CoarseExperimentConfig:
    """Configuration for coarseness experiments."""

    # Case study
    case_study_name: str
    build_ipomdp_fn: Callable

    # Experiment parameters
    seed: int
    num_trajectories: int
    trajectory_length: int
    initial_state: Any
    initial_action: int

    # Forward-sampled belief parameters
    sampler_budget: int
    sampler_k: int
    sampler_likelihood_strategy: str
    sampler_pruning_strategy: str

    # Output
    results_path: str

    # Optional build_ipomdp kwargs
    ipomdp_kwargs: dict = None

    def __post_init__(self):
        if self.ipomdp_kwargs is None:
            self.ipomdp_kwargs = {}


@dataclass
class RLShieldExperimentConfig:
    """Configuration for RL shielding experiments."""

    # Case study
    case_study_name: str
    build_ipomdp_fn: Callable

    # Experiment parameters
    seed: int
    num_trials: int
    trial_length: int

    # RL training
    rl_episodes: int
    rl_episode_length: int

    # Optimized realization training
    opt_candidates: int
    opt_trials_per_candidate: int
    opt_iterations: int

    # Shield threshold for belief-based shields
    shield_threshold: float

    # Cache and output paths
    rl_cache_path: str
    opt_cache_path: str
    results_path: str
    figures_dir: str

    # Which shield(s) the adversarial fixed realization is optimized against.
    # Default preserves legacy behavior: optimize against the envelope shield.
    adversarial_opt_targets: list[str] = field(default_factory=lambda: ["envelope"])

    # Optional build_ipomdp kwargs
    ipomdp_kwargs: dict = None

    def __post_init__(self):
        if self.ipomdp_kwargs is None:
            self.ipomdp_kwargs = {}
