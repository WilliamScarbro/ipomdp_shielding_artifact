"""Artifact Obstacle config."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=25,
    rl_episodes=500,
    rl_episode_length=25,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_obstacle_agent.pt",
    opt_cache_path="results/cache/rl_shield_obstacle_opt_realization.json",
    results_path="results/experiment/threshold/obstacle_sweep.json",
    figures_dir="results/experiment/obstacle_figures",
)
