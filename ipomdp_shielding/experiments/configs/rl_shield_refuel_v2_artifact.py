"""Artifact Refuel v2 config."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="refuel_v2",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=30,
    rl_episodes=500,
    rl_episode_length=30,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/v2_rl_shield_refuel_agent.pt",
    opt_cache_path="results/cache/rl_shield_refuel_v2_sb_opt_realization.json",
    results_path="results/experiment/threshold/refuel_v2_sweep.json",
    figures_dir="results/experiment/refuel_v2_figures",
    adversarial_opt_targets=["single_belief"],
)
