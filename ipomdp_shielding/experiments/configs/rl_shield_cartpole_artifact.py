"""Artifact CartPole config."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=15,
    rl_episodes=300,
    rl_episode_length=15,
    opt_candidates=8,
    opt_trials_per_candidate=5,
    opt_iterations=8,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_cartpole3_agent.pt",
    opt_cache_path="results/cache/rl_shield_cartpole3_sb_opt_realization.json",
    results_path="results/experiment/threshold/cartpole_sweep.json",
    figures_dir="results/experiment/cartpole_figures",
    adversarial_opt_targets=["single_belief"],
    ipomdp_kwargs={"num_bins": 3},
)
