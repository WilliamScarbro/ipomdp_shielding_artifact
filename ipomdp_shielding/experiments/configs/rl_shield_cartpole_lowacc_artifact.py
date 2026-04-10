"""Artifact CartPole low-accuracy config."""

from pathlib import Path

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp


_DATA_DIR = Path(__file__).parent.parent.parent / "CaseStudies" / "CartPole" / "artifacts_lowacc"

config = RLShieldExperimentConfig(
    case_study_name="cartpole_lowacc",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=15,
    rl_episodes=500,
    rl_episode_length=15,
    opt_candidates=8,
    opt_trials_per_candidate=5,
    opt_iterations=8,
    shield_threshold=0.8,
    rl_cache_path="results/cache/lowacc_rl_shield_cartpole_agent.pt",
    opt_cache_path="results/cache/lowacc_rl_shield_cartpole_opt_realization.json",
    results_path="results/experiment/threshold/cartpole_lowacc_sweep.json",
    figures_dir="results/experiment/cartpole_lowacc_figures",
    adversarial_opt_targets=["single_belief"],
    ipomdp_kwargs={"num_bins": 3, "data_dir": _DATA_DIR},
)
