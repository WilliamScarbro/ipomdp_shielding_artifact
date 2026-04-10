"""Artifact TaxiNet config."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_taxinet_agent.pt",
    opt_cache_path="results/cache/rl_shield_taxinet_opt_realization.json",
    results_path="results/experiment/threshold/taxinet_sweep.json",
    figures_dir="results/experiment/taxinet_figures",
)
