"""TaxiNetV2 comparison experiment config (conf=0.995)."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.TaxiNetV2 import build_taxinet_v2_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="taxinet_v2_comparison_conf995",
    build_ipomdp_fn=build_taxinet_v2_ipomdp,
    seed=42,
    num_trials=200,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/rl_shield_taxinet_v2_agent.pt",
    opt_cache_path="results/cache/rl_shield_taxinet_v2_comparison_point_opt_realization.json",
    results_path="results/taxinet_v2/taxinet_v2_comparison_conf995_results.json",
    figures_dir="results/taxinet_v2/taxinet_v2_comparison_conf995_figures",
    ipomdp_kwargs={
        "confidence_method": "Clopper_Pearson",
        "alpha": 0.05,
        "confidence_level": "0.995",
        "error": 0.1,
        "smoothing": True,
    },
)
