"""Final RL shielding experiment configuration for TaxiNet.

10x more trials, 2x longer vs prelim.
Reuses prelim RL agent and opt-realization caches (no retraining cost).
Estimated runtime: ~20 min (6 envelope combos × 100 × 20 × 0.095 s/step).
"""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trials=100,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    # Reuse prelim caches — no retraining needed
    rl_cache_path="results/cache/prelim_rl_shield_taxinet_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_taxinet_opt_realization.json",
    results_path="results/final/rl_shield_taxinet_results.json",
    figures_dir="results/final/rl_shield_taxinet_figures",
)
