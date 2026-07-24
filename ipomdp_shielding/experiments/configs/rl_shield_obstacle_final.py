"""Final RL shielding experiment configuration for Obstacle.

5x more trials, 1.25x longer vs prelim.
Reuses prelim RL agent and opt-realization caches.
Estimated runtime: ~50–85 min (6 envelope combos × 50 × 25 × 0.679 s/step,
moderated by frequent early-stuck termination).
"""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trials=50,
    trial_length=25,
    rl_episodes=500,
    rl_episode_length=25,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    # Reuse prelim caches — no retraining needed
    rl_cache_path="results/cache/prelim_rl_shield_obstacle_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_obstacle_opt_realization.json",
    results_path="results/final/rl_shield_obstacle_results.json",
    figures_dir="results/final/rl_shield_obstacle_figures",
)
