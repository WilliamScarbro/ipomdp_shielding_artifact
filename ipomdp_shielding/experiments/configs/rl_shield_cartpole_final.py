"""Final RL shielding experiment configuration for CartPole (3-bin, 82 states).

2.5x more trials, 1.5x longer vs prelim.  Uses 3-bin discretisation
consistent with the coarse and prelim RL experiments.
Reuses prelim RL agent and opt-realization caches.
Estimated runtime: ~50–70 min (6 envelope combos × 25 × 15 × 1.876 s/step).
"""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=25,
    trial_length=15,
    rl_episodes=300,
    rl_episode_length=15,
    opt_candidates=8,
    opt_trials_per_candidate=5,
    opt_iterations=8,
    shield_threshold=0.8,
    # Reuse prelim caches — no retraining needed
    rl_cache_path="results/cache/prelim_rl_shield_cartpole3_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_cartpole3_opt_realization.json",
    results_path="results/final/rl_shield_cartpole_results.json",
    figures_dir="results/final/rl_shield_cartpole_figures",
    ipomdp_kwargs={"num_bins": 3},
)
