"""RL shielding experiment config for Refuel v2.

Changes from v1/final:
  - obs_noise raised from 0.05 → 0.3: gives ~10-15% RL fail rate without shielding.
  - hascrash (old bit 5) removed from observations: obstacle position not directly visible.
  - fuel > 0 (old bit 7) removed: agent cannot directly observe fuel exhaustion.

These two changes make the safety-critical predicates unobservable; the agent must rely
on belief tracking (and shielding) to avoid crashes and fuel exhaustion.

Observation space: 8-element tuple (was 10).  Old RL agent cache is incompatible
— a fresh agent is trained and cached at results/cache/v2_rl_shield_refuel_agent.pt.

Envelope excluded: LP solve ~144 s/step for 344 states remains infeasible.
Adversarial perception optimised against single-belief shield.
Estimated runtime: ~20 min (18 non-envelope combos × 50 trials × 30 steps,
plus RL training ~10 min and adversarial opt ~5 min).
"""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="refuel_v2",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trials=50,
    trial_length=30,
    rl_episodes=500,
    rl_episode_length=30,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    adversarial_opt_targets=["single_belief"],
    rl_cache_path="results/cache/v2_rl_shield_refuel_agent.pt",
    opt_cache_path="results/cache/v2_rl_shield_refuel_sb_opt_realization.json",
    results_path="results/v2/rl_shield_refuel_v2_results.json",
    figures_dir="results/v2/rl_shield_refuel_v2_figures",
)
