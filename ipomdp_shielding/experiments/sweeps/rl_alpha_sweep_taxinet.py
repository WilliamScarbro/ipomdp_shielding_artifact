"""Default alpha-beta sweep config for TaxiNet.

Expanded version: denser alpha grid to expose the alpha trend, evaluated at
three shield thresholds (beta ∈ {0.7, 0.8, 0.9}).
"""

from .rl_alpha_sweep import AlphaSweepConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp

config = AlphaSweepConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    alphas=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
    betas=[0.7, 0.8, 0.9],
    seeds=[42, 123, 456, 789, 1024],
    num_trials=20,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=5,
    shields=["single_belief", "envelope", "forward_sampling"],
    perceptions=["uniform", "adversarial_opt"],
    results_dir="./data/sweep/rl_alpha_taxinet_v2",
    ipomdp_base_kwargs={
        "confidence_method": "Clopper_Pearson",
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    },
)
