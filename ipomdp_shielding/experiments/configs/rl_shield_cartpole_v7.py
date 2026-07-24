"""V7 CartPole (standard) config: adversarial realization trained against RL selector.

Uses single_belief as adversarial_opt_target (envelope excluded for CartPole —
too slow and Pareto-dominated by single_belief at 82-state/82-obs scale).
"""
import dataclasses
from .rl_shield_cartpole_final import config as _base

config = dataclasses.replace(
    _base,
    opt_cache_path="results/cache/v7_rl_shield_cartpole3_sb_opt_realization.json",
    adversarial_opt_targets=["single_belief"],
)
