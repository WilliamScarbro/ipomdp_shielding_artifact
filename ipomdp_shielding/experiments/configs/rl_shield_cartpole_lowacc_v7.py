"""V7 CartPole low-accuracy config: adversarial realization trained against RL selector."""
import dataclasses
from .rl_shield_cartpole_lowacc import config as _base

config = dataclasses.replace(
    _base,
    opt_cache_path="results/cache/v7_lowacc_rl_shield_cartpole_opt_realization.json",
)
