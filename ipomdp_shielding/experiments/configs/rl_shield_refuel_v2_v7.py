"""V7 Refuel v2 config: adversarial realization trained against RL selector."""
import dataclasses
from .rl_shield_refuel_v2 import config as _base

config = dataclasses.replace(
    _base,
    opt_cache_path="results/cache/v7_rl_shield_refuel_v2_sb_opt_realization.json",
)
