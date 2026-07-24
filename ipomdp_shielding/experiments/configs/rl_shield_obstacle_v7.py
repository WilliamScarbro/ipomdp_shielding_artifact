"""V7 Obstacle config: adversarial realization trained against RL selector."""
import dataclasses
from .rl_shield_obstacle_final import config as _base

config = dataclasses.replace(
    _base,
    opt_cache_path="results/cache/v7_rl_shield_obstacle_opt_realization.json",
)
