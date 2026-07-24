"""V7 TaxiNet config: adversarial realization trained against RL selector."""
import dataclasses
from .rl_shield_taxinet_final import config as _base

config = dataclasses.replace(
    _base,
    ipomdp_kwargs={**_base.ipomdp_kwargs, "error": 0.01},
    rl_cache_path="results/cache/v7_err001_rl_shield_taxinet_agent.pt",
    opt_cache_path="results/cache/v7_err001_rl_shield_taxinet_opt_realization.json",
)
