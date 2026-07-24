"""Generate v7 evaluation summary.

Bug fix from v6: adversarial perception realizations trained against RL selector
(previously RandomActionSelector — incorrect, defeats the purpose of adversarial
evaluation).  All adversarial-perception data regenerated; reads from results/sweep_v7/.

Usage:
    python -m ipomdp_shielding.experiments.plot_sweep_v7
"""

from __future__ import annotations
from pathlib import Path

import ipomdp_shielding.experiments.plot_sweep_v5 as _v5

_V7_BASE = Path("results/sweep_v7")
_V7_DATA = _V7_BASE / "threshold"
_V7_OBS  = _V7_BASE / "obs"
_V7_FS   = _V7_BASE / "fs"
_V7_CARR = _V7_BASE / "carr"

_v5.VERSION = "v7"
_v5.OUTDIR  = _V7_BASE
_v5.MD_PATH = _v5.OUTDIR / "evaluation_summary.md"
_v5.DATA    = _V7_DATA
_v5.OBS     = _V7_OBS
_v5.FS      = _V7_FS

# Re-point all CASES paths to v7 directories.
# (CASES is a module-level dict built at import time; mutate in-place.)
_v5.CASES["taxinet"]["sweep"] = _V7_DATA / "taxinet_sweep.json"
_v5.CASES["taxinet"]["carr"]  = _V7_CARR / "taxinet_carr_results.json"
_v5.CASES["taxinet"]["obs"]   = _V7_OBS  / "taxinet_obs_sweep.json"
_v5.CASES["taxinet"]["fs"]    = _V7_FS   / "taxinet_fs_sweep.json"

_v5.CASES["obstacle"]["sweep"] = _V7_DATA / "obstacle_sweep.json"
_v5.CASES["obstacle"]["carr"]  = _V7_CARR / "obstacle_carr_results.json"
_v5.CASES["obstacle"]["obs"]   = _V7_OBS  / "obstacle_obs_sweep.json"
_v5.CASES["obstacle"]["fs"]    = _V7_FS   / "obstacle_fs_sweep.json"

_v5.CASES["cartpole_lowacc"]["sweep"] = _V7_DATA / "cartpole_lowacc_sweep.json"
_v5.CASES["cartpole_lowacc"]["carr"]  = _V7_CARR / "cartpole_lowacc_carr_results.json"
_v5.CASES["cartpole_lowacc"]["obs"]   = _V7_OBS  / "cartpole_lowacc_obs_sweep.json"
_v5.CASES["cartpole_lowacc"]["fs"]    = _V7_FS   / "cartpole_lowacc_fs_sweep.json"

_v5.CASES["refuel_v2"]["sweep"] = _V7_DATA / "refuel_v2_sweep.json"
_v5.CASES["refuel_v2"]["carr"]  = None   # support-MDP BFS infeasible
_v5.CASES["refuel_v2"]["obs"]   = _V7_OBS / "refuel_v2_obs_sweep.json"
_v5.CASES["refuel_v2"]["fs"]    = _V7_FS  / "refuel_v2_fs_sweep.json"

_original_generate_markdown = _v5.generate_markdown


def _generate_markdown_v7(figures: dict) -> str:
    md = _original_generate_markdown(figures)
    md = md.replace("# Evaluation Summary — v5", "# Evaluation Summary — v7")
    md = md.replace("# Evaluation Summary — v6", "# Evaluation Summary — v7")
    return md


_v5.generate_markdown = _generate_markdown_v7


def main() -> None:
    _v5.main()


if __name__ == "__main__":
    main()
