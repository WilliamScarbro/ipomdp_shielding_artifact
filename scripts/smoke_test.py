from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

def assert_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty artifact output at {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from ipomdp_shielding.experiments.plot_experiment_summary import main as plot_summary
    from ipomdp_shielding.experiments.run_threshold_sweep import run_sweep_for_case_study, save_sweep

    with tempfile.TemporaryDirectory(prefix="ipomdp_shielding_artifact_") as tmp:
        tmp_root = Path(tmp)
        shutil.copytree(repo_root / "results", tmp_root / "results")

        params = {
            "num_trials": 2,
            "trial_length": 5,
            "exclude_envelope": True,
            "config_name": "rl_shield_cartpole_lowacc_artifact",
        }

        old_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_root)
            sweep_results, setup_info, base_config = run_sweep_for_case_study("cartpole_lowacc", params)
            save_sweep(
                "cartpole_lowacc",
                sweep_results,
                base_config,
                params,
                setup_info,
                output_dir="results/experiment/threshold",
            )
            plot_summary()
        finally:
            os.chdir(old_cwd)

        assert_nonempty(tmp_root / "results/experiment/summary_experiment_bars.png")
        assert_nonempty(tmp_root / "results/experiment/barchart_experiment_cartpole_lowacc.png")
        assert_nonempty(tmp_root / "evaluation_summary.md")


if __name__ == "__main__":
    main()
