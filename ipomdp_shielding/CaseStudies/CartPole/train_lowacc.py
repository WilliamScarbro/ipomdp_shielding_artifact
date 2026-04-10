"""Train a low-accuracy CartPole perception model to match TaxiNet's uncertainty level.

Target: mean P_mid(own_obs | state) ≈ 0.354 (matching TaxiNet).
Current (200 episodes): mean P_mid ≈ 0.539.

Usage:
    python -m ipomdp_shielding.CaseStudies.CartPole.train_lowacc [--episodes N] [--epochs E]
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Inject training directory into path so model.py is importable
TRAINING_DIR = Path(__file__).parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from .data_preparation import prepare_perception_data


def measure_accuracy(data_dir: Path, num_bins: int = 3) -> float:
    """Compute mean P_mid(own bin | true bin) across all dimensions."""
    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    p_mids = []

    for dim in dim_names:
        cm_path = data_dir / f"{dim}_confusion.npy"
        if not cm_path.exists():
            raise FileNotFoundError(f"Missing {cm_path}")
        cm = np.load(cm_path).astype(float)  # (k, k)
        k = cm.shape[0]

        # Normalize rows to get P(est_bin | true_bin)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        p_row = cm / row_sums

        # Mean diagonal (own bin probability)
        p_mids.append(np.mean(np.diag(p_row)))

    mean_p_mid = float(np.mean(p_mids))
    print(f"\nPer-dimension accuracy (mean P_mid on diagonal):")
    for dim, p in zip(dim_names, p_mids):
        print(f"  {dim:12s}: {p:.3f}")
    print(f"  Mean P_mid: {mean_p_mid:.3f}")
    return mean_p_mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of training episodes (default: 20)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--bins", type=int, default=3,
                        help="Bins per dimension (default: 3)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: artifacts_lowacc/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "artifacts_lowacc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training low-accuracy CartPole perception model")
    print(f"  episodes={args.episodes}, epochs={args.epochs}, bins={args.bins}")
    print(f"  Output: {out_dir}")
    print()

    # Train perception model
    prepare_perception_data(
        num_episodes=args.episodes,
        epochs=args.epochs,
        num_bins=args.bins,
        seed=args.seed,
        device=args.device,
        data_dir=out_dir,
    )

    # Measure accuracy
    print("\n" + "=" * 60)
    print("Measuring perception accuracy...")
    print("=" * 60)
    mean_p_mid = measure_accuracy(out_dir, num_bins=args.bins)

    # Compare with target
    target = 0.354
    print(f"\nTarget (TaxiNet): {target:.3f}")
    print(f"Achieved:         {mean_p_mid:.3f}")
    if abs(mean_p_mid - target) < 0.05:
        print("✓ Close to target!")
    elif mean_p_mid > target + 0.05:
        print(f"→ Still too accurate. Try fewer episodes (e.g., {args.episodes // 2})")
    else:
        print(f"→ Too inaccurate. Try more episodes (e.g., {args.episodes * 2})")


if __name__ == "__main__":
    main()
