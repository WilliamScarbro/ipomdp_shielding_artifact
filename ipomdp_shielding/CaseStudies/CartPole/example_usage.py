"""Example usage of the CartPole case study.

This script demonstrates how to:
1. Build a CartPole IPOMDP
2. Inspect the model structure
3. Test the perception bounds
4. Prepare for evaluation
"""

from ipomdp_shielding.CaseStudies.CartPole import (
    build_cartpole_ipomdp,
    cartpole_states,
    cartpole_safe,
    get_bin_edges,
    FAIL,
)


def example_build_ipomdp():
    """Example: Build a CartPole IPOMDP with default parameters."""
    print("=" * 70)
    print("Example 1: Building CartPole IPOMDP")
    print("=" * 70)

    ipomdp, pp_shield, test_data = build_cartpole_ipomdp(
        confidence_method="Clopper_Pearson",
        alpha=0.05,
        train_fraction=0.8,
        num_bins=7,
        smoothing=True,
        seed=42,
    )

    print(f"\n✓ IPOMDP built successfully!")
    print(f"  States: {len(ipomdp.states)}")
    print(f"  Observations: {len(ipomdp.observations)}")
    print(f"  Actions: {ipomdp.actions}")
    print(f"  Dynamics transitions: {len(ipomdp.T)}")
    print(f"  Perception bounds: {len(ipomdp.P_lower)} lower, {len(ipomdp.P_upper)} upper")

    return ipomdp, pp_shield, test_data


def example_inspect_shield(pp_shield):
    """Example: Inspect the perfect-perception shield."""
    print("\n" + "=" * 70)
    print("Example 2: Inspecting Perfect-Perception Shield")
    print("=" * 70)

    # Center state (bin 3 for all dimensions)
    center_state = (3, 3, 3, 3)
    print(f"\nCenter state {center_state}:")
    print(f"  Safe actions: {pp_shield[center_state]}")

    # Count states by number of safe actions
    no_safe = sum(1 for s, actions in pp_shield.items() if len(actions) == 0 and s != FAIL)
    one_safe = sum(1 for s, actions in pp_shield.items() if len(actions) == 1)
    two_safe = sum(1 for s, actions in pp_shield.items() if len(actions) == 2)

    print(f"\n✓ Shield statistics:")
    print(f"  States with 0 safe actions: {no_safe}")
    print(f"  States with 1 safe action: {one_safe}")
    print(f"  States with 2 safe actions: {two_safe}")
    print(f"  Total: {no_safe + one_safe + two_safe + 1} (including FAIL)")


def example_check_safety():
    """Example: Check safety predicate for different states."""
    print("\n" + "=" * 70)
    print("Example 3: Testing Safety Predicate")
    print("=" * 70)

    bin_edges = get_bin_edges()

    test_states = [
        ((3, 3, 3, 3), "center"),
        ((0, 3, 3, 3), "left edge"),
        ((6, 3, 3, 3), "right edge"),
        ((3, 3, 0, 3), "pole left"),
        ((3, 3, 6, 3), "pole right"),
        (FAIL, "FAIL"),
    ]

    print("\nSafety checks:")
    for state, description in test_states:
        safe = cartpole_safe(state, bin_edges)
        status = "✓ SAFE" if safe else "✗ UNSAFE"
        print(f"  {description:12} {str(state):20} {status}")


def example_perception_intervals(ipomdp):
    """Example: Examine perception interval widths."""
    print("\n" + "=" * 70)
    print("Example 4: Perception Interval Analysis")
    print("=" * 70)

    # Pick a sample state
    sample_state = (3, 3, 3, 3)

    # Compute interval widths
    widths = []
    for obs in ipomdp.observations:
        if obs in ipomdp.P_lower[sample_state] and obs in ipomdp.P_upper[sample_state]:
            lower = ipomdp.P_lower[sample_state][obs]
            upper = ipomdp.P_upper[sample_state][obs]
            width = upper - lower
            if width > 0.01:  # Only show significant widths
                widths.append((obs, lower, upper, width))

    # Sort by width
    widths.sort(key=lambda x: x[3], reverse=True)

    print(f"\nPerception intervals for state {sample_state}:")
    print(f"  Total observations with non-zero prob: {len(widths)}")
    print(f"\n  Top 5 largest intervals:")
    for obs, lower, upper, width in widths[:5]:
        print(f"    {str(obs):20} [{lower:.4f}, {upper:.4f}]  width={width:.4f}")

    # Summary statistics
    if widths:
        avg_width = sum(w[3] for w in widths) / len(widths)
        max_width = max(w[3] for w in widths)
        print(f"\n✓ Interval statistics:")
        print(f"    Average width: {avg_width:.4f}")
        print(f"    Maximum width: {max_width:.4f}")


def example_test_data_coverage(test_data):
    """Example: Check test data coverage."""
    print("\n" + "=" * 70)
    print("Example 5: Test Data Coverage")
    print("=" * 70)

    dim_names = ["x", "x_dot", "theta", "theta_dot"]

    print("\nTest set sizes:")
    for dim in dim_names:
        print(f"  {dim:10} {len(test_data[dim])} observations")

    # Check bin coverage in test data
    print("\nBin coverage in test data:")
    for dim in dim_names:
        true_bins = set(true_bin for true_bin, _ in test_data[dim])
        est_bins = set(est_bin for _, est_bin in test_data[dim])
        print(f"  {dim:10} true bins: {sorted(true_bins)}, est bins: {sorted(est_bins)}")


def main():
    """Run all examples."""
    print("\n" + "█" * 70)
    print("CartPole Case Study - Usage Examples")
    print("█" * 70 + "\n")

    # Build IPOMDP
    ipomdp, pp_shield, test_data = example_build_ipomdp()

    # Inspect shield
    example_inspect_shield(pp_shield)

    # Check safety
    example_check_safety()

    # Analyze perception
    example_perception_intervals(ipomdp)

    # Check test coverage
    example_test_data_coverage(test_data)

    print("\n" + "█" * 70)
    print("All examples completed!")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
