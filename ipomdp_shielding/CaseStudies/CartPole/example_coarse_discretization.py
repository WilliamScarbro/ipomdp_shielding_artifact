"""Example: Using configurable discretization for CartPole.

This example demonstrates how to use different discretization configurations
to trade off state space size vs. precision.
"""

from ipomdp_shielding.CaseStudies.CartPole import build_cartpole_ipomdp

# Example 1: Uniform coarse discretization (5 bins per dimension)
# Total states: 5^4 = 625
print("=" * 60)
print("Example 1: Uniform coarse (5 bins per dimension)")
print("=" * 60)
ipomdp_coarse, pp_shield_coarse, test_data_coarse, _ = build_cartpole_ipomdp(
    num_bins=5,
    seed=42
)
print(f"States: {len(ipomdp_coarse.states)}")
print(f"Total: 5^4 = {5**4} states (+ 1 FAIL state)")
print()

# Example 2: Non-uniform discretization (higher precision on position and angle)
# Total states: 7 * 5 * 7 * 5 = 1225
print("=" * 60)
print("Example 2: Non-uniform (higher precision on position and angle)")
print("=" * 60)
num_bins_config = [7, 5, 7, 5]  # [n_x, n_xdot, n_theta, n_thetadot]
ipomdp_nonuniform, pp_shield_nonuniform, test_data_nonuniform, _ = build_cartpole_ipomdp(
    num_bins=num_bins_config,
    seed=42
)
print(f"States: {len(ipomdp_nonuniform.states)}")
print(f"Configuration: {num_bins_config}")
print(f"Total: 7 × 5 × 7 × 5 = {7*5*7*5} states (+ 1 FAIL state)")
print()

# Example 3: Very coarse discretization (4 bins per dimension)
# Total states: 4^4 = 256
print("=" * 60)
print("Example 3: Very coarse (4 bins per dimension)")
print("=" * 60)
ipomdp_very_coarse, pp_shield_very_coarse, test_data_very_coarse, _ = build_cartpole_ipomdp(
    num_bins=4,
    seed=42
)
print(f"States: {len(ipomdp_very_coarse.states)}")
print(f"Total: 4^4 = {4**4} states (+ 1 FAIL state)")
print()

# Example 4: Minimal discretization for fastest experiments
# Total states: 3^4 = 81
print("=" * 60)
print("Example 4: Minimal (3 bins per dimension)")
print("=" * 60)
ipomdp_minimal, pp_shield_minimal, test_data_minimal, _ = build_cartpole_ipomdp(
    num_bins=3,
    seed=42
)
print(f"States: {len(ipomdp_minimal.states)}")
print(f"Total: 3^4 = {3**4} states (+ 1 FAIL state)")
print()

# Comparison table
print("=" * 60)
print("Discretization Comparison")
print("=" * 60)
print(f"{'Configuration':<30} {'States':<10} {'Feasible?':<10}")
print("-" * 60)
print(f"{'Original (7x7x7x7)':<30} {7**4:<10} {'Too slow':<10}")
print(f"{'Coarse uniform (5x5x5x5)':<30} {5**4:<10} {'Yes':<10}")
print(f"{'Non-uniform (7x5x7x5)':<30} {7*5*7*5:<10} {'Marginal':<10}")
print(f"{'Very coarse (4x4x4x4)':<30} {4**4:<10} {'Yes':<10}")
print(f"{'Minimal (3x3x3x3)':<30} {3**4:<10} {'Very fast':<10}")
print()

print("Recommendation: Use 4-5 bins per dimension for LFP propagator.")
print("For non-uniform: prioritize position and angle over velocities.")
