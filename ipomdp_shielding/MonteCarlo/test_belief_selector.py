"""Test script for BeliefSelector implementation."""

# import sys
# sys.path.insert(0, '/home/scarbro/claude')

from ipomdp_shielding.MonteCarlo import BeliefSelector


class MockRuntimeShield:
    """Mock runtime shield for testing."""

    def __init__(self, action_probs):
        """
        Parameters
        ----------
        action_probs : list of (action, allowed_prob, disallowed_prob)
        """
        self.action_probs = action_probs

    def get_action_probs(self):
        return self.action_probs


def test_belief_selector_best_mode():
    """Test BeliefSelector in 'best' mode selects highest probability action."""
    print("Testing BeliefSelector in 'best' mode...")

    # Create selector
    selector = BeliefSelector(mode="best", exploration_rate=0.0)

    # Mock shield with known probabilities
    # action1: 0.9, action2: 0.5, action3: 0.7
    mock_shield = MockRuntimeShield([
        ("action1", 0.9, 0.1),
        ("action2", 0.5, 0.5),
        ("action3", 0.7, 0.3),
    ])

    # Create context
    context = {"rt_shield": mock_shield, "history": []}

    # Test selection
    allowed_actions = ["action1", "action2", "action3"]
    selected = selector.select([], allowed_actions, context=context)

    assert selected == "action1", f"Expected 'action1', got {selected}"
    print(f"  ✓ Selected highest probability action: {selected}")

    # Test with subset of actions
    allowed_actions = ["action2", "action3"]
    selected = selector.select([], allowed_actions, context=context)

    assert selected == "action3", f"Expected 'action3', got {selected}"
    print(f"  ✓ Selected highest probability from subset: {selected}")


def test_belief_selector_worst_mode():
    """Test BeliefSelector in 'worst' mode selects lowest probability action."""
    print("\nTesting BeliefSelector in 'worst' mode...")

    # Create selector
    selector = BeliefSelector(mode="worst", exploration_rate=0.0)

    # Mock shield with known probabilities
    mock_shield = MockRuntimeShield([
        ("action1", 0.9, 0.1),
        ("action2", 0.5, 0.5),
        ("action3", 0.7, 0.3),
    ])

    # Create context
    context = {"rt_shield": mock_shield, "history": []}

    # Test selection
    allowed_actions = ["action1", "action2", "action3"]
    selected = selector.select([], allowed_actions, context=context)

    assert selected == "action2", f"Expected 'action2', got {selected}"
    print(f"  ✓ Selected lowest probability action: {selected}")


def test_belief_selector_fallback():
    """Test BeliefSelector falls back to random when no context."""
    print("\nTesting BeliefSelector fallback to random...")

    selector = BeliefSelector(mode="best", exploration_rate=0.0)

    # Test without context
    allowed_actions = ["action1", "action2", "action3"]
    selected = selector.select([], allowed_actions, context=None)

    assert selected in allowed_actions, f"Selected {selected} not in allowed actions"
    print(f"  ✓ Fallback works without context: {selected}")


def test_belief_selector_single_action():
    """Test BeliefSelector handles single action correctly."""
    print("\nTesting BeliefSelector with single action...")

    selector = BeliefSelector(mode="best", exploration_rate=0.0)

    mock_shield = MockRuntimeShield([
        ("action1", 0.9, 0.1),
    ])

    context = {"rt_shield": mock_shield, "history": []}

    # Test with single action
    allowed_actions = ["action1"]
    selected = selector.select([], allowed_actions, context=context)

    assert selected == "action1", f"Expected 'action1', got {selected}"
    print(f"  ✓ Single action handled correctly: {selected}")


def test_belief_selector_invalid_mode():
    """Test BeliefSelector raises error for invalid mode."""
    print("\nTesting BeliefSelector with invalid mode...")

    try:
        selector = BeliefSelector(mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "mode must be 'best' or 'worst'" in str(e)
        print(f"  ✓ Invalid mode raises ValueError: {e}")


def test_belief_selector_empty_actions():
    """Test BeliefSelector raises error for empty allowed actions."""
    print("\nTesting BeliefSelector with empty allowed actions...")

    selector = BeliefSelector(mode="best", exploration_rate=0.0)

    try:
        selected = selector.select([], [], context=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No allowed actions" in str(e)
        print(f"  ✓ Empty actions raises ValueError: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("BeliefSelector Test Suite")
    print("=" * 60)

    test_belief_selector_best_mode()
    test_belief_selector_worst_mode()
    test_belief_selector_fallback()
    test_belief_selector_single_action()
    test_belief_selector_invalid_mode()
    test_belief_selector_empty_actions()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
