"""
Single run debugger for LFPPropagator shielding.

Runs a single IPOMDP instance and prints detailed information about the state
of the system and the belief of the propagator at each step.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Set, Union, cast
import random
import numpy as np

from ..Models import IPOMDP
from ..Propagators import LFPPropagator, BeliefPolytope, Template, TemplateFactory, IPOMDP_Belief
from ..Propagators.lfp_propagator import default_solver
from .runtime_shield import RuntimeImpShield


# Type alias for belief propagators (LFPPropagator is duck-typed as IPOMDP_Belief)
BeliefPropagator = IPOMDP_Belief


# =============================================================================
# CONFIGURATION - Modify these to customize the debugger
# =============================================================================

@dataclass
class DebugConfig:
    """Configuration for the single run debugger."""

    # Simulation parameters
    num_steps: int = 20
    random_seed: Optional[int] = 42

    # Shielding parameters
    action_threshold: float = 0.5  # Min probability for action to be allowed
    default_action: Any = 0        # Fallback when no action is allowed

    # Template configuration: "canonical", "safe_set", "hybrid", or "custom"
    template_type: str = "canonical"

    # Output verbosity: 0=minimal, 1=normal, 2=verbose, 3=debug
    verbosity: int = 2

    # Print belief polytope constraints (can be verbose)
    print_belief_constraints: bool = False

    # Print per-action probability details
    print_action_probabilities: bool = True


# =============================================================================
# STATISTICS TRACKER
# =============================================================================

@dataclass
class RunStatistics:
    """Tracks aggregate statistics during a run."""

    total_steps: int = 0
    propagation_failures: int = 0
    stuck_count: int = 0  # No action met threshold
    default_action_used: int = 0
    actions_taken: Dict[Any, int] = field(default_factory=dict)
    states_visited: Dict[Any, int] = field(default_factory=dict)
    observations_seen: Dict[Any, int] = field(default_factory=dict)

    # Per-step allowed action counts
    allowed_action_counts: List[int] = field(default_factory=list)

    # Track when we entered FAIL state
    failed_at_step: Optional[int] = None

    # Per-step top states data: list of [(state, max_prob), ...]
    top_states_per_step: List[List[tuple]] = field(default_factory=list)

    # Track if true state was in top-k
    true_state_in_top_k: int = 0

    def record_step(
        self,
        state: Any,
        obs: Any,
        action: Any,
        allowed_actions: List[Any],
        used_default: bool,
        propagation_failed: bool,
        top_states: Optional[List[tuple]] = None
    ):
        self.total_steps += 1

        # Track state visits
        self.states_visited[state] = self.states_visited.get(state, 0) + 1

        # Track observations
        self.observations_seen[obs] = self.observations_seen.get(obs, 0) + 1

        # Track actions
        self.actions_taken[action] = self.actions_taken.get(action, 0) + 1

        # Track allowed action count
        self.allowed_action_counts.append(len(allowed_actions))

        if len(allowed_actions) == 0:
            self.stuck_count += 1

        if used_default:
            self.default_action_used += 1

        if propagation_failed:
            self.propagation_failures += 1

        # Track top states
        if top_states is not None:
            self.top_states_per_step.append(top_states)
            top_state_names = [s for s, _ in top_states]
            if state in top_state_names:
                self.true_state_in_top_k += 1

    def report(self) -> str:
        """Generate a statistics report."""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("AGGREGATE STATISTICS")
        lines.append("=" * 60)
        lines.append(f"Total steps:              {self.total_steps}")
        lines.append(f"Propagation failures:     {self.propagation_failures}")
        lines.append(f"Times stuck (no action):  {self.stuck_count}")
        lines.append(f"Default action used:      {self.default_action_used}")

        if self.failed_at_step is not None:
            lines.append(f"Failed at step:           {self.failed_at_step}")
        else:
            lines.append(f"Final state:              SAFE (completed all steps)")

        if self.allowed_action_counts:
            avg_allowed = sum(self.allowed_action_counts) / len(self.allowed_action_counts)
            min_allowed = min(self.allowed_action_counts)
            max_allowed = max(self.allowed_action_counts)
            lines.append("")
            lines.append(f"Allowed actions per step:")
            lines.append(f"  Average: {avg_allowed:.2f}")
            lines.append(f"  Min:     {min_allowed}")
            lines.append(f"  Max:     {max_allowed}")

        lines.append("")
        lines.append("Actions taken:")
        for action, count in sorted(self.actions_taken.items(), key=lambda x: -x[1]):
            lines.append(f"  {action}: {count} times")

        lines.append("")
        lines.append("States visited:")
        for state, count in sorted(self.states_visited.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {state}: {count} times")
        if len(self.states_visited) > 10:
            lines.append(f"  ... and {len(self.states_visited) - 10} more states")

        # Top states tracking
        if self.top_states_per_step:
            lines.append("")
            lines.append("Belief accuracy (true state in top-5):")
            accuracy = self.true_state_in_top_k / len(self.top_states_per_step)
            lines.append(f"  {self.true_state_in_top_k}/{len(self.top_states_per_step)} steps ({accuracy:.1%})")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# DEBUGGER IMPLEMENTATION
# =============================================================================

def create_template(
    ipomdp: IPOMDP,
    template_type: str,
    pp_shield: Optional[Dict[Any, Set[Any]]] = None
) -> Template:
    """
    Create a template based on configuration.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    template_type : str
        One of: "canonical", "safe_set", "hybrid", "custom"
    pp_shield : dict, optional
        Perfect perception shield (needed for safe_set template)
    """
    n = len(ipomdp.states)

    if template_type == "canonical":
        return TemplateFactory.canonical(n)

    elif template_type == "safe_set":
        if pp_shield is None:
            raise ValueError("safe_set template requires pp_shield")

        # Build safe sets for each action
        state_to_idx = {s: i for i, s in enumerate(ipomdp.states)}
        safe_sets = {}
        for action in ipomdp.actions:
            idxs = [state_to_idx[s] for s in ipomdp.states if action in pp_shield.get(s, set())]
            safe_sets[f"safe_{action}"] = idxs
        return TemplateFactory.safe_set_indicators(n, safe_sets)

    elif template_type == "hybrid":
        canonical = TemplateFactory.canonical(n)
        if pp_shield is not None:
            state_to_idx = {s: i for i, s in enumerate(ipomdp.states)}
            safe_sets = {}
            for action in ipomdp.actions:
                idxs = [state_to_idx[s] for s in ipomdp.states if action in pp_shield.get(s, set())]
                safe_sets[f"safe_{action}"] = idxs
            safe_template = TemplateFactory.safe_set_indicators(n, safe_sets)
            return TemplateFactory.hybrid([canonical, safe_template])
        return canonical

    else:
        # Default to canonical for unknown types
        print(f"Unknown template type '{template_type}', using canonical")
        return TemplateFactory.canonical(n)


def print_belief_info(
    propagator: LFPPropagator,
    config: DebugConfig,
    state_names: Optional[List[str]] = None
):
    """Print information about the current belief polytope."""
    belief = propagator.belief
    n = belief.n

    if state_names is None:
        state_names = [str(s) for s in propagator.ipomdp.states]

    if config.verbosity >= 2:
        print(f"  Belief polytope: {belief.A.shape[0]} constraints over {n} states")

        if config.print_belief_constraints and config.verbosity >= 3:
            print("  Constraints (v^T b <= d):")
            for i in range(min(belief.A.shape[0], 10)):
                v = belief.A[i]
                d = belief.d[i]
                nonzero = np.where(np.abs(v) > 1e-10)[0]
                if len(nonzero) <= 3:
                    terms = " + ".join(f"{v[j]:.3f}*b[{state_names[j]}]" for j in nonzero)
                    print(f"    {terms} <= {d:.4f}")
            if belief.A.shape[0] > 10:
                print(f"    ... and {belief.A.shape[0] - 10} more constraints")


def print_action_info(
    propagator: LFPPropagator,
    inv_shield: Dict[Any, List[int]],
    actions: List[Any],
    threshold: float,
    config: DebugConfig
):
    """Print information about action allowability."""
    if not config.print_action_probabilities:
        return

    print("  Action safety probabilities:")
    for action in actions:
        allowed_states = inv_shield[action]
        prob = propagator.allowed_probability(allowed_states)
        status = "ALLOWED" if prob >= threshold else "BLOCKED"
        print(f"    Action {action}: P(safe) >= {prob:.4f} [{status}]")


def compute_top_k_states(
    belief: BeliefPolytope,
    states: List[Any],
    k: int = 5
) -> List[tuple]:
    """
    Compute the k states with highest maximum probability in the belief polytope.

    Uses maximum_allowed_prob to find the upper bound on each state's probability.

    Parameters
    ----------
    belief : BeliefPolytope
        Current belief polytope
    states : list
        List of states (must match belief.n in length)
    k : int
        Number of top states to return

    Returns
    -------
    list of (state, max_prob) tuples
        Top k states sorted by maximum probability (descending)
    """
    state_probs = []

    for i, state in enumerate(states):
        # maximum_allowed_prob takes a list of state indices
        max_prob = belief.maximum_allowed_prob([i])
        state_probs.append((state, max_prob))

    # Sort by probability descending
    state_probs.sort(key=lambda x: -x[1])

    return state_probs[:k]


def run_single_debug(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, Set[Any]],
    perceptor: Callable[[Any], Any],
    initial_state: Any,
    initial_action: Any,
    config: Optional[DebugConfig] = None,
    action_selector: Optional[Callable[[List[Any]], Any]] = None
) -> RunStatistics:
    """
    Run a single IPOMDP episode with detailed debugging output.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    perceptor : callable
        Maps true state to observation
    initial_state : any
        Starting state
    initial_action : any
        Initial action (for first propagation step)
    config : DebugConfig, optional
        Configuration options (uses defaults if None)
    action_selector : callable, optional
        Function to select action from allowed set (default: random choice)

    Returns
    -------
    RunStatistics
        Aggregate statistics from the run
    """
    if config is None:
        config = DebugConfig()

    if action_selector is None:
        action_selector = lambda actions: random.choice(actions) if actions else config.default_action

    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    # Create template and propagator
    template = create_template(ipomdp, config.template_type, pp_shield)
    solver = default_solver()
    n = len(ipomdp.states)
    initial_belief = BeliefPolytope.uniform_prior(n)

    propagator = LFPPropagator(
        ipomdp=ipomdp,
        template=template,
        solver=solver,
        belief=initial_belief
    )

    # Create runtime shield
    # Note: LFPPropagator is duck-typed as IPOMDP_Belief
    rt_shield = RuntimeImpShield(
        pp_shield=pp_shield,
        ipomdp_belief=cast(IPOMDP_Belief, propagator),
        action_shield=config.action_threshold,
        default_action=config.default_action
    )

    # State name mapping for printing
    state_names = [str(s) for s in ipomdp.states]

    # Initialize statistics
    stats = RunStatistics()

    # Print header
    print("=" * 60)
    print("LFPPropagator Single Run Debugger")
    print("=" * 60)
    print(f"States: {n}")
    print(f"Actions: {list(ipomdp.actions)}")
    print(f"Template type: {config.template_type}")
    print(f"Template functions: {template.K}")
    print(f"Action threshold: {config.action_threshold}")
    print(f"Max steps: {config.num_steps}")
    print("=" * 60)
    print()

    # Run simulation
    state = initial_state
    action = initial_action

    for step in range(config.num_steps):
        # Check for failure
        if state == "FAIL":
            stats.failed_at_step = step
            print(f"\n*** FAILED at step {step} ***")
            break

        # Get observation
        obs = perceptor(state)

        print(f"\n--- Step {step} ---")
        print(f"  True state:   {state}")
        print(f"  Observation:  {obs}")
        print(f"  Last action:  {action}")

        # Propagate belief
        propagation_success = propagator.propagate(action, obs)

        if not propagation_success:
            print("  [WARNING] Propagation failed (numerical error)")

        # Print belief info
        print_belief_info(propagator, config, state_names)

        # Compute and print top-5 most likely states
        top_states = compute_top_k_states(propagator.belief, list(ipomdp.states), k=5)
        true_state_in_top = state in [s for s, _ in top_states]
        print(f"  Top 5 most likely states (by max probability):")
        for rank, (s, prob) in enumerate(top_states, 1):
            marker = " <-- TRUE STATE" if s == state else ""
            print(f"    {rank}. {s}: max P <= {prob:.4f}{marker}")
        if not true_state_in_top:
            # Find the true state's rank
            all_state_probs = []
            for i, s in enumerate(ipomdp.states):
                max_prob = propagator.belief.maximum_allowed_prob([i])
                all_state_probs.append((s, max_prob))
            all_state_probs.sort(key=lambda x: -x[1])
            true_rank = next(i for i, (s, _) in enumerate(all_state_probs, 1) if s == state)
            true_prob = next(p for s, p in all_state_probs if s == state)
            print(f"    ... true state {state} ranked #{true_rank} with max P <= {true_prob:.4f}")

        # Get allowed actions
        # We need to compute this manually to get detailed info
        action_probs = rt_shield.get_action_probs()
        allowed_actions = [a for a, ap, dp in action_probs if ap >= config.action_threshold or dp <= 1- config.action_threshold]

        # Print action info
        if config.print_action_probabilities:
            print("  Action safety probabilities:")
            for a, prob, dp in action_probs:
                status = "ALLOWED" if prob >= config.action_threshold else "BLOCKED"
                print(f"    Action {a}: P(safe) >= {prob:.4f} [{status}]")

        print(f"  Allowed actions: {allowed_actions}")

        # Select action
        used_default = False
        if allowed_actions:
            action = action_selector(allowed_actions)
        else:
            action = config.default_action
            used_default = True
            print(f"  [STUCK] No action allowed, using default: {action}")

        print(f"  Selected action: {action}")

        # Record statistics
        stats.record_step(
            state=state,
            obs=obs,
            action=action,
            allowed_actions=allowed_actions,
            used_default=used_default,
            propagation_failed=not propagation_success,
            top_states=top_states
        )

        # Evolve state
        state = ipomdp.evolve(state, action)

        if config.verbosity >= 2:
            print(f"  Next state:   {state}")

    # Final check
    if state == "FAIL" and stats.failed_at_step is None:
        stats.failed_at_step = config.num_steps

    # Print statistics
    print(stats.report())

    return stats


# =============================================================================
# EXAMPLE USAGE WITH TAXINET
# =============================================================================

def run_taxinet_debug():
    """
    Example: Run debugger with TaxiNet case study.

    Modify this function to customize the debugging session.
    """
    from ..CaseStudies.Taxinet import build_taxinet_ipomdp

    # Build the model
    print("Building TaxiNet IPOMDP...")
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp(
        confidence_method="Clopper_Pearson",
        alpha=0.05,
        train_fraction=0.8
    )

    # Define perception function
    def perceptor(state):
        if state == "FAIL":
            return "FAIL"
        cte, he = state
        cte_obs = random.choice(test_cte_model.get(cte, [cte]))
        he_obs = random.choice(test_he_model.get(he, [he]))
        return (cte_obs, he_obs)

    # Configure debugger
    config = DebugConfig(
        num_steps=30,
        random_seed=42,
        action_threshold=0.9,
        default_action=0,
        template_type="canonical",  # Try: "canonical", "safe_set", "hybrid"
        verbosity=2,
        print_belief_constraints=False,
        print_action_probabilities=True
    )

    # Run debugger
    stats = run_single_debug(
        ipomdp=ipomdp,
        pp_shield=dyn_shield,
        perceptor=perceptor,
        initial_state=(0, 0),
        initial_action=0,
        config=config
    )

    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_taxinet_debug()
