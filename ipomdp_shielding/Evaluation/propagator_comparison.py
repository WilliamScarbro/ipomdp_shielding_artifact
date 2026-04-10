"""
Comparison testing of ExactIHMMBelief vs LFPPropagator.

The LFPPropagator computes an over-approximation of the true belief set.
This module verifies soundness by checking that all extreme points from
ExactIHMMBelief are contained within the LFPPropagator's BeliefPolytope.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from ..Models import IPOMDP
from ..Propagators import (
    ExactIHMMBelief,
    LFPPropagator,
    BeliefPolytope,
    Template,
    TemplateFactory,
)
from ..Propagators.lfp_propagator import SciPyHiGHSSolver, default_solver


def point_in_polytope(point: Dict, polytope: BeliefPolytope, states: List, tol: float = 1e-6) -> Tuple[bool, float]:
    """
    Check if a belief point is contained in the polytope.

    Parameters
    ----------
    point : dict
        Belief distribution as state -> probability mapping
    polytope : BeliefPolytope
        The polytope to check containment in
    states : list
        Ordered list of states (defines index mapping)
    tol : float
        Tolerance for constraint violations

    Returns
    -------
    tuple
        (is_contained, max_violation) where max_violation is the largest
        constraint violation (negative means inside)
    """
    n = polytope.n
    b = np.zeros(n)
    for i, s in enumerate(states):
        b[i] = point.get(s, 0.0)

    # Check A @ b <= d
    if polytope.A.size > 0:
        violations = polytope.A @ b - polytope.d
        max_ineq_violation = np.max(violations)
    else:
        max_ineq_violation = 0.0

    # Check equality constraints if present
    if polytope.Aeq is not None and polytope.Aeq.size > 0:
        eq_violations = np.abs(polytope.Aeq @ b - polytope.beq)
        max_eq_violation = np.max(eq_violations)
    else:
        max_eq_violation = 0.0

    # Check simplex constraints: sum = 1, all >= 0
    simplex_sum_violation = abs(sum(b) - 1.0)
    simplex_neg_violation = max(-np.min(b), 0.0)

    total_violation = max(max_ineq_violation, max_eq_violation, simplex_sum_violation, simplex_neg_violation)

    return total_violation <= tol, total_violation


def compare_propagators(
    ipomdp: IPOMDP,
    history: List[Tuple],
    template: Optional[Template] = None,
    verbose: bool = True
) -> Dict:
    """
    Run both ExactIHMMBelief and LFPPropagator on the same history
    and verify that exact points are contained in the LFP polytope.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    history : list
        List of (observation, action) evidence tuples
    template : Template, optional
        Template for LFP propagator. If None, uses canonical templates.
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Results including containment status, violations, and statistics
    """
    n = len(ipomdp.states)
    states = list(ipomdp.states)  # Fixed ordering for index mapping

    # Initialize exact belief propagator
    exact_belief = ExactIHMMBelief(ipomdp, max_points=10000, auto_prune=False)

    # Initialize LFP propagator
    if template is None:
        template = TemplateFactory.canonical(n)

    solver = default_solver()
    lfp_propagator = LFPPropagator(model=ipomdp, template=template, solver=solver)

    # Start with uniform prior polytope
    lfp_polytope = BeliefPolytope.uniform_prior(n)

    results = {
        "steps": [],
        "all_contained": True,
        "total_points_checked": 0,
        "total_violations": 0,
        "max_violation": 0.0
    }

    if verbose:
        print(f"Starting propagator comparison with {len(history)} steps")
        print(f"State space size: {n}")
        print(f"States: {states}")
        print("-" * 60)

    for step, (obs, action) in enumerate(history):
        if verbose:
            print(f"\nStep {step + 1}: observation={obs}, action={action}")

        # Propagate exact belief
        exact_belief.propogate((obs, action))

        # Propagate LFP polytope using the actual LFPPropagator
        lfp_polytope = lfp_propagator.propagate(lfp_polytope, action, obs)

        # Check containment of all exact points
        step_result = {
            "step": step + 1,
            "num_exact_points": len(exact_belief.points),
            "contained": [],
            "violations": [],
        }

        for i, point in enumerate(exact_belief.points):
            contained, violation = point_in_polytope(point, lfp_polytope, states)

            step_result["contained"].append(contained)
            step_result["violations"].append(violation)

            if not contained:
                results["all_contained"] = False
                results["total_violations"] += 1
                if verbose:
                    print(f"  WARNING: Point {i} NOT contained! Violation: {violation:.6f}")
                    print(f"    Point: {point}")

            results["max_violation"] = max(results["max_violation"], violation)

        results["total_points_checked"] += len(exact_belief.points)

        num_contained = sum(step_result["contained"])
        if verbose:
            print(f"  Exact points: {len(exact_belief.points)}")
            print(f"  Contained: {num_contained}/{len(exact_belief.points)}")
            if step_result["violations"]:
                print(f"  Max violation this step: {max(step_result['violations']):.6f}")

        results["steps"].append(step_result)

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total steps: {len(history)}")
        print(f"Total points checked: {results['total_points_checked']}")
        print(f"All contained: {results['all_contained']}")
        print(f"Total violations: {results['total_violations']}")
        print(f"Max violation: {results['max_violation']:.6f}")

        if results["all_contained"]:
            print("\n*** SUCCESS: All exact belief points are contained in LFP polytope ***")
        else:
            print("\n*** FAILURE: Some exact belief points are outside LFP polytope ***")

    return results


def run_simple_test():
    """Run a simple test with a small IPOMDP."""
    # Create a simple 3-state IPOMDP
    states = [0, 1, 2]
    observations = [0, 1, 2]
    actions = [0, 1]

    # Simple dynamics: action 0 stays, action 1 moves right (with some stochasticity)
    dynamics = {}
    for s in states:
        for a in actions:
            if a == 0:
                dynamics[(s, a)] = {s: 1.0}
            else:  # a == 1
                next_s = min(s + 1, 2)
                if next_s == s:
                    dynamics[(s, a)] = {s: 1.0}
                else:
                    dynamics[(s, a)] = {s: 0.2, next_s: 0.8}

    # Interval observation model - some uncertainty about observations
    P_low = {
        0: {0: 0.6, 1: 0.1, 2: 0.0},
        1: {0: 0.1, 1: 0.5, 2: 0.1},
        2: {0: 0.0, 1: 0.1, 2: 0.6},
    }
    P_high = {
        0: {0: 0.9, 1: 0.3, 2: 0.1},
        1: {0: 0.3, 1: 0.8, 2: 0.3},
        2: {0: 0.1, 1: 0.3, 2: 0.9},
    }

    ipomdp = IPOMDP(states, observations, actions, dynamics, P_low, P_high)

    # Run comparison with a history of observations and actions
    history = [(0, 0), (1, 1), (1, 0), (2, 1)]

    print("=" * 60)
    print("PROPAGATOR COMPARISON TEST")
    print("=" * 60)

    results = compare_propagators(ipomdp, history, verbose=True)

    return results


if __name__ == "__main__":
    run_simple_test()
