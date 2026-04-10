"""Linear Fractional Programming based belief propagation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, List, Tuple, Dict, Callable
import numpy as np

from ..Models import IPOMDP
from .belief_base import IPOMDP_Belief
from .belief_polytope import BeliefPolytope

@dataclass
class LPResult:
    """LP solver result."""
    status: str
    obj: Optional[float]
    x: Optional[np.ndarray]


class LPSolver(Protocol):
    """Protocol for LP solvers."""
    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        sense: str = "min",
    ) -> LPResult:
        ...


class SciPyHiGHSSolver:
    """LP solver using scipy.optimize.linprog with HiGHS."""

    def __init__(self):
        import scipy.optimize  # noqa

    def solve(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None, sense="min"):
        import scipy.optimize as opt

        c = np.array(c, dtype=float)
        if sense == "max":
            c = -c

        n = c.shape[0]
        if lb is None:
            lb = -np.inf * np.ones(n)
        if ub is None:
            ub = np.inf * np.ones(n)

        bounds = list(zip(lb.tolist(), ub.tolist()))
        res = opt.linprog(
            c,
            A_ub=None if A_ub is None else np.array(A_ub, dtype=float),
            b_ub=None if b_ub is None else np.array(b_ub, dtype=float),
            A_eq=None if A_eq is None else np.array(A_eq, dtype=float),
            b_eq=None if b_eq is None else np.array(b_eq, dtype=float),
            bounds=bounds,
            method="highs",
            options={"presolve": False},
        )

        if not res.success:
            return LPResult(status=res.status, obj=None, x=None)

        obj = float(res.fun)
        if sense == "max":
            obj = -obj
        return LPResult(status="optimal", obj=obj, x=np.array(res.x, dtype=float))


def default_solver() -> LPSolver:
    """Return default LP solver."""
    try:
        return SciPyHiGHSSolver()
    except Exception:
        raise RuntimeError("SciPy HiGHS not available.")


@dataclass
class Template:
    """Stores vectors v_k such that phi_k(b) = v_k^T b."""
    V: np.ndarray
    names: Optional[List[str]] = None

    @property
    def K(self) -> int:
        return self.V.shape[0]


class TemplateFactory:
    """Factory for creating template basis functions."""

    @staticmethod
    def canonical(n: int) -> Template:
        """Canonical templates: bound each coordinate."""
        V = np.eye(n)
        names = [f"b[{i}]" for i in range(n)]
        return Template(V, names)

    @staticmethod
    def safe_set_indicators(n: int, safe_sets: Dict[str, List[int]]) -> Template:
        """Templates for action-set probability queries."""
        V = []
        names = []
        for name, idxs in safe_sets.items():
            v = np.zeros(n)
            v[idxs] = 1.0
            V.append(v)
            names.append(f"P({name})")
        return Template(np.stack(V, axis=0), names)

    @staticmethod
    def pca_templates(samples: np.ndarray, k: int) -> Template:
        """PCA on belief samples."""
        X = samples - samples.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        V = VT[:k, :]
        names = [f"pca[{i}]" for i in range(k)]
        return Template(V, names)

    @staticmethod
    def hybrid(templates: List[Template]) -> Template:
        """Combine multiple templates."""
        V = np.vstack([t.V for t in templates])
        names = []
        for t in templates:
            if t.names is None:
                names.extend([None] * t.K)
            else:
                names.extend(t.names)
        return Template(V, names)


def solve_lfp_charnes_cooper(
    solver: LPSolver,
    num: np.ndarray,
    den: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    sense: str = "min",
) -> LPResult:
    """
    Min/Max (num^T x) / (den^T x) subject to linear constraints on x.

    Uses Charnes-Cooper transform:
      y = x / (den^T x), t = 1 / (den^T x)
    """
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)
    n = num.shape[0]

    # Variables are (y, t) in R^(n+1)
    if A_ub is not None:
        A1 = np.hstack([A_ub, -b_ub.reshape(-1, 1)])
        b1 = np.zeros(A_ub.shape[0])
    else:
        A1, b1 = None, None

    Aeq_list = []
    beq_list = []
    if A_eq is not None:
        Aeq_list.append(np.hstack([A_eq, -b_eq.reshape(-1, 1)]))
        beq_list.append(np.zeros(A_eq.shape[0]))

    Aeq_list.append(np.hstack([den.reshape(1, -1), np.array([[0.0]])]))
    beq_list.append(np.array([1.0]))

    A_eq2 = np.vstack(Aeq_list)
    b_eq2 = np.concatenate(beq_list)

    A_bound1 = np.hstack([np.eye(n), -ub.reshape(-1, 1)])
    b_bound1 = np.zeros(n)
    A_bound2 = np.hstack([-np.eye(n), lb.reshape(-1, 1)])
    b_bound2 = np.zeros(n)

    if A1 is None:
        A_ub2 = np.vstack([A_bound1, A_bound2])
        b_ub2 = np.concatenate([b_bound1, b_bound2])
    else:
        A_ub2 = np.vstack([A1, A_bound1, A_bound2])
        b_ub2 = np.concatenate([b1, b_bound1, b_bound2])

    lb2 = np.concatenate([-np.inf * np.ones(n), np.array([0.0])])
    ub2 = np.concatenate([np.inf * np.ones(n), np.array([np.inf])])

    c2 = np.concatenate([num, np.array([0.0])])

    return solver.solve(c2, A_ub2, b_ub2, A_eq2, b_eq2, lb2, ub2, sense=sense)


@dataclass
class LFPPropagator(IPOMDP_Belief):
    """
    Template-based belief propagation using linear fractional programming.

    Over-approximates post-belief set by bounding each template function.
    """
    ipomdp: IPOMDP
    template: Template
    solver: LPSolver
    belief: BeliefPolytope
    denom_vector: Optional[np.ndarray] = None

    # updates belief
    def propagate(self, action: int, obs: int) -> bool:
        """
        Propagate belief polytope through action and observation.

        The feasible_unnormalized_posterior_polytope returns constraints over
        a 4n-dimensional space z = [b, y, w, x] where:
          - b: prior belief (n-dim)
          - y: predicted belief after dynamics (n-dim)
          - w: observation likelihoods (n-dim)
          - x: unnormalized posterior = w * y (n-dim)

        We optimize v^T (x / sum(x)) to bound template values on the normalized posterior.

        Returns
        -------
        bool
            True if propagation succeeded, False if numerical error occurred.
        """
        prior = self.belief

        A_ub, b_ub, A_eq, b_eq, lb, ub, slices = self.ipomdp.feasible_unnormalized_posterior_polytope(
            prior, action, obs
        )

        n = prior.n
        dim = 4 * n  # Total dimension of extended variable space
        sx = slices["x"]  # Slice for x variables (unnormalized posterior)

        # Denominator: 1^T x -> ones on x-slice
        den_extended = np.zeros(dim, dtype=float)
        den_extended[sx] = 1.0

        # ------------------------------------------------------------------
        # Precompute Charnes-Cooper LP constraint matrices ONCE for all K
        # template directions.  The same A_ub2 / A_eq2 / bounds are reused
        # across all K LP pairs; only the objective vector c2 changes.
        # ------------------------------------------------------------------
        #
        # Charnes-Cooper substitution: y_cc = z * t,  t = 1/(den^T z) > 0
        #   A_ub z <= b_ub   ->  A_ub y_cc - b_ub * t <= 0      (A1 block)
        #   A_eq z  = b_eq   ->  A_eq y_cc - b_eq * t  = 0      (Aeq1 block)
        #   den^T z = 1      ->  den^T y_cc             = 1      (normalization)
        #   lb <= z <= ub    ->  y_cc - ub*t <= 0                (A_bound1)
        #                        -y_cc + lb*t <= 0               (A_bound2)

        # Inequality from original problem
        if A_ub is not None and A_ub.size > 0:
            # hstack once: (m_ub, dim+1)
            A1 = np.hstack([A_ub, -b_ub.reshape(-1, 1)])
            b1 = np.zeros(A_ub.shape[0], dtype=float)
        else:
            A1 = np.empty((0, dim + 1), dtype=float)
            b1 = np.empty(0, dtype=float)

        # Bound encoding (avoids creating np.eye(dim) inside the K-loop)
        eye_dim = np.eye(dim, dtype=float)
        A_bound1 = np.hstack([eye_dim, -ub.reshape(-1, 1)])   # (dim, dim+1)
        A_bound2 = np.hstack([-eye_dim, lb.reshape(-1, 1)])   # (dim, dim+1)
        del eye_dim

        A_ub2 = np.vstack([A1, A_bound1, A_bound2])           # (m_ub+2*dim, dim+1)
        b_ub2 = np.concatenate([b1, np.zeros(2 * dim, dtype=float)])
        del A1, A_bound1, A_bound2

        # Equality from original problem + normalization
        Aeq_parts: List[np.ndarray] = []
        beq_parts: List[np.ndarray] = []
        if A_eq is not None and A_eq.size > 0:
            Aeq_parts.append(np.hstack([A_eq, -b_eq.reshape(-1, 1)]))
            beq_parts.append(np.zeros(A_eq.shape[0], dtype=float))
        Aeq_parts.append(np.hstack([den_extended.reshape(1, -1), np.array([[0.0]])]))
        beq_parts.append(np.array([1.0]))
        A_eq2 = np.vstack(Aeq_parts)
        b_eq2 = np.concatenate(beq_parts)

        # Variable bounds: y_cc is unconstrained; t >= 0
        lb2 = np.concatenate([-np.inf * np.ones(dim), [0.0]])
        ub2 = np.concatenate([np.inf * np.ones(dim), [np.inf]])

        # ------------------------------------------------------------------
        # CC scaling: set normalization RHS to S0 ≈ E[sum(x)] to keep the
        # Charnes-Cooper parameter t_s = S0/sum(x) ≈ O(1), avoiding the
        # numerical conditioning issues that arise when sum(x) << 1.
        # With this scaling the LP objective equals S0 * v^T(x/sum(x)),
        # so we divide LP results by S0 to recover actual belief bounds.
        # ------------------------------------------------------------------
        sy = slices["y"]
        sw_sl = slices["w"]
        y_lo_v = lb[sy]
        y_hi_v = ub[sy]
        w_lo_v = lb[sw_sl]
        w_hi_v = ub[sw_sl]
        S_lo_est = float(np.dot(np.maximum(w_lo_v, 0.0), np.maximum(y_lo_v, 0.0)))
        S_hi_est = float(np.dot(w_hi_v, y_hi_v))
        if S_lo_est <= 0.0 or S_hi_est <= 0.0 or S_hi_est < S_lo_est:
            S0 = 1.0
        else:
            S0 = float(np.sqrt(S_lo_est * S_hi_est))
        b_eq2 = b_eq2.copy()
        b_eq2[-1] = S0

        # ------------------------------------------------------------------
        # Solve K pairs of LPs, varying only the objective vector c2
        # ------------------------------------------------------------------
        A_new = []
        d_new = []

        for k in range(self.template.K):
            v = self.template.V[k]

            # Objective: v^T x_cc  (only the x-slice of y_cc is nonzero)
            c2 = np.zeros(dim + 1, dtype=float)
            c2[sx] = v

            lo_res = self.solver.solve(c2, A_ub2, b_ub2, A_eq2, b_eq2, lb2, ub2, sense="min")
            hi_res = self.solver.solve(c2, A_ub2, b_ub2, A_eq2, b_eq2, lb2, ub2, sense="max")

            if lo_res.status != "optimal" or hi_res.status != "optimal":
                # Numerical error - propagation failed
                return False

            lo = lo_res.obj / S0
            hi = hi_res.obj / S0

            # Constraints on the n-dimensional normalized posterior belief
            A_new.append(v.copy())
            d_new.append(hi)
            A_new.append(-v.copy())
            d_new.append(-lo)

        A_new = np.stack(A_new, axis=0)
        d_new = np.array(d_new, dtype=float)

        self.belief = BeliefPolytope(n=prior.n, A=A_new, d=d_new)
        return True

 
    def minimum_allowed_probability(self, allowed) -> float:
        return self.belief.minimum_allowed_prob(allowed)

    def maximum_disallowed_probability(self, disallowed) -> float:
        return self.belief.maximum_allowed_prob(disallowed)

    def restart(self):
        self.belief = BeliefPolytope.uniform_prior(len(self.ipomdp.states))
