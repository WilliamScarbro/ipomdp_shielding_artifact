"""Belief polytope representation and volume computation.

This module provides the BeliefPolytope class for representing convex polytopes
on the probability simplex, along with methods for computing polytope volume.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

try:
    import polytope as pc
except ImportError:
    pc = None


@dataclass
class BeliefPolytope:
    """Represents a belief polytope: { b | A b <= d, b >= 0, 1^T b = 1 }.

    A convex subset of the probability simplex defined by linear inequality
    constraints.

    Attributes
    ----------
    n : int
        Dimension of the belief space (number of states)
    A : np.ndarray
        Constraint matrix (m x n) for inequality constraints A @ b <= d
    d : np.ndarray
        Right-hand side vector (m,) for inequality constraints
    Aeq : Optional[np.ndarray]
        Optional equality constraint matrix (beyond sum-to-one)
    beq : Optional[np.ndarray]
        Optional equality constraint RHS
    """
    n: int
    A: np.ndarray
    d: np.ndarray
    Aeq: Optional[np.ndarray] = None
    beq: Optional[np.ndarray] = None

    def with_added_inequalities(self, A_new: np.ndarray, d_new: np.ndarray) -> BeliefPolytope:
        """Return a new polytope with additional inequality constraints."""
        A2 = np.vstack([self.A, A_new]) if self.A.size else A_new.copy()
        d2 = np.concatenate([self.d, d_new]) if self.d.size else d_new.copy()
        return BeliefPolytope(self.n, A2, d2, self.Aeq, self.beq)

    def as_lp_constraints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (A_ub, b_ub, A_eq, b_eq, lb, ub) suitable for LP."""
        A_ub, b_ub = self.A, self.d

        ones = np.ones((1, self.n))
        if self.Aeq is None:
            A_eq = ones
            b_eq = np.array([1.0])
        else:
            A_eq = np.vstack([self.Aeq, ones])
            b_eq = np.concatenate([self.beq, np.array([1.0])])

        lb = np.zeros(self.n)
        ub = np.ones(self.n)
        return A_ub, b_ub, A_eq, b_eq, lb, ub

    def as_pypolytope(self):
        if pc is None:
            raise ImportError("polytope package not installed")
        return pc.Polytope(self.A, self.d)
    
    @staticmethod
    def uniform_prior(n: int) -> BeliefPolytope:
        """Returns a BeliefPolytope representing a uniform prior belief."""
        A = np.vstack([np.eye(n), -np.eye(n)])
        d = np.concatenate([(1.0 / n) * np.ones(n), -(1.0 / n) * np.ones(n)])
        return BeliefPolytope(n=n, A=A, d=d, Aeq=None, beq=None)

    def maximize_linear(self, c: np.ndarray) -> float:
        """Solve: max_b c^T b s.t. b in this BeliefPolytope."""
        c = np.asarray(c, dtype=float)
        if c.shape != (self.n,):
            raise ValueError(f"Objective must have shape ({self.n},), got {c.shape}")

        A_ub, b_ub = self.A, self.d

        ones = np.ones((1, self.n))
        if self.Aeq is None:
            A_eq = ones
            b_eq = np.array([1.0])
        else:
            A_eq = np.vstack([self.Aeq, ones])
            b_eq = np.concatenate([self.beq, np.array([1.0])])

        bounds = [(0.0, 1.0) for _ in range(self.n)]

        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        return float(-res.fun)

    def maximum_allowed_prob(self, allowed: List[int]) -> float:
        """Compute max_b sum_{i in allowed} b_i."""
        c = np.zeros(self.n, dtype=float)
        for i in allowed:
            c[i] = 1.0
        return self.maximize_linear(c)

    def minimum_allowed_prob(self, allowed: List[int]) -> float:
        """Compute min_b sum_{i in allowed} b_i."""
        c = np.zeros(self.n, dtype=float)
        for i in allowed:
            if i < 0 or i >= self.n:
                raise ValueError(f"State index out of range: {i}")
            c[i] = 1.0
        return -self.maximize_linear(-c)

    def volume(self) -> float:
        """Compute the volume of this polytope as a fraction of the simplex.

        Returns
        -------
        float
            Volume as a fraction of the full probability simplex (0 to 1).
            Returns 0.0 if volume computation fails.
        """
        return compute_volume(self)




def _project_constraints(polytope: BeliefPolytope) -> Tuple[np.ndarray, np.ndarray]:
    """Project polytope constraints to (n-1) dimensional space.

    Eliminates the last coordinate using b_n = 1 - sum(b_1, ..., b_{n-1}).

    Original constraint: A @ b <= d where b in R^n, sum(b) = 1
    With substitution b_n = 1 - sum(b'), where b' = [b_1, ..., b_{n-1}]:
        A @ [b'; 1-sum(b')] <= d
        A[:, :-1] @ b' + A[:, -1] * (1 - sum(b')) <= d
        A[:, :-1] @ b' + A[:, -1] - A[:, -1] * sum(b') <= d
        (A[:, :-1] - A[:, -1] @ ones^T) @ b' <= d - A[:, -1]

    Returns (A_proj, d_proj) for the projected constraints.
    """
    n = polytope.n
    m = n - 1

    if polytope.A.size == 0:
        return np.empty((0, m)), np.empty(0)

    # A_proj = A[:, :-1] - outer(A[:, -1], ones)
    A_orig = polytope.A
    last_col = A_orig[:, -1].reshape(-1, 1)
    ones_row = np.ones((1, m))
    A_proj = A_orig[:, :-1] - last_col @ ones_row

    # d_proj = d - A[:, -1]
    d_proj = polytope.d - A_orig[:, -1]

    return A_proj, d_proj


def _find_interior_point_projected(polytope: BeliefPolytope) -> Optional[np.ndarray]:
    """Find a strictly interior point in (n-1) dimensional projected space.

    The projected space has variables b' = [b_1, ..., b_{n-1}] with:
    - Projected original constraints
    - b'_i >= 0 for all i
    - sum(b') <= 1  (so that b_n = 1 - sum(b') >= 0)
    """
    n = polytope.n
    m = n - 1  # Projected dimension

    # Get projected constraints
    A_proj, d_proj = _project_constraints(polytope)

    # Variables: [b'_1, ..., b'_{m}, t] where t is the slack for Chebyshev center
    # Maximize t
    c = np.zeros(m + 1)
    c[-1] = -1  # Maximize t (minimize -t)

    # Build inequality constraints with slack
    halfspaces = []
    rhs = []

    # Projected original constraints: A_proj @ b' + t <= d_proj
    if A_proj.size > 0:
        for i in range(A_proj.shape[0]):
            row = np.append(A_proj[i], 1.0)  # Add slack
            halfspaces.append(row)
            rhs.append(d_proj[i])

    # Non-negativity: b'_i >= t, i.e., -b'_i + t <= 0
    for i in range(m):
        row = np.zeros(m + 1)
        row[i] = -1.0
        row[-1] = 1.0
        halfspaces.append(row)
        rhs.append(0.0)

    # b_n >= t: 1 - sum(b') >= t, i.e., sum(b') + t <= 1
    row = np.ones(m + 1)
    halfspaces.append(row)
    rhs.append(1.0)

    A_ub = np.array(halfspaces) if halfspaces else np.empty((0, m + 1))
    b_ub = np.array(rhs) if rhs else np.empty(0)

    # Bounds: b'_i in [0, 1], t >= 0
    bounds = [(0, 1) for _ in range(m)] + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success and res.x[-1] > 1e-9:
        return res.x[:m]
    return None


def _extract_equality_constraints(A: np.ndarray, d: np.ndarray,
                                    tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract implicit equality constraints from opposing inequalities.

    Detects pairs of constraints a^T x <= b and -a^T x <= -b (i.e., a^T x >= b)
    which together form the equality a^T x = b.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix (m x n)
    d : np.ndarray
        Right-hand side vector (m,)
    tol : float
        Tolerance for detecting opposing constraints

    Returns
    -------
    A_eq : np.ndarray
        Equality constraint matrix
    b_eq : np.ndarray
        Equality constraint RHS
    A_ineq : np.ndarray
        Remaining inequality constraint matrix
    b_ineq : np.ndarray
        Remaining inequality constraint RHS
    """
    if A.size == 0:
        n = A.shape[1] if len(A.shape) > 1 else 0
        return np.empty((0, n)), np.empty(0), A.copy(), d.copy()

    m, n = A.shape
    used = np.zeros(m, dtype=bool)
    eq_rows = []
    eq_rhs = []

    for i in range(m):
        if used[i]:
            continue
        for j in range(i + 1, m):
            if used[j]:
                continue
            # Check if A[i] ≈ -A[j] and d[i] ≈ -d[j]
            if np.allclose(A[i], -A[j], atol=tol) and np.isclose(d[i], -d[j], atol=tol):
                eq_rows.append(A[i])
                eq_rhs.append(d[i])
                used[i] = True
                used[j] = True
                break

    # Build equality and inequality matrices
    if eq_rows:
        A_eq = np.array(eq_rows)
        b_eq = np.array(eq_rhs)
    else:
        A_eq = np.empty((0, n))
        b_eq = np.empty(0)

    remaining_idx = ~used
    A_ineq = A[remaining_idx]
    b_ineq = d[remaining_idx]

    return A_eq, b_eq, A_ineq, b_ineq


def _find_affine_subspace_basis(A_eq: np.ndarray, b_eq: np.ndarray,
                                  n: int, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Find a basis for the affine subspace defined by equality constraints.

    Given Ax = b, finds a particular solution x0 and a basis for the null space of A.
    The affine subspace is then { x0 + N @ z : z in R^k } where k = n - rank(A).

    Parameters
    ----------
    A_eq : np.ndarray
        Equality constraint matrix (m x n)
    b_eq : np.ndarray
        Equality constraint RHS (m,)
    n : int
        Ambient dimension
    tol : float
        Tolerance for determining rank

    Returns
    -------
    x0 : np.ndarray
        A particular solution (n,)
    null_basis : np.ndarray
        Orthonormal basis for null space (n x k), where k = n - rank(A)
    """
    if A_eq.size == 0:
        # No equality constraints - full space
        return np.zeros(n), np.eye(n)

    # Use SVD to find null space
    _, S, Vt = np.linalg.svd(A_eq, full_matrices=True)

    # Determine rank
    rank = np.sum(S > tol * S[0]) if len(S) > 0 and S[0] > tol else 0

    # Null space basis: last (n - rank) rows of Vt, transposed to columns
    null_basis = Vt[rank:].T

    # Particular solution via least squares
    if rank > 0:
        x0, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
    else:
        x0 = np.zeros(n)

    return x0, null_basis


def _check_feasibility(polytope: BeliefPolytope) -> Optional[np.ndarray]:
    """Check if polytope is feasible and return a feasible point if so.

    Returns
    -------
    Optional[np.ndarray]
        A feasible point if the polytope is non-empty, None if infeasible
    """
    n = polytope.n
    A_ub, b_ub, A_eq, b_eq, lb, ub = polytope.as_lp_constraints()
    bounds = list(zip(lb, ub))

    c = np.zeros(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.success:
        return res.x
    return None


def _enumerate_vertices_with_equalities(polytope: BeliefPolytope,
                                          max_vertices: int = 1000) -> Optional[np.ndarray]:
    """Enumerate vertices handling both explicit and implicit equality constraints.

    First extracts implicit equalities from opposing inequalities and tight
    coordinate bounds, then projects to the affine subspace defined by all
    equalities before vertex enumeration.
    """
    n = polytope.n

    # First check if polytope is feasible
    feasible_point = _check_feasibility(polytope)
    if feasible_point is None:
        return None  # Empty polytope

    # Extract implicit equalities from the inequality constraints
    A_eq_implicit, b_eq_implicit, A_ineq, b_ineq = _extract_equality_constraints(
        polytope.A, polytope.d
    )

    # Combine with explicit equality constraints
    # Always include simplex constraint: sum = 1
    simplex_row = np.ones((1, n))
    simplex_rhs = np.array([1.0])

    eq_parts = [simplex_row]
    eq_rhs_parts = [simplex_rhs]

    if A_eq_implicit.size > 0:
        eq_parts.append(A_eq_implicit)
        eq_rhs_parts.append(b_eq_implicit)

    if polytope.Aeq is not None and polytope.Aeq.size > 0:
        eq_parts.append(polytope.Aeq)
        eq_rhs_parts.append(polytope.beq)

    A_eq_full = np.vstack(eq_parts)
    b_eq_full = np.concatenate(eq_rhs_parts)

    # Find the affine subspace
    x0, null_basis = _find_affine_subspace_basis(A_eq_full, b_eq_full, n)
    k = null_basis.shape[1]  # Dimension of null space

    if k == 0:
        # Zero-dimensional: single point
        # Verify x0 satisfies constraints
        if A_ineq.size > 0 and np.any(A_ineq @ x0 > polytope.d[~np.isin(np.arange(len(polytope.d)),
                                                                         np.where(np.isin(polytope.A.tolist(),
                                                                                           A_eq_implicit.tolist()))[0])] + 1e-8):
            return None
        return x0.reshape(1, -1)

    # Project inequalities to the null space coordinates
    # Original: A_ineq @ x <= b_ineq, with x = x0 + null_basis @ z
    # Projected: A_ineq @ (x0 + null_basis @ z) <= b_ineq
    #            A_ineq @ null_basis @ z <= b_ineq - A_ineq @ x0
    if A_ineq.size > 0:
        A_proj = A_ineq @ null_basis
        b_proj = b_ineq - A_ineq @ x0
    else:
        A_proj = np.empty((0, k))
        b_proj = np.empty(0)

    # Add non-negativity constraints: x >= 0, i.e., x0 + null_basis @ z >= 0
    # -null_basis @ z <= x0
    A_nonneg = -null_basis.T  # Each row is -null_basis[i, :]
    b_nonneg = x0

    # Combine all inequality constraints in projected space
    A_all = np.vstack([A_proj, A_nonneg.T]) if A_proj.size > 0 else A_nonneg.T
    b_all = np.concatenate([b_proj, b_nonneg]) if b_proj.size > 0 else b_nonneg

    if k == 1:
        # 1D case: find the interval
        # For each constraint a @ z <= b, the bound is z <= b/a (if a > 0) or z >= b/a (if a < 0)
        z_min, z_max = -np.inf, np.inf
        for i in range(len(A_all)):
            a, b = A_all[i, 0], b_all[i]
            if abs(a) < 1e-12:
                if b < -1e-9:
                    return None  # Infeasible
            elif a > 0:
                z_max = min(z_max, b / a)
            else:
                z_min = max(z_min, b / a)

        if z_min > z_max + 1e-9:
            return None  # Infeasible

        z_min = max(z_min, -1e6)  # Bound for numerical stability
        z_max = min(z_max, 1e6)

        vertices_z = np.array([[z_min], [z_max]])
        vertices = x0 + vertices_z @ null_basis.T
        return vertices

    # General case: use HalfspaceIntersection
    # Build halfspace representation: [a_1, ..., a_k, -b] for a @ z <= b
    halfspaces = []
    for i in range(len(A_all)):
        halfspaces.append(np.append(A_all[i], -b_all[i]))
    halfspaces = np.array(halfspaces)

    # Find interior point using Chebyshev center
    c = np.zeros(k + 1)
    c[-1] = -1  # Maximize slack t

    A_cheb = []
    b_cheb = []
    for i in range(len(A_all)):
        norm = np.linalg.norm(A_all[i])
        if norm > 1e-12:
            row = np.append(A_all[i] / norm, 1.0)
            A_cheb.append(row)
            b_cheb.append(b_all[i] / norm)

    if not A_cheb:
        return None

    A_cheb = np.array(A_cheb)
    b_cheb = np.array(b_cheb)

    bounds = [(None, None) for _ in range(k)] + [(0, None)]
    res = linprog(c, A_ub=A_cheb, b_ub=b_cheb, bounds=bounds, method='highs')

    if not res.success or res.x[-1] < 1e-10:
        return None

    interior = res.x[:k]

    try:
        hs = HalfspaceIntersection(halfspaces, interior)
        vertices_z = hs.intersections

        if len(vertices_z) > max_vertices:
            warnings.warn(
                f"Polytope has {len(vertices_z)} vertices (>{max_vertices}). "
                "Volume computation may be slow or inaccurate.",
                RuntimeWarning
            )

        # Lift back to original space: x = x0 + null_basis @ z
        vertices = x0 + vertices_z @ null_basis.T
        return vertices
    except Exception:
        return None


def _try_find_single_point(polytope: BeliefPolytope, tol: float = 1e-8) -> Optional[np.ndarray]:
    """Try to find a single feasible point when the polytope might be a point or empty.

    This is a fallback when vertex enumeration fails. It finds a feasible point
    and checks if the polytope is essentially a single point (diameter < tol).

    Parameters
    ----------
    polytope : BeliefPolytope
        The polytope to check
    tol : float
        Tolerance for considering the polytope as a single point

    Returns
    -------
    Optional[np.ndarray]
        Single point as (1, n) array if polytope is a point, None otherwise
    """
    n = polytope.n
    A_ub, b_ub, A_eq, b_eq, lb, ub = polytope.as_lp_constraints()
    bounds = list(zip(lb, ub))

    # First find any feasible point
    c = np.zeros(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return None  # Infeasible

    center = res.x

    # Check diameter by maximizing distance from center in a few directions
    # Use coordinate directions
    max_dist = 0.0
    directions = list(np.eye(n))

    for direction in directions:
        # Max in positive direction
        res_pos = linprog(-direction, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        if res_pos.success:
            dist = np.linalg.norm(res_pos.x - center)
            max_dist = max(max_dist, dist)

        # Max in negative direction
        res_neg = linprog(direction, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        if res_neg.success:
            dist = np.linalg.norm(res_neg.x - center)
            max_dist = max(max_dist, dist)

        if max_dist > tol:
            return None  # Not a single point

    # Polytope is essentially a single point
    return center.reshape(1, -1)


def _enumerate_vertices(polytope: BeliefPolytope,
                        max_vertices: int = 1000) -> Optional[np.ndarray]:
    """Enumerate vertices of the polytope.

    First tries the fast approach (projecting via simplex constraint only).
    If that fails (e.g., due to implicit equality constraints), falls back
    to the more robust approach that handles arbitrary equality constraints.

    Parameters
    ----------
    polytope : BeliefPolytope
        The polytope to enumerate vertices for
    max_vertices : int
        Maximum number of vertices before warning

    Returns
    -------
    Optional[np.ndarray]
        Array of vertices in n-dimensional space (each row is a vertex),
        or None if enumeration fails
    """
    n = polytope.n
    m = n - 1  # Projected dimension

    if m == 0:
        # 1D case: the only point is [1]
        return np.array([[1.0]])

    # First try the fast path (no additional equality constraints)
    interior = _find_interior_point_projected(polytope)
    if interior is not None:
        # Get projected constraints
        A_proj, d_proj = _project_constraints(polytope)

        # Build halfspace representation in (n-1) dimensions
        halfspaces = []

        if A_proj.size > 0:
            for i in range(A_proj.shape[0]):
                halfspaces.append(np.append(A_proj[i], -d_proj[i]))

        # Non-negativity
        for i in range(m):
            row = np.zeros(m + 1)
            row[i] = -1.0
            halfspaces.append(row)

        # sum(b') <= 1
        row = np.zeros(m + 1)
        row[:m] = 1.0
        row[-1] = -1.0
        halfspaces.append(row)

        halfspaces = np.array(halfspaces)

        try:
            hs = HalfspaceIntersection(halfspaces, interior)
            vertices_proj = hs.intersections

            if len(vertices_proj) > max_vertices:
                warnings.warn(
                    f"Polytope has {len(vertices_proj)} vertices (>{max_vertices}). "
                    "Volume computation may be slow or inaccurate.",
                    RuntimeWarning
                )

            vertices = np.zeros((len(vertices_proj), n))
            vertices[:, :-1] = vertices_proj
            vertices[:, -1] = 1.0 - np.sum(vertices_proj, axis=1)

            return vertices
        except Exception:
            pass

    # Fall back to the robust approach that handles equality constraints
    result = _enumerate_vertices_with_equalities(polytope, max_vertices)
    if result is not None:
        return result

    # Final fallback: try to find a single feasible point
    # This handles cases where the polytope is a single point due to
    # tight constraints that aren't detected as explicit equalities
    return _try_find_single_point(polytope)


def _find_affine_basis(points: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, int]:
    """Find an orthonormal basis for the affine subspace spanned by points.

    Uses SVD to find the effective dimension and an orthonormal basis for
    the affine subspace that the points lie in.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_points, ambient_dim) containing the points
    tol : float
        Tolerance for determining zero singular values

    Returns
    -------
    centroid : np.ndarray
        The centroid of the points (shape: ambient_dim)
    basis : np.ndarray
        Orthonormal basis for the tangent space (shape: ambient_dim x effective_dim)
    effective_dim : int
        The effective dimension of the affine subspace
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Use SVD to find the basis for the affine subspace
    # centered = U @ S @ Vt, where Vt rows are the principal directions
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Determine effective dimension by counting significant singular values
    effective_dim = np.sum(S > tol * S[0]) if len(S) > 0 and S[0] > tol else 0

    # The basis vectors are the first effective_dim rows of Vt (transposed to columns)
    basis = Vt[:effective_dim].T if effective_dim > 0 else np.empty((points.shape[1], 0))

    return centroid, basis, effective_dim


def _project_to_affine_subspace(points: np.ndarray, centroid: np.ndarray,
                                  basis: np.ndarray) -> np.ndarray:
    """Project points onto an affine subspace.

    Parameters
    ----------
    points : np.ndarray
        Points to project (shape: num_points x ambient_dim)
    centroid : np.ndarray
        Centroid of the affine subspace
    basis : np.ndarray
        Orthonormal basis for the tangent space (shape: ambient_dim x effective_dim)

    Returns
    -------
    np.ndarray
        Projected points in the reduced coordinate system (shape: num_points x effective_dim)
    """
    centered = points - centroid
    return centered @ basis


def compute_volume(polytope: BeliefPolytope) -> float:
    """Compute polytope volume using vertex enumeration and ConvexHull.

    Projects the polytope onto its intrinsic dimension using SVD to find
    the affine subspace the vertices lie in. This handles cases where the
    polytope is low-dimensional relative to the ambient space (e.g., due to
    equality constraints).

    Parameters
    ----------
    polytope : BeliefPolytope
        The belief polytope to compute volume for

    Returns
    -------
    float
        Volume as a fraction of the simplex volume (between 0 and 1).
        Returns 0.0 if volume computation fails.
    """
    n = polytope.n
    if n <= 1:
        return 1.0

    # Enumerate vertices
    vertices = _enumerate_vertices(polytope)
    if vertices is None:
        # Check if polytope is empty (infeasible)
        A_ub, b_ub, A_eq, b_eq, lb, ub = polytope.as_lp_constraints()
        res = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=list(zip(lb, ub)), method='highs')
        if not res.success:
            # Empty polytope - volume is 0
            return 0.0
        # Non-empty but enumeration failed (numerical issues in high dimensions)
        warnings.warn(
            "Could not enumerate polytope vertices (numerical issues in high dimensions). Returning 0.",
            RuntimeWarning
        )
        return 0.0

    # Remove duplicate vertices
    vertices = np.unique(vertices, axis=0)

    if len(vertices) < 2:
        # Single point or empty - zero volume (not an error, just degenerate)
        return 0.0

    # Find the affine subspace the vertices lie in
    centroid, basis, effective_dim = _find_affine_basis(vertices)

    if effective_dim == 0:
        # All vertices are the same point - zero volume
        return 0.0

    # Project vertices to the reduced coordinate system
    projected = _project_to_affine_subspace(vertices, centroid, basis)

    # Need at least effective_dim + 1 vertices to form a simplex
    if len(projected) < effective_dim + 1:
        return 0.0

    # Handle 1D case separately (ConvexHull doesn't work for 1D)
    if effective_dim == 1:
        raw_volume = np.max(projected) - np.min(projected)
    else:
        try:
            hull = ConvexHull(projected)
            raw_volume = hull.volume
        except Exception as e:
            warnings.warn(
                f"ConvexHull computation failed: {e}. Returning 0.",
                RuntimeWarning
            )
            return 0.0

    # Normalize by the volume of the simplex in the same dimension.
    #
    # The k-simplex embedded in R^{k+1} with vertices at coordinate axis endpoints
    # (i.e., standard probability simplex faces) has k-dimensional volume:
    #   sqrt(k+1) / k!
    #
    # This accounts for the "tilt" of the simplex relative to coordinate axes.
    # For example:
    #   - 1-simplex (line from (1,0) to (0,1)): length = sqrt(2) = sqrt(2)/1!
    #   - 2-simplex (triangle at (1,0,0), (0,1,0), (0,0,1)): area = sqrt(3)/2 = sqrt(3)/2!
    #   - 3-simplex (tetrahedron): volume = sqrt(4)/6 = 2/6 = 1/3 = sqrt(4)/3!
    def embedded_simplex_volume(k: int) -> float:
        """Volume of k-simplex with vertices at coordinate axis endpoints in R^{k+1}."""
        if k <= 0:
            return 1.0
        return math.sqrt(k + 1) / math.factorial(k)

    if effective_dim < n - 1:
        # For low-dimensional polytopes, normalize by the volume of the simplex
        # face in the same dimension (a k-face of the (n-1)-simplex)
        intrinsic_simplex_vol = embedded_simplex_volume(effective_dim)
        return raw_volume / intrinsic_simplex_vol if intrinsic_simplex_vol > 0 else 0.0

    # Full dimensional case: use the (n-1)-simplex volume
    simplex_vol = embedded_simplex_volume(n - 1)
    return raw_volume / simplex_vol if simplex_vol > 0 else 0.0
