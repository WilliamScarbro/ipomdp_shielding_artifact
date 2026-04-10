"""Interval Partially Observable Markov Decision Process model."""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable, Optional
import random
import numpy as np

from .pomdp import POMDP

State = Hashable
Action = Hashable
Observation = Hashable


@dataclass
class IPOMDP:
    """
    Interval Partially Observed Markov Decision Process.

    states       : list of states
    observations : list of possible observations
    actions      : list of actions
    T            : (s, a) -> {s' -> P(s' | s, a)}   (exact dynamics)
    P_lower      : s -> {o -> lower bound on P(o | s)}
    P_upper      : s -> {o -> upper bound on P(o | s)}

    For each state s, we assume:
        sum_o P_lower[s][o] <= 1 <= sum_o P_upper[s][o]
    """
    states: List[State]
    observations: List[Observation]
    actions: List[Action]
    T: Dict[Tuple[State, Action], Dict[State, float]]
    P_lower: Dict[State, Dict[Observation, float]]
    P_upper: Dict[State, Dict[Observation, float]]

    def _state_index(self) -> Dict[State, int]:
        """Return mapping from state to index."""
        return {s: i for i, s in enumerate(self.states)}

    def _build_T_matrix(self, action: Action) -> np.ndarray:
        """
        Returns T_a as an n x n matrix where [i,j] = P(s_j | s_i, a).
        """
        idx = self._state_index()
        n = len(self.states)
        Tmat = np.zeros((n, n), dtype=float)
        for s in self.states:
            row = self.T.get((s, action), {})
            i = idx[s]
            for sp, p in row.items():
                j = idx[sp]
                Tmat[i, j] = float(p)
        return Tmat

    def _obs_bounds_vectors(self, obs: Observation) -> Tuple[np.ndarray, np.ndarray]:
        """
        w_lo[j] = P_lower[s_j][obs], w_hi[j] = P_upper[s_j][obs]
        """
        n = len(self.states)
        w_lo = np.zeros(n, dtype=float)
        w_hi = np.zeros(n, dtype=float)
        for j, s in enumerate(self.states):
            w_lo[j] = float(self.P_lower[s].get(obs, 0.0))
            w_hi[j] = float(self.P_upper[s].get(obs, 0.0))
        return w_lo, w_hi

    def _compute_y_bounds(self, prior, action: Action, solver=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        y = T_a^T b. Compute per-component bounds y_j in [yl_j, yu_j] by LP over prior.
        """
        if solver is None:
            from ..Propagators.lfp_propagator import SciPyHiGHSSolver
            solver = SciPyHiGHSSolver()

        Tmat = self._build_T_matrix(action)
        TaT = Tmat.T
        n = prior.n
        assert n == len(self.states)

        A_ub, b_ub, A_eq, b_eq, lb, ub = prior.as_lp_constraints()

        yl = np.zeros(n, dtype=float)
        yu = np.zeros(n, dtype=float)
        for j in range(n):
            c = TaT[j, :]
            lo = solver.solve(c, A_ub, b_ub, A_eq, b_eq, lb, ub, sense="min")
            hi = solver.solve(c, A_ub, b_ub, A_eq, b_eq, lb, ub, sense="max")
            if lo.status != "optimal" or hi.status != "optimal":
                raise RuntimeError(f"Failed to bound y[{j}] under prior polytope")
            yl[j] = lo.obj
            yu[j] = hi.obj

        yl = np.clip(yl, 0.0, 1.0)
        yu = np.clip(yu, 0.0, 1.0)
        return yl, yu

    def feasible_unnormalized_posterior_polytope(
        self,
        prior,
        action: Action,
        obs: Observation,
        solver=None,
        eps_denom: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, slice]]:
        """
        Returns an LP outer-approximation of feasible tuples (b, y, w, x) where:

          b in prior polytope
          y = T_a^T b
          w_j in [P_lower[s_j][obs], P_upper[s_j][obs]]
          x_j = w_j * y_j   (McCormick relaxation)

        Variable vector:
            z = [b(0..n-1), y(0..n-1), w(0..n-1), x(0..n-1)]  in R^(4n)

        The returned constraints describe a polytope:
            A_ub z <= b_ub
            A_eq z  = b_eq
            lb <= z <= ub

        eps_denom (optional): if > 0, also enforce sum_j x_j >= eps_denom

        Returns:
            (A_ub, b_ub, A_eq, b_eq, lb, ub, slices)
        where slices tells you where b,y,w,x live in z.
        """
        if solver is None:
            from ..Propagators.lfp_propagator import SciPyHiGHSSolver
            solver = SciPyHiGHSSolver()

        n = len(self.states)
        assert prior.n == n

        # Index slices in z
        sb = slice(0, n)
        sy = slice(n, 2*n)
        sw = slice(2*n, 3*n)
        sx = slice(3*n, 4*n)
        slices = {"b": sb, "y": sy, "w": sw, "x": sx}

        # Prior constraints on b
        A_bub, b_bub, A_beq, b_beq, lb_b, ub_b = prior.as_lp_constraints()

        # Build y = T_a^T b equalities
        Tmat = self._build_T_matrix(action)
        TaT = Tmat.T
        A_y_eq = np.zeros((n, 4*n), dtype=float)
        b_y_eq = np.zeros(n, dtype=float)
        for j in range(n):
            A_y_eq[j, sy.start + j] = 1.0
            A_y_eq[j, sb] = -TaT[j, :]

        # Bounds for y (needed for McCormick)
        y_lo, y_hi = self._compute_y_bounds(prior, action, solver=solver)

        # Bounds for w from intervals
        w_lo, w_hi = self._obs_bounds_vectors(obs)

        # Variable bounds lb/ub for z
        lb = -np.inf * np.ones(4*n, dtype=float)
        ub = np.inf * np.ones(4*n, dtype=float)
        lb[sb] = lb_b
        ub[sb] = ub_b
        lb[sy] = y_lo
        ub[sy] = y_hi
        lb[sw] = w_lo
        ub[sw] = w_hi
        lb[sx] = 0.0
        ub[sx] = 1.0

        # McCormick envelope for each j: x_j = y_j * w_j
        A_mcc = np.zeros((4*n, 4*n), dtype=float)
        b_mcc = np.zeros(4*n, dtype=float)

        for j in range(n):
            yl, yu = float(y_lo[j]), float(y_hi[j])
            wl, wu = float(w_lo[j]), float(w_hi[j])

            r0 = 4*j

            # x - wl*y - yl*w >= -yl*wl  ->  -x + wl*y + yl*w <= yl*wl
            A_mcc[r0 + 0, sx.start + j] = -1.0
            A_mcc[r0 + 0, sy.start + j] = wl
            A_mcc[r0 + 0, sw.start + j] = yl
            b_mcc[r0 + 0] = yl * wl

            # x - wu*y - yu*w >= -yu*wu  ->  -x + wu*y + yu*w <= yu*wu
            A_mcc[r0 + 1, sx.start + j] = -1.0
            A_mcc[r0 + 1, sy.start + j] = wu
            A_mcc[r0 + 1, sw.start + j] = yu
            b_mcc[r0 + 1] = yu * wu

            # x - wl*y - yu*w <= -yu*wl
            A_mcc[r0 + 2, sx.start + j] = 1.0
            A_mcc[r0 + 2, sy.start + j] = -wl
            A_mcc[r0 + 2, sw.start + j] = -yu
            b_mcc[r0 + 2] = -yu * wl

            # x - wu*y - yl*w <= -yl*wu
            A_mcc[r0 + 3, sx.start + j] = 1.0
            A_mcc[r0 + 3, sy.start + j] = -wu
            A_mcc[r0 + 3, sw.start + j] = -yl
            b_mcc[r0 + 3] = -yl * wu

        # Assemble inequality constraints
        if A_bub is None or A_bub.size == 0:
            A_prior_ub = np.zeros((0, 4*n))
            b_prior_ub = np.zeros((0,))
        else:
            A_prior_ub = np.zeros((A_bub.shape[0], 4*n), dtype=float)
            A_prior_ub[:, sb] = A_bub
            b_prior_ub = b_bub.copy()

        A_ub_list = [A_prior_ub, A_mcc]
        b_ub_list = [b_prior_ub, b_mcc]

        if eps_denom > 0.0:
            row = np.zeros((1, 4*n), dtype=float)
            row[0, sx] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(np.array([-float(eps_denom)], dtype=float))

        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)

        # Assemble equality constraints
        if A_beq is None or A_beq.size == 0:
            A_prior_eq = np.zeros((0, 4*n))
            b_prior_eq = np.zeros((0,))
        else:
            A_prior_eq = np.zeros((A_beq.shape[0], 4*n), dtype=float)
            A_prior_eq[:, sb] = A_beq
            b_prior_eq = b_beq.copy()

        A_eq = np.vstack([A_prior_eq, A_y_eq])
        b_eq = np.concatenate([b_prior_eq, b_y_eq])

        return A_ub, b_ub, A_eq, b_eq, lb, ub, slices
    
    
    def evolve(self, state: State, action: Action) -> State:
        """Sample next state from IPOMDP (non-interval) dynamics."""
        out = self.T[(state, action)]
        weights = [out.get(s, 0.0) for s in self.states]
        return random.choices(self.states, weights=weights, k=1)[0]
    

    def to_pomdp(self) -> "POMDP":
        expected_realizations = {s:
         {o: (self.P_lower[s].get(o,0.0) + self.P_upper[s].get(o,0.0))/2 for o in self.observations}
         for s in self.states}

        return POMDP(
            self.states,
            self.observations,
            self.actions,
            self.T,
            expected_realizations)
    
