"""Check the two-state LP example from the supplement.

The script verifies three things:
  1. The row-stochastic projection of the raw observation intervals.
  2. The exact Bayes posterior extrema over the projected intervals.
  3. The Charnes-Cooper LP optimum built from the same matrices as the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9


@dataclass(frozen=True)
class ExactResult:
    value: float
    b1: float
    w1: float
    w2: float
    eta: float
    posterior: np.ndarray


def project_received_symbol_bounds(
    lower: np.ndarray, upper: np.ndarray, received_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project row-stochastic observation rows onto one received symbol."""
    lo_eff = []
    hi_eff = []
    for s in range(lower.shape[0]):
        other = [o for o in range(lower.shape[1]) if o != received_obs]
        lo = max(lower[s, received_obs], 1.0 - float(np.sum(upper[s, other])))
        hi = min(upper[s, received_obs], 1.0 - float(np.sum(lower[s, other])))
        lo_eff.append(lo)
        hi_eff.append(hi)
    return np.array(lo_eff), np.array(hi_eff)


def posterior_state1(b1: float, w1: float, w2: float) -> tuple[float, float, np.ndarray]:
    """Return eta, b'_1, and the full two-state posterior."""
    u = np.array([b1 * w1, (1.0 - b1) * w2], dtype=float)
    eta = float(np.sum(u))
    posterior = u / eta
    return eta, float(posterior[0]), posterior


def exact_corner_extrema(
    b1_bounds: tuple[float, float], w_lo: np.ndarray, w_hi: np.ndarray
) -> tuple[ExactResult, ExactResult]:
    """Enumerate the corners of the exact two-state feasible box."""
    results = []
    for b1, w1, w2 in product(b1_bounds, (w_lo[0], w_hi[0]), (w_lo[1], w_hi[1])):
        eta, value, posterior = posterior_state1(b1, w1, w2)
        results.append(ExactResult(value, b1, w1, w2, eta, posterior))
    return min(results, key=lambda r: r.value), max(results, key=lambda r: r.value)


def build_unnormalized_lp(w_lo: np.ndarray, w_hi: np.ndarray):
    """Build G0, h0, E0, f0, lower z, and upper z for z=(b,y,w,u)."""
    n = 2
    transition = np.eye(n)

    # Current envelope: 0.4 <= b_1 <= 0.6.
    a_t = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d_t = np.array([0.6, -0.4])

    # Identity dynamics makes y=b, hence y_1,y_2 in [0.4,0.6].
    y_lo = np.array([0.4, 0.4])
    y_hi = np.array([0.6, 0.6])

    e0 = np.block(
        [
            [
                np.ones((1, n)),
                np.zeros((1, n)),
                np.zeros((1, n)),
                np.zeros((1, n)),
            ],
            [
                -transition.T,
                np.eye(n),
                np.zeros((n, n)),
                np.zeros((n, n)),
            ],
        ]
    )
    f0 = np.array([1.0, 0.0, 0.0])

    zeros = np.zeros((n, n))
    dw_lo = np.diag(w_lo)
    dw_hi = np.diag(w_hi)
    dy_lo = np.diag(y_lo)
    dy_hi = np.diag(y_hi)
    eye = np.eye(n)

    g0 = np.vstack(
        [
            np.hstack([a_t, zeros, zeros, zeros]),
            np.hstack([zeros, dw_lo, dy_lo, -eye]),
            np.hstack([zeros, dw_hi, dy_hi, -eye]),
            np.hstack([zeros, -dw_lo, -dy_hi, eye]),
            np.hstack([zeros, -dw_hi, -dy_lo, eye]),
        ]
    )
    h0 = np.concatenate(
        [
            d_t,
            y_lo * w_lo,
            y_hi * w_hi,
            -y_hi * w_lo,
            -y_lo * w_hi,
        ]
    )

    z_lo = np.concatenate([np.zeros(n), y_lo, w_lo, np.zeros(n)])
    z_hi = np.concatenate([np.ones(n), y_hi, w_hi, np.ones(n)])
    return g0, h0, e0, f0, z_lo, z_hi


def solve_charnes_cooper(direction: np.ndarray, w_lo: np.ndarray, w_hi: np.ndarray):
    """Maximize direction^T b' using the Charnes-Cooper LP."""
    g0, h0, e0, f0, z_lo, z_hi = build_unnormalized_lp(w_lo, w_hi)
    dim_z = z_lo.size

    # Variables are x=(tilde z,t), with tilde z=t*z and t=1/(1^T u).
    a_ub = np.vstack(
        [
            np.hstack([g0, -h0[:, None]]),
            np.hstack([np.eye(dim_z), -z_hi[:, None]]),
            np.hstack([-np.eye(dim_z), z_lo[:, None]]),
        ]
    )
    b_ub = np.zeros(a_ub.shape[0])

    den = np.zeros(dim_z)
    den[6:8] = 1.0
    a_eq = np.vstack(
        [
            np.hstack([e0, -f0[:, None]]),
            np.concatenate([den, [0.0]])[None, :],
        ]
    )
    b_eq = np.concatenate([np.zeros(e0.shape[0]), [1.0]])

    objective = np.zeros(dim_z + 1)
    objective[6:8] = direction
    bounds = [(-np.inf, np.inf)] * dim_z + [(0.0, np.inf)]

    result = linprog(
        -objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    opt_value = -float(result.fun)
    tilde_z = result.x[:dim_z]
    t = float(result.x[-1])
    z = tilde_z / t
    posterior = z[6:8] / np.sum(z[6:8])
    return opt_value, t, z, posterior


def check_close(name: str, actual: float, expected: float) -> None:
    if abs(actual - expected) > TOL:
        raise AssertionError(f"{name}: got {actual:.12f}, expected {expected:.12f}")


def main() -> None:
    raw_lower = np.array(
        [
            [0.65, 0.10],
            [0.10, 0.60],
        ]
    )
    raw_upper = np.array(
        [
            [0.95, 0.30],
            [0.50, 0.80],
        ]
    )

    w_lo, w_hi = project_received_symbol_bounds(raw_lower, raw_upper, received_obs=0)
    expected_lo = np.array([0.7, 0.2])
    expected_hi = np.array([0.9, 0.4])
    np.testing.assert_allclose(w_lo, expected_lo, atol=TOL)
    np.testing.assert_allclose(w_hi, expected_hi, atol=TOL)

    exact_min, exact_max = exact_corner_extrema((0.4, 0.6), w_lo, w_hi)
    check_close("exact min b'_1", exact_min.value, float(Fraction(7, 13)))
    check_close("exact max b'_1", exact_max.value, float(Fraction(27, 31)))

    lp_max, t_max, _, lp_max_posterior = solve_charnes_cooper(
        np.array([1.0, 0.0]), w_lo, w_hi
    )
    lp_neg_min, t_min, _, lp_min_posterior = solve_charnes_cooper(
        np.array([-1.0, 0.0]), w_lo, w_hi
    )
    lp_min = -lp_neg_min

    check_close("LP min b'_1", lp_min, float(Fraction(7, 13)))
    check_close("LP max b'_1", lp_max, float(Fraction(27, 31)))
    check_close("LP min t", t_min, float(Fraction(25, 13)))
    check_close("LP max t", t_max, float(Fraction(50, 31)))

    print("row-stochastic projection")
    print(f"  w lower = {w_lo}")
    print(f"  w upper = {w_hi}")
    print()
    print("exact corner extrema")
    print(
        "  min b'_1 = "
        f"{exact_min.value:.12f} = 7/13, "
        f"at b1={exact_min.b1}, w1={exact_min.w1}, w2={exact_min.w2}, "
        f"eta={exact_min.eta:.12f}, b'={exact_min.posterior}"
    )
    print(
        "  max b'_1 = "
        f"{exact_max.value:.12f} = 27/31, "
        f"at b1={exact_max.b1}, w1={exact_max.w1}, w2={exact_max.w2}, "
        f"eta={exact_max.eta:.12f}, b'={exact_max.posterior}"
    )
    print()
    print("Charnes-Cooper LP")
    print(f"  max -b'_1 gives min b'_1 = {lp_min:.12f}, t={t_min:.12f}, b'={lp_min_posterior}")
    print(f"  max  b'_1 gives max b'_1 = {lp_max:.12f}, t={t_max:.12f}, b'={lp_max_posterior}")
    print()
    print("all checks passed")


if __name__ == "__main__":
    main()
