"""Convex long-only baselines (cvxpy): minimum variance and minimum CVaR.

All portfolios satisfy sum(w) = 1 and w >= 0.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def equal_weight(n: int) -> np.ndarray:
    """1/N portfolio."""
    return np.full(n, 1.0 / n)


def _solve(prob: cp.Problem) -> None:
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status: {prob.status}")


def min_variance(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights for covariance ``cov``."""
    n = cov.shape[0]
    c = 0.5 * (cov + cov.T) + 1e-12 * np.trace(cov) / n * np.eye(n)
    w = cp.Variable(n)
    _solve(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(c))), [cp.sum(w) == 1, w >= 0]))
    ww = np.clip(w.value, 0.0, None)
    return ww / ww.sum()


def min_cvar(returns: np.ndarray, level: float = 0.975) -> np.ndarray:
    """Long-only minimum-CVaR weights on historical scenarios (Rockafellar-Uryasev).

    min_{w,a,u}  a + 1/((1-level) T) sum_t u_t   s.t.  u_t >= -r_t'w - a,  u >= 0.
    """
    r = np.asarray(returns, dtype=float)
    t, n = r.shape
    w, a, u = cp.Variable(n), cp.Variable(), cp.Variable(t)
    obj = a + cp.sum(u) / ((1.0 - level) * t)
    _solve(cp.Problem(cp.Minimize(obj), [u >= -r @ w - a, u >= 0, cp.sum(w) == 1, w >= 0]))
    ww = np.clip(w.value, 0.0, None)
    return ww / ww.sum()
