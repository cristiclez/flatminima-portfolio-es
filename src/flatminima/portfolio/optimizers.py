"""Convex long-only baselines (cvxpy): minimum variance and minimum CVaR.

All portfolios satisfy sum(w) = 1 and w >= 0, and optionally w <= cap (a per-asset limit; it
must satisfy cap >= 1/N for the problem to be feasible).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def equal_weight(n: int) -> np.ndarray:
    """1/N portfolio."""
    return np.full(n, 1.0 / n)


def _check_cap(cap: float | None, n: int) -> None:
    if cap is not None and cap * n < 1.0 - 1e-12:
        raise ValueError(f"cap={cap} is infeasible for {n} assets (need cap >= 1/N)")


def _finish(w: np.ndarray, cap: float | None) -> np.ndarray:
    ww = np.clip(w, 0.0, cap)
    return ww / ww.sum()


def _solve(prob: cp.Problem) -> None:
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status: {prob.status}")


def min_variance(cov: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Long-only minimum-variance weights for covariance ``cov`` (optional per-asset ``cap``)."""
    n = cov.shape[0]
    _check_cap(cap, n)
    c = 0.5 * (cov + cov.T) + 1e-12 * np.trace(cov) / n * np.eye(n)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0] + ([w <= cap] if cap is not None else [])
    _solve(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(c))), cons))
    return _finish(w.value, cap)


def min_cvar(returns: np.ndarray, level: float = 0.975, cap: float | None = None) -> np.ndarray:
    """Long-only minimum-CVaR weights on historical scenarios (Rockafellar-Uryasev).

    min_{w,a,u}  a + 1/((1-level) T) sum_t u_t   s.t.  u_t >= -r_t'w - a,  u >= 0.
    """
    r = np.asarray(returns, dtype=float)
    t, n = r.shape
    _check_cap(cap, n)
    w, a, u = cp.Variable(n), cp.Variable(), cp.Variable(t)
    obj = a + cp.sum(u) / ((1.0 - level) * t)
    cons = [u >= -r @ w - a, u >= 0, cp.sum(w) == 1, w >= 0] + ([w <= cap] if cap else [])
    _solve(cp.Problem(cp.Minimize(obj), cons))
    return _finish(w.value, cap)
