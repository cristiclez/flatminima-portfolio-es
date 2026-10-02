"""GARCH(1,1) and GJR-GARCH(1,1) with Student-t innovations, estimated by exact MLE.

Model (e_t = r_t - mu, z_t i.i.d. Student-t(nu) scaled to unit variance):

    r_t = mu + sigma_t z_t
    sigma_t^2 = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e_{t-1}^2 + beta * sigma_{t-1}^2

``gamma = 0`` recovers plain GARCH. The variance recursion is a linear filter in the
squared residuals, so it is evaluated with ``scipy.signal.lfilter`` (no Python loop).
Optimisation uses SLSQP with box bounds and the covariance-stationarity constraint
alpha + beta + gamma/2 < 1. Standard errors come from the numerical *observed* Hessian
of the negative log-likelihood.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

PERSISTENCE_CAP = 0.9999
_NAMES = ("mu", "omega", "alpha", "gamma", "beta", "nu")


@dataclass(frozen=True)
class GarchParams:
    """Parameters in the units of the return series passed to the estimator."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + self.gamma / 2.0

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def as_array(self) -> np.ndarray:
        return np.array([self.mu, self.omega, self.alpha, self.gamma, self.beta, self.nu])

    @classmethod
    def from_array(cls, a: np.ndarray) -> GarchParams:
        return cls(*[float(x) for x in a])


@dataclass(frozen=True)
class GarchFit:
    """Result of :func:`fit_garch`."""

    model: str
    params: GarchParams
    loglik: float
    n_obs: int
    converged: bool
    std_errors: dict[str, float]
    hessian_condition_number: float

    @property
    def n_free_params(self) -> int:
        return 6 if self.model == "gjr" else 5

    @property
    def aic(self) -> float:
        return 2 * self.n_free_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_free_params * np.log(self.n_obs) - 2 * self.loglik


def conditional_variance(
    params: GarchParams, r: np.ndarray, init_var: float | None = None
) -> tuple[np.ndarray, float]:
    """Filtered conditional variances s2[0..n-1] and the one-step-ahead forecast s2[n].

    The recursion is initialised at ``init_var`` (default: sample variance of ``r``).
    """
    r = np.asarray(r, dtype=float)
    e = r - params.mu
    s0 = float(np.var(r)) if init_var is None else float(init_var)
    drive = np.empty(len(r))
    drive[0] = s0
    ind = (e[:-1] < 0).astype(float)
    drive[1:] = params.omega + (params.alpha + params.gamma * ind) * e[:-1] ** 2
    s2 = lfilter([1.0], [1.0, -params.beta], drive)
    last = params.alpha + params.gamma * float(e[-1] < 0)
    s2_next = params.omega + last * e[-1] ** 2 + params.beta * s2[-1]
    return s2, float(s2_next)


def _student_t_ll(e: np.ndarray, s2: np.ndarray, nu: float) -> np.ndarray:
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return c - 0.5 * np.log(s2) - 0.5 * (nu + 1) * np.log1p(e**2 / (s2 * (nu - 2)))


def log_likelihood(params: GarchParams, r: np.ndarray) -> float:
    """Exact Gaussian-free Student-t log-likelihood of ``r`` under ``params``."""
    r = np.asarray(r, dtype=float)
    s2, _ = conditional_variance(params, r)
    return float(np.sum(_student_t_ll(r - params.mu, s2, params.nu)))


def _full_theta(free: np.ndarray, model: str) -> np.ndarray:
    if model == "gjr":
        return free
    return np.insert(free, 3, 0.0)  # gamma = 0


def _nll(free: np.ndarray, r: np.ndarray, model: str) -> float:
    theta = _full_theta(free, model)
    p = GarchParams.from_array(theta)
    if p.omega <= 0 or p.nu <= 2:
        return 1e12
    val = -log_likelihood(p, r)
    return val if np.isfinite(val) else 1e12


def _numerical_hessian(f, x: np.ndarray) -> np.ndarray:
    n = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (
                    f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
                ) / (4 * h[i] * h[j])
    return H


def fit_garch(
    returns: np.ndarray,
    model: str = "gjr",
    scale: float = 100.0,
    start: GarchParams | None = None,
    compute_se: bool = True,
) -> GarchFit:
    """Fit GARCH(1,1) (``model='garch'``) or GJR-GARCH(1,1) (``model='gjr'``) by MLE.

    Returns are rescaled by ``scale`` internally for numerical conditioning and parameters
    are mapped back to the original units. ``start`` allows a warm start (used by the
    rolling engine). Std. errors use the observed Hessian; they are NaN if it is not
    positive definite (e.g. a parameter on a bound).
    """
    if model not in ("garch", "gjr"):
        raise ValueError("model must be 'garch' or 'gjr'")
    r = np.asarray(returns, dtype=float)
    if len(r) < 200:
        raise ValueError("need at least 200 observations")
    x = r * scale
    var = float(np.var(x))

    def pack(p: GarchParams) -> np.ndarray:
        a = np.array([p.mu * scale, p.omega * scale**2, p.alpha, p.gamma, p.beta, p.nu])
        return a if model == "gjr" else np.delete(a, 3)

    starts = []
    if start is not None:
        starts.append(pack(start))
    for a0, b0 in ((0.05, 0.90), (0.10, 0.80)):
        g0 = 0.05 if model == "gjr" else 0.0
        omega0 = var * (1 - a0 - b0 - g0 / 2)
        s = [float(np.mean(x)), omega0, a0, g0, b0, 8.0]
        starts.append(np.array(s) if model == "gjr" else np.delete(np.array(s), 3))

    if model == "gjr":
        bounds = [(None, None), (1e-8, None), (0, 1), (0, 1), (0, 1), (2.05, 200)]
        cons = [{"type": "ineq", "fun": lambda t: PERSISTENCE_CAP - (t[2] + t[4] + t[3] / 2)}]
    else:
        bounds = [(None, None), (1e-8, None), (0, 1), (0, 1), (2.05, 200)]
        cons = [{"type": "ineq", "fun": lambda t: PERSISTENCE_CAP - (t[2] + t[3])}]

    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s0 in starts:
            res = minimize(
                _nll,
                s0,
                args=(x, model),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res

    theta = _full_theta(best.x, model)
    p_scaled = GarchParams.from_array(theta)
    params = GarchParams(
        mu=p_scaled.mu / scale,
        omega=p_scaled.omega / scale**2,
        alpha=p_scaled.alpha,
        gamma=p_scaled.gamma,
        beta=p_scaled.beta,
        nu=p_scaled.nu,
    )

    se: dict[str, float] = {}
    cond = float("nan")
    names = [n for n in _NAMES if model == "gjr" or n != "gamma"]
    if compute_se:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = _numerical_hessian(lambda t: _nll(t, x, model), best.x)
        try:
            eig = np.linalg.eigvalsh(H)
            cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("nan")
            cov = np.linalg.inv(H)
            diag = np.diag(cov)
            raw = np.sqrt(np.where(diag > 0, diag, np.nan))
        except np.linalg.LinAlgError:
            raw = np.full(len(names), np.nan)
        unit = {"mu": 1 / scale, "omega": 1 / scale**2}
        se = {n: float(v * unit.get(n, 1.0)) for n, v in zip(names, raw, strict=True)}

    return GarchFit(
        model=model,
        params=params,
        loglik=log_likelihood(params, r),
        n_obs=len(r),
        converged=bool(best.success),
        std_errors=se,
        hessian_condition_number=cond,
    )
