"""Covariance estimators for N assets from a T x N matrix of returns.

* ``sample_cov``: maximum-likelihood sample covariance (divides by T).
* ``ledoit_wolf``: Ledoit and Wolf (2004) linear shrinkage towards mu*I, closed-form intensity.
* ``rmt_clean``: Random-Matrix-Theory cleaning of the correlation matrix. Eigenvalues inside
  the Marchenko-Pastur bulk [(1-sqrt(q))^2, (1+sqrt(q))^2], q = N/T, are treated as noise and
  replaced by their mean (trace preserved); eigenvalues above the upper edge are kept.
"""

from __future__ import annotations

import numpy as np


def _centered(returns: np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("returns must be a T x N matrix with T >= 3")
    return x - x.mean(axis=0)


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Sample covariance with the 1/T normalisation."""
    x = _centered(returns)
    return x.T @ x / x.shape[0]


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage towards ``mu * I``. Returns (covariance, intensity)."""
    x = _centered(returns)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = np.sum((s - mu * np.eye(n)) ** 2) / n
    b2_bar = (np.sum(np.sum(x**2, axis=1) ** 2) - t * np.sum(s**2)) / (t**2 * n)
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0 else b2 / d2
    return delta * mu * np.eye(n) + (1.0 - delta) * s, float(delta)


def marchenko_pastur_edges(n_assets: int, n_obs: int) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur support for a unit-variance correlation matrix."""
    q = n_assets / n_obs
    return (1.0 - np.sqrt(q)) ** 2, (1.0 + np.sqrt(q)) ** 2


def rmt_clean(returns: np.ndarray) -> tuple[np.ndarray, int]:
    """RMT-cleaned covariance. Returns (covariance, number of eigenvalues kept as signal)."""
    x = _centered(returns)
    t, n = x.shape
    s = x.T @ x / t
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    vals, vecs = np.linalg.eigh(corr)
    _, upper = marchenko_pastur_edges(n, t)
    noise = vals <= upper
    if noise.any():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    cleaned = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)
    return cleaned * np.outer(sd, sd), int((~noise).sum())
