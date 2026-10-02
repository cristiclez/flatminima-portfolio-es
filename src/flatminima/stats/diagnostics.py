"""Residual diagnostics: Ljung-Box, ARCH-LM and Jarque-Bera (implemented from first principles)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ljung_box(x: np.ndarray, lags: int = 10, ddof: int = 0) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value for autocorrelation up to ``lags``.

    Q = n(n+2) sum_{k=1}^{m} rho_k^2 / (n-k), asymptotically chi2(m - ddof) under H0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lags >= n or lags - ddof < 1:
        raise ValueError("need lags < n and lags > ddof")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    rho = np.array([np.dot(xc[k:], xc[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * float(np.sum(rho**2 / (n - np.arange(1, lags + 1))))
    return q, float(stats.chi2.sf(q, lags - ddof))


def arch_lm(z: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Engle's ARCH-LM test: n * R^2 of the regression of z^2 on its own ``lags`` lags."""
    z2 = np.asarray(z, dtype=float) ** 2
    n = len(z2)
    if lags >= n - 1:
        raise ValueError("lags too large")
    y = z2[lags:]
    X = np.column_stack([np.ones(len(y))] + [z2[lags - k : n - k] for k in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot
    stat = len(y) * r2
    return stat, float(stats.chi2.sf(stat, lags))


def jarque_bera(x: np.ndarray) -> tuple[float, float]:
    """Jarque-Bera normality test (statistic, p-value)."""
    res = stats.jarque_bera(np.asarray(x, dtype=float))
    return float(res.statistic), float(res.pvalue)


def residual_diagnostics(z: np.ndarray, lags: int = 10) -> pd.DataFrame:
    """Standard battery on standardised residuals ``z``.

    Under a correctly specified model: no autocorrelation in z (Ljung-Box), no remaining
    ARCH effects (Ljung-Box on z^2 and ARCH-LM). Jarque-Bera is reported to document the
    departure from normality that motivates Student-t innovations.
    """
    z = np.asarray(z, dtype=float)
    rows = []
    for name, (stat, p) in {
        f"Ljung-Box z (lags={lags})": ljung_box(z, lags),
        f"Ljung-Box z^2 (lags={lags})": ljung_box(z**2, lags),
        f"ARCH-LM (lags={lags})": arch_lm(z, lags),
        "Jarque-Bera z": jarque_bera(z),
    }.items():
        rows.append({"test": name, "statistic": stat, "p_value": p})
    return pd.DataFrame(rows)
