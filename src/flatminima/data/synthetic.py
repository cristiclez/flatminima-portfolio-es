"""Synthetic data with KNOWN ground truth. Always labelled as synthetic downstream.

Used to (i) run the whole pipeline offline and (ii) validate estimators and backtests:
a correct GARCH fit must recover the true parameters, and a correct backtest must not
reject a correctly specified model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_garch_t(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    rng: np.random.Generator,
    burn: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GARCH(1,1) with unit-variance Student-t innovations.

    r_t = sigma_t z_t,  sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2.
    Returns (returns, conditional_sigma), both of length ``n``.
    """
    if not (alpha >= 0 and beta >= 0 and alpha + beta < 1 and nu > 2 and omega > 0):
        raise ValueError("need omega>0, alpha,beta>=0, alpha+beta<1, nu>2")
    total = n + burn
    z = rng.standard_t(nu, size=total) * np.sqrt((nu - 2.0) / nu)
    r = np.empty(total)
    s2 = np.empty(total)
    s2[0] = omega / (1.0 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, total):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:], np.sqrt(s2[burn:])


def simulate_panel(
    tickers: list[str],
    n: int,
    start: str,
    seed: int,
    daily_vol: float = 0.01,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float = 6.0,
    factor_weight: float = 0.5,
) -> pd.DataFrame:
    """Long-format synthetic prices: GARCH-t assets sharing one common GARCH-t factor.

    Cross-sectional correlation is roughly ``factor_weight``; marginals are approximately
    (not exactly) Student-t. Intended for pipeline demos, not for estimator validation
    (use :func:`simulate_garch_t` for that).
    """
    rng = np.random.default_rng(seed)
    omega = daily_vol**2 * (1.0 - alpha - beta)
    factor, _ = simulate_garch_t(n, omega, alpha, beta, nu, rng)
    dates = pd.bdate_range(start, periods=n + 1)
    frames = []
    for tk in tickers:
        idio, _ = simulate_garch_t(n, omega, alpha, beta, nu, rng)
        r = np.sqrt(factor_weight) * factor + np.sqrt(1.0 - factor_weight) * idio
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": tk,
                    "date": dates,
                    "adj_close": prices,
                    "volume": rng.integers(1_000_000, 5_000_000, size=n + 1).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
