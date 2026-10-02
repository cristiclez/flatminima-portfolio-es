"""Filtered historical simulation (FHS): one-step-ahead VaR and Expected Shortfall.

Barone-Adesi, Giannopoulos and Vosper (1999). Standardised residuals z_t = (r_t - mu)/sigma_t
from a fitted GARCH-type model are resampled (here: used empirically) and rescaled by the
one-step-ahead volatility forecast:

    VaR_a = -(mu + sigma_{T+1} * q_{1-a}(z))
    ES_a  = -(mu + sigma_{T+1} * E[z | z <= q_{1-a}(z)])

Risk figures are reported as POSITIVE losses on a unit-notional position (log returns are
used as the P&L proxy).
"""

from __future__ import annotations

import numpy as np


def fhs_var_es(z: np.ndarray, mu: float, sigma_next: float, level: float) -> tuple[float, float]:
    """FHS VaR and ES at confidence ``level`` (e.g. 0.99 or 0.975). Returns (VaR, ES)."""
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
    z = np.asarray(z, dtype=float)
    q = float(np.quantile(z, 1.0 - level))
    tail = z[z <= q]
    var = -(mu + sigma_next * q)
    es = -(mu + sigma_next * float(tail.mean()))
    return var, es


def historical_var_es(r: np.ndarray, level: float) -> tuple[float, float]:
    """Plain (unfiltered) historical-simulation VaR and ES over a window of returns.

    Used as a baseline: it ignores volatility clustering, which is exactly what FHS adds.
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
    r = np.asarray(r, dtype=float)
    q = float(np.quantile(r, 1.0 - level))
    return -q, -float(r[r <= q].mean())
