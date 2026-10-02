"""Rolling out-of-sample VaR/ES engine.

For each forecast date t (strictly after the first ``window`` observations):
  1. the model is (re)fitted on returns [t-window, t-1] every ``refit_every`` days (warm start);
  2. between refits the parameters are frozen but the variance filter and the standardised
     residuals are recomputed on the latest window, so forecasts use information up to t-1;
  3. FHS VaR/ES for day t are produced and stored next to the realised return.

No information from day t or later enters the forecast for day t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flatminima.risk.fhs import fhs_var_es, historical_var_es
from flatminima.risk.garch import GarchParams, conditional_variance, fit_garch


def rolling_risk(
    returns: pd.Series,
    window: int = 1000,
    refit_every: int = 20,
    model: str = "gjr",
    var_level: float = 0.99,
    es_level: float = 0.975,
) -> pd.DataFrame:
    """Rolling one-step-ahead VaR (``var_level`` and ``es_level``) and ES (``es_level``).

    Output columns: ret, sigma, var_99, var_975, es_975 (names reflect the default levels),
    and the unfiltered baselines hs_var_99, hs_es_975. Losses are positive numbers.
    ``DataFrame.attrs`` records the number of refits and failed refits.
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n <= window:
        raise ValueError("series shorter than the estimation window")

    params: GarchParams | None = None
    n_refits = n_failed = 0
    rows = []
    for t in range(window, n):
        w = r[t - window : t]
        if params is None or (t - window) % refit_every == 0:
            fit = fit_garch(w, model=model, start=params, compute_se=False)
            n_refits += 1
            if fit.converged or params is None:
                params = fit.params
            else:
                n_failed += 1  # keep the previous parameters
        s2, s2_next = conditional_variance(params, w)
        z = (w - params.mu) / np.sqrt(s2)
        sig = float(np.sqrt(s2_next))
        v_hi, _ = fhs_var_es(z, params.mu, sig, var_level)
        v_es, es = fhs_var_es(z, params.mu, sig, es_level)
        hv, _ = historical_var_es(w, var_level)
        _, hes = historical_var_es(w, es_level)
        rows.append((r[t], sig, v_hi, v_es, es, hv, hes))

    out = pd.DataFrame(
        rows,
        index=returns.index[window:],
        columns=["ret", "sigma", "var_99", "var_975", "es_975", "hs_var_99", "hs_es_975"],
    )
    out.attrs.update(n_refits=n_refits, n_failed_refits=n_failed, model=model, window=window)
    return out
