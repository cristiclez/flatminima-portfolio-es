"""Walk-forward evaluation of long-only portfolio strategies.

At each rebalance date t the strategy sees ONLY returns [t-lookback, t-1], sets target weights,
and holds them (drifting with realised returns) until the next rebalance. Portfolio returns are
simple returns; trading costs are modelled explicitly as ``cost_bps`` per unit of traded
notional (sum |w_target - w_drifted|) and deducted on the rebalance day.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flatminima.covariance.estimators import ledoit_wolf, rmt_clean, sample_cov
from flatminima.portfolio.optimizers import equal_weight, min_cvar, min_variance

Strategy = Callable[[np.ndarray], np.ndarray]  # window of log returns (L x N) -> weights (N,)

DEFAULT_STRATEGIES: dict[str, Strategy] = {
    "EqualWeight": lambda r: equal_weight(r.shape[1]),
    "MinVar-Sample": lambda r: min_variance(sample_cov(r)),
    "MinVar-LedoitWolf": lambda r: min_variance(ledoit_wolf(r)[0]),
    "MinVar-RMT": lambda r: min_variance(rmt_clean(r)[0]),
    "MinCVaR-97.5": lambda r: min_cvar(r, 0.975),
}


@dataclass
class WalkForwardResult:
    """Output of :func:`walk_forward`."""

    returns_net: pd.DataFrame
    returns_gross: pd.DataFrame
    turnover: pd.DataFrame
    weights: dict[str, np.ndarray]
    tickers: list[str]
    lookback: int
    rebalance_every: int
    cost_bps: float


def walk_forward(
    log_returns: pd.DataFrame,
    strategies: dict[str, Strategy] | None = None,
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
) -> WalkForwardResult:
    """Run every strategy over the out-of-sample period (from index ``lookback`` onwards)."""
    strategies = strategies or DEFAULT_STRATEGIES
    r = log_returns.to_numpy(dtype=float)
    simple = np.expm1(r)
    t_total, n = r.shape
    if t_total <= lookback + rebalance_every:
        raise ValueError("series too short for the chosen lookback")
    idx = log_returns.index[lookback:]
    n_oos = t_total - lookback
    net = {k: np.empty(n_oos) for k in strategies}
    gross = {k: np.empty(n_oos) for k in strategies}
    turn: dict[str, list[float]] = {k: [] for k in strategies}
    wh: dict[str, list[np.ndarray]] = {k: [] for k in strategies}
    reb_dates: list[pd.Timestamp] = []
    cur = {k: np.full(n, 1.0 / n) for k in strategies}

    for i, t in enumerate(range(lookback, t_total)):
        rebalance = i % rebalance_every == 0
        if rebalance:
            reb_dates.append(log_returns.index[t])
        for k, fn in strategies.items():
            cost = 0.0
            if rebalance:
                target = np.asarray(fn(r[t - lookback : t]), dtype=float)
                tv = float(np.abs(target - cur[k]).sum()) if i > 0 else float(target.sum())
                cost = cost_bps / 1e4 * tv
                turn[k].append(tv)
                wh[k].append(target)
                cur[k] = target
            g = float(cur[k] @ simple[t])
            gross[k][i], net[k][i] = g, g - cost
            drifted = cur[k] * (1.0 + simple[t])
            cur[k] = drifted / drifted.sum()

    return WalkForwardResult(
        returns_net=pd.DataFrame(net, index=idx),
        returns_gross=pd.DataFrame(gross, index=idx),
        turnover=pd.DataFrame(turn, index=reb_dates),
        weights={k: np.vstack(v) for k, v in wh.items()},
        tickers=list(log_returns.columns),
        lookback=lookback,
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
    )
