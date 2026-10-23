"""Performance statistics and Moving Block Bootstrap (MBB) confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flatminima.portfolio.walkforward import WalkForwardResult

ANN = 252


def sharpe(r: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Annualised Sharpe ratio (mean over std, no risk-free rate)."""
    r = np.asarray(r, dtype=float)
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1) * np.sqrt(ANN)


def empirical_cvar(r: np.ndarray, level: float = 0.975) -> float:
    """Mean loss over the worst ``1 - level`` share of days (losses are positive numbers)."""
    losses = -np.asarray(r, dtype=float)
    q = np.quantile(losses, level)
    return float(losses[losses >= q].mean())


def max_drawdown(r: np.ndarray) -> float:
    """Maximum peak-to-trough loss of cumulative wealth (negative number)."""
    wealth = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def moving_block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (n_boot x n) of Moving Block Bootstrap resamples (Kunsch, 1989).

    Overlapping blocks of length ``block`` start uniformly in [0, n - block]; ceil(n / block)
    blocks are concatenated and truncated to length n.
    """
    if not 1 <= block <= n:
        raise ValueError("need 1 <= block <= n")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def default_block_length(n: int) -> int:
    """Rule of thumb of order n^(1/3) (about 17 for ten-plus years of daily data)."""
    return max(1, int(round(n ** (1.0 / 3.0))))


def mbb_sharpe_ci(
    returns: pd.DataFrame,
    benchmark: str | None = None,
    block: int | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    batch: int = 250,
) -> pd.DataFrame:
    """Sharpe ratio with MBB percentile intervals, plus paired differences vs ``benchmark``.

    All strategies share the SAME resampled indices, so differences of Sharpe ratios keep their
    cross-sectional dependence. ``p_better`` is the share of resamples where the strategy's
    Sharpe exceeds the benchmark's.
    """
    x = returns.to_numpy(dtype=float)
    n, _ = x.shape
    b = block or default_block_length(n)
    rng = np.random.default_rng(seed)
    boots = []
    done = 0
    while done < n_boot:
        m = min(batch, n_boot - done)
        idx = moving_block_indices(n, b, m, rng)
        boots.append(sharpe(x[idx], axis=1))
        done += m
    sb = np.vstack(boots)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = pd.DataFrame(
        {
            "sharpe": sharpe(x),
            "ci_low": np.quantile(sb, lo, axis=0),
            "ci_high": np.quantile(sb, hi, axis=0),
        },
        index=returns.columns,
    )
    if benchmark is not None:
        j = list(returns.columns).index(benchmark)
        diff = sb - sb[:, [j]]
        out["diff_vs_bench"] = out["sharpe"].to_numpy() - out["sharpe"].to_numpy()[j]
        out["diff_ci_low"] = np.quantile(diff, lo, axis=0)
        out["diff_ci_high"] = np.quantile(diff, hi, axis=0)
        out["p_better"] = (diff > 0).mean(axis=0)
    out.attrs.update(block=b, n_boot=n_boot, level=level)
    return out


def summarize(
    res: WalkForwardResult,
    benchmark: str = "EqualWeight",
    rf: pd.Series | None = None,
    **mbb,
) -> pd.DataFrame:
    """One row per strategy: return, vol, Sharpe (net, gross, MBB CI), drawdown, turnover.

    If ``rf`` (daily simple risk-free return, indexed by date) is given, every Sharpe ratio and
    its interval are computed on EXCESS returns (strategy minus rf); returns, volatility and
    drawdown still refer to total net returns.
    """
    net, gross = res.returns_net, res.returns_gross
    shift = 0.0
    if rf is not None:
        r = rf.reindex(net.index)
        if r.isna().any():
            raise ValueError("risk-free series does not cover the evaluation period")
        shift = r.to_numpy()[:, None]
    net_x = net - shift
    gross_x = gross - shift
    ci = mbb_sharpe_ci(net_x, benchmark=benchmark, **mbb)
    rows = []
    per_year = ANN / res.rebalance_every
    for k in net.columns:
        w = res.weights[k]
        rows.append(
            {
                "strategy": k,
                "ann_return": net[k].mean() * ANN,
                "ann_excess_return": net_x[k].mean() * ANN,
                "ann_vol": net[k].std(ddof=1) * np.sqrt(ANN),
                "sharpe_net": ci.loc[k, "sharpe"],
                "sharpe_ci_low": ci.loc[k, "ci_low"],
                "sharpe_ci_high": ci.loc[k, "ci_high"],
                "sharpe_gross": float(sharpe(gross_x[k].to_numpy())),
                "diff_vs_ew": ci.loc[k, "diff_vs_bench"],
                "diff_ci_low": ci.loc[k, "diff_ci_low"],
                "diff_ci_high": ci.loc[k, "diff_ci_high"],
                "oos_cvar_975": empirical_cvar(net_x[k].to_numpy()),
                "max_drawdown": max_drawdown(net[k].to_numpy()),
                "ann_turnover": res.turnover[k].iloc[1:].mean() * per_year,
                "weight_std": float(w.std(axis=1).mean()),
                "effective_n": float((1.0 / (w**2).sum(axis=1)).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(ci.attrs, excess=rf is not None)
    return out
