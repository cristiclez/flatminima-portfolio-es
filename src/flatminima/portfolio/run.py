"""Walk-forward evaluation of the baselines on the stored returns, with a Markdown report.

    python -m flatminima.portfolio.run --cost-bps 5

Reports (net of the explicit transaction cost) annualised return/vol, Sharpe with Moving
Block Bootstrap 95% intervals, paired Sharpe difference vs equal weight, drawdown, turnover,
weight dispersion and effective number of assets. Out-of-sample means each rebalance uses only
the preceding ``lookback`` days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from flatminima.data import db, returns
from flatminima.portfolio.metrics import summarize
from flatminima.portfolio.walkforward import walk_forward


def run(
    db_path: str | Path,
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 5.0,
    n_boot: int = 5000,
    report: str | Path | None = "docs/portfolio_baselines.md",
) -> pd.DataFrame:
    """Run all baselines and write the summary table (and optionally a report)."""
    con = db.connect(db_path)
    wide = returns.wide_returns(con)
    source = con.execute("SELECT DISTINCT source FROM raw_prices").fetchall()[0][0]
    con.close()
    res = walk_forward(wide, lookback=lookback, rebalance_every=rebalance_every, cost_bps=cost_bps)
    s = summarize(res, n_boot=n_boot)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        d0, d1 = res.returns_net.index.min().date(), res.returns_net.index.max().date()
        lines = [
            "# Portfolio baselines (walk-forward)",
            "",
            f"Data source: `{source}` ({wide.shape[1]} ETFs). Out-of-sample: {d0} to {d1} "
            f"({len(res.returns_net)} days). Lookback {lookback} days, rebalance every "
            f"{rebalance_every} days, long-only, fully invested, cost {cost_bps} bps per unit "
            f"traded notional. Sharpe = annualised mean/std of net returns (no risk-free rate); "
            f"intervals: Moving Block Bootstrap, block {s.attrs['block']}, {s.attrs['n_boot']} "
            f"resamples, 95% percentile. Not multiple-testing corrected.",
            "",
            "| Strategy | Ann. ret | Ann. vol | Sharpe (95% CI) | vs EW (95% CI) | Max DD "
            "| Turnover/yr | Weight std | Eff. N |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in s.itertuples():
            lines.append(
                f"| {r.strategy} | {r.ann_return:.1%} | {r.ann_vol:.1%} | {r.sharpe_net:.2f} "
                f"({r.sharpe_ci_low:.2f}, {r.sharpe_ci_high:.2f}) | {r.diff_vs_ew:+.2f} "
                f"({r.diff_ci_low:+.2f}, {r.diff_ci_high:+.2f}) | {r.max_drawdown:.1%} "
                f"| {r.ann_turnover:.2f} | {r.weight_std:.3f} | {r.effective_n:.1f} |"
            )
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/flatminima.duckdb")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--report", default="docs/portfolio_baselines.md")
    a = p.parse_args()
    run(a.db, a.lookback, a.rebalance_every, a.cost_bps, a.n_boot, a.report)


if __name__ == "__main__":
    main()
