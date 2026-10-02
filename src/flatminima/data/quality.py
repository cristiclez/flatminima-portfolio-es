"""Data-quality checks. Every check returns flagged rows; nothing is silently fixed."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

MAX_ABS_LOGRET = 0.25  # |daily log return| above this is flagged for review
STALE_RUN = 5  # this many consecutive identical prices is flagged


def run_checks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run all checks, store the summary in ``data_quality`` and return it."""
    px = con.execute("SELECT ticker, date, adj_close FROM raw_prices ORDER BY ticker, date").df()
    px["date"] = pd.to_datetime(px["date"])
    rows: list[dict] = []
    calendar = pd.DatetimeIndex(sorted(px["date"].unique()))

    for ticker, g in px.groupby("ticker"):
        p = g.set_index("date")["adj_close"]

        bad = int((p <= 0).sum() + p.isna().sum())
        rows.append(_row(ticker, "non_positive_or_null_price", bad, ""))

        # Gaps: dates present for other tickers inside this ticker's own life span.
        span = calendar[(calendar >= p.index.min()) & (calendar <= p.index.max())]
        missing = span.difference(p.index)
        detail = f"first missing: {missing[0].date()}" if len(missing) else ""
        rows.append(_row(ticker, "missing_dates_in_span", len(missing), detail))

        pos = p[p > 0]
        lr = np.log(pos).diff().dropna()
        big = lr[lr.abs() > MAX_ABS_LOGRET]
        detail = f"largest: {big.abs().idxmax().date()}" if len(big) else ""
        rows.append(_row(ticker, f"abs_logret_gt_{MAX_ABS_LOGRET}", len(big), detail))

        same = (pos.diff() == 0).astype(int)
        run = same.groupby((same == 0).cumsum()).cumsum()
        rows.append(_row(ticker, f"stale_run_ge_{STALE_RUN}", int((run == STALE_RUN).sum()), ""))

    report = pd.DataFrame(rows)
    con.execute("DELETE FROM data_quality")
    con.register("_dq", report)
    con.execute("INSERT INTO data_quality SELECT * FROM _dq")
    con.unregister("_dq")
    return report


def _row(ticker: str, check: str, n: int, detail: str) -> dict:
    return {"ticker": ticker, "check_name": check, "n_flagged": int(n), "detail": detail}
