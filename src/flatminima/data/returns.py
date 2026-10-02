"""Log-return construction in SQL and wide-panel extraction."""

from __future__ import annotations

import duckdb
import pandas as pd


def build_returns(con: duckdb.DuckDBPyConnection) -> int:
    """Create table ``returns`` (ticker, date, log_ret) from adjusted prices."""
    con.execute(
        """
        CREATE OR REPLACE TABLE returns AS
        SELECT ticker, date, log_ret FROM (
            SELECT ticker, date,
                   ln(adj_close / lag(adj_close) OVER w) AS log_ret
            FROM raw_prices
            WHERE adj_close > 0
            WINDOW w AS (PARTITION BY ticker ORDER BY date)
        ) WHERE log_ret IS NOT NULL
        """
    )
    return con.execute("SELECT count(*) FROM returns").fetchone()[0]


def wide_returns(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Dates x tickers matrix of log returns on the common calendar (no forward-filling).

    Only dates where *every* selected ticker has a return are kept, so the panel is
    rectangular. Tickers that start late shorten the sample; choose the universe accordingly.
    """
    df = con.execute("SELECT ticker, date, log_ret FROM returns").df()
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    wide = df.pivot(index="date", columns="ticker", values="log_ret").sort_index()
    wide.index = pd.to_datetime(wide.index)
    if start:
        wide = wide.loc[start:]
    if end:
        wide = wide.loc[:end]
    return wide.dropna(how="any")
