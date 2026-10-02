"""Price ingestion. Long format: ticker, date, adj_close, volume."""

from __future__ import annotations

import duckdb
import pandas as pd

REQUIRED = ["ticker", "date", "adj_close", "volume"]


def load_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, source: str) -> int:
    """Insert (or replace) rows into ``raw_prices``. Returns the number of rows written."""
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[REQUIRED].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["source"] = source
    con.register("_incoming", out)
    con.execute(
        "INSERT OR REPLACE INTO raw_prices "
        "SELECT ticker, date, adj_close, volume, source FROM _incoming"
    )
    con.unregister("_incoming")
    return len(out)


def download_yahoo(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download split/dividend-adjusted daily prices from Yahoo Finance.

    Terms of use: Yahoo data is for personal/research use. Do not redistribute it; the
    repository therefore never commits data and rebuilds it with ``make data``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the data extra: pip install -e '.[data]'") from exc

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    close = (
        raw["Close"].reset_index().melt(id_vars="Date", var_name="ticker", value_name="adj_close")
    )
    vol = raw["Volume"].reset_index().melt(id_vars="Date", var_name="ticker", value_name="volume")
    out = close.merge(vol, on=["Date", "ticker"]).rename(columns={"Date": "date"})
    return out.dropna(subset=["adj_close"])[REQUIRED]
