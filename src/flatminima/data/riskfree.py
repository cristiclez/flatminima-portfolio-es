"""Risk-free rate: US 3-month Treasury bill, FRED series DTB3 (annualised percent, daily).

The daily simple risk-free return is approximated as ``DTB3 / 100 / 252`` so that it is
consistent with the 252-day annualisation used for Sharpe ratios. DTB3 is quoted on a bank
discount basis, so this is an approximation (documented as a limitation).

    python -m flatminima.data.riskfree                 # download from FRED into the database
    python -m flatminima.data.riskfree --csv dtb3.csv  # or load a CSV downloaded by hand
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
TRADING_DAYS = 252


def parse_fred_csv(text: str) -> pd.Series:
    """Parse a FRED CSV (two columns: date, value; '.' marks a missing value)."""
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("expected a CSV with a date column and a value column")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    values.index = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]), name="date").as_unit("ns")
    out = values.dropna().sort_index().rename("rate_pct")
    if out.empty:
        raise ValueError("no numeric observations found")
    return out


def fetch_fred(url: str = FRED_URL, timeout: float = 30.0) -> pd.Series:
    """Download DTB3 from FRED (public CSV endpoint, no API key)."""
    req = urllib.request.Request(url, headers={"User-Agent": "flatminima-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    except OSError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "could not download DTB3 from FRED. Download the CSV manually from "
            f"{url} and run: python -m flatminima.data.riskfree --csv <file>"
        ) from exc
    return parse_fred_csv(text)


def daily_rf(rate_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily simple risk-free return on ``index`` (last known rate carried forward)."""
    union = rate_pct.index.union(index)
    aligned = rate_pct.reindex(union).ffill().reindex(index)
    if aligned.isna().any():
        raise ValueError("risk-free series does not cover the start of the requested index")
    return (aligned / 100.0 / TRADING_DAYS).rename("rf")


def store(con: duckdb.DuckDBPyConnection, rate_pct: pd.Series) -> int:
    """Write the series to table ``riskfree`` (date, rate_pct). Returns the row count."""
    df = pd.DataFrame({"date": rate_pct.index.date, "rate_pct": rate_pct.to_numpy()})
    con.register("_rf", df)
    con.execute("CREATE OR REPLACE TABLE riskfree AS SELECT * FROM _rf")
    con.unregister("_rf")
    return len(df)


def read(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Read table ``riskfree`` back as a Series of annualised percent rates."""
    df = con.execute("SELECT date, rate_pct FROM riskfree ORDER BY date").df()
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]), name="date").as_unit("ns")
    return pd.Series(df["rate_pct"].to_numpy(), index=idx, name="rate_pct")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/flatminima.duckdb")
    p.add_argument("--csv", help="path of a DTB3 CSV downloaded by hand")
    a = p.parse_args()
    from flatminima.data import db

    rate = parse_fred_csv(Path(a.csv).read_text(encoding="utf-8")) if a.csv else fetch_fred()
    con = db.connect(a.db)
    n = store(con, rate)
    con.close()
    print(f"riskfree: {n} rows, {rate.index.min().date()} to {rate.index.max().date()}")


if __name__ == "__main__":
    main()