"""DuckDB connection and schema."""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_prices (
    ticker    VARCHAR NOT NULL,
    date      DATE    NOT NULL,
    adj_close DOUBLE,
    volume    DOUBLE,
    source    VARCHAR NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS data_quality (
    ticker     VARCHAR,
    check_name VARCHAR,
    n_flagged  INTEGER,
    detail     VARCHAR
);
"""


def connect(path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open (and initialise) a DuckDB database."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con
