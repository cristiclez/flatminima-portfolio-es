import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatminima.data import db, ingest, quality, returns


def _prices(tickers, n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for t in tickers:
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"ticker": t, "date": dates, "adj_close": p, "volume": 1e6}))
    return pd.concat(frames, ignore_index=True)


def test_load_rejects_missing_columns():
    con = db.connect()
    with pytest.raises(ValueError):
        ingest.load_prices(con, pd.DataFrame({"ticker": ["A"]}), "test")


def test_load_is_idempotent():
    con = db.connect()
    df = _prices(["A", "B"])
    ingest.load_prices(con, df, "test")
    ingest.load_prices(con, df, "test")
    assert con.execute("SELECT count(*) FROM raw_prices").fetchone()[0] == len(df)


def test_log_returns_match_numpy():
    con = db.connect()
    df = _prices(["A"])
    ingest.load_prices(con, df, "test")
    returns.build_returns(con)
    got = con.execute("SELECT log_ret FROM returns ORDER BY date").df()["log_ret"].to_numpy()
    np.testing.assert_allclose(got, np.diff(np.log(df["adj_close"].to_numpy())), atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(5, 80))
def test_log_returns_telescope(seed, n):
    """Property: sum of log returns equals log(P_T / P_0)."""
    con = db.connect()
    df = _prices(["A"], n=n, seed=seed)
    ingest.load_prices(con, df, "test")
    returns.build_returns(con)
    total = con.execute("SELECT sum(log_ret) FROM returns").fetchone()[0]
    p = df["adj_close"].to_numpy()
    assert total == pytest.approx(np.log(p[-1] / p[0]), abs=1e-9)


def test_wide_returns_is_rectangular_and_has_no_nans():
    con = db.connect()
    df = _prices(["A", "B", "C"])
    df = df.drop(df[(df.ticker == "B") & (df.index % 30 < 3)].index)  # B starts late
    ingest.load_prices(con, df, "test")
    returns.build_returns(con)
    wide = returns.wide_returns(con)
    assert not wide.isna().any().any()
    assert list(wide.columns) == ["A", "B", "C"]


def test_quality_checks_detect_injected_anomalies():
    con = db.connect()
    df = _prices(["A", "B"], n=60)
    # B: price jump of +100% (log ret ~0.69), a stale run, and a missing date.
    idx = df[df.ticker == "B"].index
    df.loc[idx[10:], "adj_close"] *= 2.0
    df.loc[idx[30:36], "adj_close"] = df.loc[idx[30], "adj_close"]
    df = df.drop(idx[45])
    ingest.load_prices(con, df, "test")
    rep = quality.run_checks(con).set_index(["ticker", "check_name"])["n_flagged"]
    assert rep[("B", "abs_logret_gt_0.25")] == 1
    assert rep[("B", "stale_run_ge_5")] == 1
    assert rep[("B", "missing_dates_in_span")] == 1
    assert rep[("A", "abs_logret_gt_0.25")] == 0
    assert rep[("A", "missing_dates_in_span")] == 0
