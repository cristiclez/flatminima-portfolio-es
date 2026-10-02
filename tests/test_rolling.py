import numpy as np
import pandas as pd
import pytest

from flatminima.data import db, ingest, returns
from flatminima.data.synthetic import simulate_garch_t, simulate_panel
from flatminima.risk.rolling import rolling_risk
from flatminima.risk.run import run

TRUE = dict(omega=2e-6, alpha=0.03, gamma=0.08, beta=0.90, nu=6.0)


def _series(n, seed):
    r, _ = simulate_garch_t(n, rng=np.random.default_rng(seed), **TRUE)
    return pd.Series(r, index=pd.bdate_range("2010-01-04", periods=n))


@pytest.fixture(scope="module")
def forecasts():
    return rolling_risk(_series(2200, 21), window=600, refit_every=50)


def test_output_structure_and_ordering(forecasts):
    assert len(forecasts) == 1600
    assert (forecasts["es_975"] >= forecasts["var_975"]).all()
    assert (forecasts["var_99"] >= forecasts["var_975"]).all()
    assert (forecasts["sigma"] > 0).all()
    assert forecasts.attrs["n_refits"] == 32 and forecasts.attrs["n_failed_refits"] == 0


def test_exception_rates_are_plausible_on_correctly_specified_data(forecasts):
    """Smoke check only; formal Kupiec/Christoffersen tests are a separate module."""
    loss = -forecasts["ret"]
    assert 0.004 < (loss > forecasts["var_99"]).mean() < 0.022
    assert 0.012 < (loss > forecasts["var_975"]).mean() < 0.045


def test_no_look_ahead():
    """Altering returns from day k onward must not change any forecast made before day k."""
    s = _series(1500, 9)
    k = 1100
    s2 = s.copy()
    s2.iloc[k:] = np.random.default_rng(0).standard_normal(len(s) - k) * 0.05
    a = rolling_risk(s, window=600, refit_every=50)
    b = rolling_risk(s2, window=600, refit_every=50)
    cols = ["sigma", "var_99", "var_975", "es_975"]
    before = a.index < s.index[k]
    pd.testing.assert_frame_equal(a.loc[before, cols], b.loc[before, cols])
    assert not np.allclose(a.loc[~before, "sigma"], b.loc[~before, "sigma"])


def test_series_shorter_than_window_rejected():
    with pytest.raises(ValueError):
        rolling_risk(_series(300, 1), window=500)


def test_equal_weight_return():
    con = db.connect()
    df = simulate_panel(["A", "B"], n=50, start="2020-01-01", seed=1)
    df.loc[df.ticker == "B", "adj_close"] = df.loc[df.ticker == "A", "adj_close"].to_numpy()
    ingest.load_prices(con, df, "test")
    returns.build_returns(con)
    wide = returns.wide_returns(con)
    ew = returns.equal_weight_return(wide)
    np.testing.assert_allclose(ew.to_numpy(), wide["A"].to_numpy(), atol=1e-12)


def test_run_end_to_end_on_synthetic_database(tmp_path):
    from flatminima.config import UNIVERSE
    from flatminima.data.pipeline import run as build_data

    path = tmp_path / "t.duckdb"
    build_data("synthetic", path, n_synth=900)
    out = run(path, series=("SPY", "EW"), window=500, refit_every=100)
    con = db.connect(path)
    n = con.execute("SELECT count(*) FROM risk_forecasts").fetchone()[0]
    assert n == len(out) == 2 * (900 - 500)
    assert "SPY" in UNIVERSE
    assert set(out["series"]) == {"SPY", "EW"}
