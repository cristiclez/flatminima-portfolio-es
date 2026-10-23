import numpy as np
import pandas as pd
import pytest

from flatminima.data import db, riskfree

CSV = """observation_date,DTB3
2020-01-02,1.54
2020-01-03,1.52
2020-01-06,.
2020-01-07,1.55
"""


def test_parse_drops_missing_and_sorts():
    s = riskfree.parse_fred_csv(CSV)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03", "2020-01-07"]
    assert s.iloc[0] == pytest.approx(1.54)


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError):
        riskfree.parse_fred_csv("a\n1\n")
    with pytest.raises(ValueError):
        riskfree.parse_fred_csv("d,v\n2020-01-01,.\n")


def test_daily_conversion_and_forward_fill():
    s = riskfree.parse_fred_csv(CSV)
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])
    rf = riskfree.daily_rf(s, idx)
    np.testing.assert_allclose(rf.to_numpy(), np.array([1.52, 1.52, 1.55]) / 100 / 252)


def test_daily_rf_requires_coverage():
    s = riskfree.parse_fred_csv(CSV)
    with pytest.raises(ValueError):
        riskfree.daily_rf(s, pd.DatetimeIndex(["2019-12-30"]))


def test_database_roundtrip():
    con = db.connect()
    s = riskfree.parse_fred_csv(CSV)
    assert riskfree.store(con, s) == 3
    back = riskfree.read(con)
    pd.testing.assert_series_equal(back, s, check_freq=False)
