from flatminima.config import UNIVERSE
from flatminima.data import db, returns
from flatminima.data.pipeline import run


def test_end_to_end_synthetic(tmp_path):
    path = tmp_path / "t.duckdb"
    run("synthetic", path, n_synth=300)
    con = db.connect(path)
    assert con.execute("SELECT count(DISTINCT ticker) FROM raw_prices").fetchone()[0] == len(
        UNIVERSE
    )
    assert con.execute("SELECT DISTINCT source FROM raw_prices").fetchall() == [("synthetic",)]
    wide = returns.wide_returns(con)
    assert wide.shape == (300, len(UNIVERSE))
    assert not wide.isna().any().any()
