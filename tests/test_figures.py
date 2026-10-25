"""Figure module: builds PNGs from stored tables and skips (never fakes) missing ones."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from flatminima.data import db  # noqa: E402
from flatminima.report import figures  # noqa: E402


def _db(path):
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2015-01-01", periods=400)
    rows = []
    for k in (
        "EqualWeight",
        "MinVar-Sample(cap)",
        "Direct-Adam",
        "Direct-SAM-0.05",
        "Policy-Adam",
        "Policy-SAM-0.05",
    ):
        rows.append(pd.DataFrame({"date": d, "strategy": k, "ret": rng.normal(2e-4, 0.006, 400)}))
    ret = pd.concat(rows)
    h = pd.DataFrame(
        {
            "strategy": ["Direct-Adam", "Direct-SAM-0.05", "Policy-Adam", "Policy-SAM-0.05"] * 3,
            "seed": 0,
            "date": np.repeat(d[[10, 100, 200]], 4),
            "lambda_max": rng.uniform(1e-5, 1e-3, 12),
        }
    )
    f = pd.DataFrame(
        {
            "series": "SPY",
            "date": d,
            "ret": rng.normal(0, 0.01, 400),
            "var_99": 0.025,
            "es_975": 0.03,
            "window": 1000,
            "model": "gjr",
        }
    )
    con = db.connect(path)
    for name, df in (("specb_returns", ret), ("specb_hessian", h), ("risk_forecasts", f)):
        con.register("_t", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    con.close()


def test_all_figures_are_written(tmp_path):
    _db(tmp_path / "t.duckdb")
    made = figures.run(tmp_path / "t.duckdb", tmp_path / "figs", extended=True)
    names = {p.name for p in made}
    assert names == {
        "fig1_var_es.png",
        "fig2_traffic_light.png",
        "fig3_cumulative_specB.png",
        "fig4_curvature_specB.png",
        "specB_drawdown.png",
        "specB_paired_sharpe.png",
    }
    assert all(p.stat().st_size > 5_000 for p in made)


def test_missing_tables_are_skipped_not_faked(tmp_path, capsys):
    con = db.connect(tmp_path / "e.duckdb")
    con.close()
    assert figures.run(tmp_path / "e.duckdb", tmp_path / "figs") == []
    assert "skip" in capsys.readouterr().out
