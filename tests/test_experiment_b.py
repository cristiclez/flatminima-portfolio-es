"""Specification B runner: end-to-end smoke on a small synthetic panel, plus helpers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from flatminima.data import db, pipeline, riskfree  # noqa: E402
from flatminima.portfolio import experiment_b as eb  # noqa: E402
from flatminima.portfolio.metrics import empirical_cvar  # noqa: E402


def test_empirical_cvar_matches_hand_computation():
    r = np.array([-0.10, -0.08, 0.01, 0.02, 0.03, 0.01, 0.0, 0.02, 0.01, 0.02] * 10)
    # worst 2.5% of 100 days = the three most negative (the 97.5% quantile cuts at -0.08/-0.10)
    losses = np.sort(-r)[::-1]
    q = np.quantile(-r, 0.975)
    assert empirical_cvar(r, 0.975) == pytest.approx(losses[losses >= q].mean())
    assert empirical_cvar(r, 0.975) > 0.07


def test_excess_log_returns_subtract_log_of_daily_rate():
    idx = pd.bdate_range("2020-01-01", periods=5)
    wide = pd.DataFrame(0.001, index=idx, columns=["A", "B"])
    rate = pd.Series(2.52, index=idx)  # 2.52% annual -> 0.0001 per day
    x = eb.excess_log_returns(wide, rate)
    np.testing.assert_allclose(x.to_numpy(), 0.001 - np.log1p(0.0001), atol=1e-12)


def test_runner_end_to_end_on_synthetic_data(tmp_path):
    path = tmp_path / "s.duckdb"
    pipeline.run("synthetic", path, n_synth=960)
    out = eb.run(
        path,
        seeds=(0, 1),
        grad_budget=20,
        analyse_every=2,
        n_boot=50,
        rf_constant_pct=2.0,
        report=tmp_path / "specB.md",
    )
    main = out["main"]
    assert {"EqualWeight", "Direct-Adam", "Direct-SAM-0.05", "Policy-SAM-0.2"} <= set(main.strategy)
    assert set(out["robustness"].seed) == {0, 1}
    assert len(out["hessian"]) > 0 and (out["hessian"].lambda_max.notna()).all()
    text = (tmp_path / "specB.md").read_text(encoding="utf-8")
    assert "post hoc" in text and "SMOKE TEST ONLY" in text
    con = db.connect(path)
    assert con.execute("SELECT COUNT(*) FROM specb_returns").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM specb_hessian").fetchone()[0] > 0
    con.close()


def test_runner_requires_riskfree_table_when_no_constant(tmp_path):
    path = tmp_path / "s.duckdb"
    pipeline.run("synthetic", path, n_synth=960)
    with pytest.raises(SystemExit, match="riskfree"):
        eb.run(path, seeds=(0,), grad_budget=10, report=None)
    assert riskfree.FRED_URL.endswith("DTB3")
