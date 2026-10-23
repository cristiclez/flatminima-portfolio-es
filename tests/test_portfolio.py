import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatminima.covariance.estimators import sample_cov
from flatminima.data.synthetic import simulate_panel
from flatminima.portfolio.metrics import (
    default_block_length,
    max_drawdown,
    mbb_sharpe_ci,
    moving_block_indices,
    sharpe,
    summarize,
)
from flatminima.portfolio.optimizers import equal_weight, min_cvar, min_variance
from flatminima.portfolio.walkforward import walk_forward


def _rets(n_assets=8, n=700, seed=3):
    df = simulate_panel([f"A{i}" for i in range(n_assets)], n=n, start="2015-01-05", seed=seed)
    p = df.pivot(index="date", columns="ticker", values="adj_close")
    return np.log(p).diff().dropna()


# ----------------------------------------------------------------- optimisers
@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 1000), n=st.integers(3, 12))
def test_min_variance_weights_are_a_long_only_budget(seed, n):
    x = np.random.default_rng(seed).standard_normal((200, n)) * 0.01
    w = min_variance(sample_cov(x))
    assert w.sum() == pytest.approx(1.0, abs=1e-9) and (w >= 0).all()


def test_min_variance_known_solution_two_uncorrelated_assets():
    w = min_variance(np.diag([0.04, 0.01]))
    np.testing.assert_allclose(w, [0.2, 0.8], atol=1e-5)  # inverse-variance weights


def test_min_variance_beats_equal_weight_in_sample():
    x = np.random.default_rng(1).standard_normal((300, 6)) * np.linspace(0.005, 0.03, 6)
    c = sample_cov(x)
    w, ew = min_variance(c), equal_weight(6)
    assert w @ c @ w <= ew @ c @ ew + 1e-12


def test_min_cvar_budget_and_optimality_in_sample():
    x = np.random.default_rng(2).standard_normal((400, 5)) * np.linspace(0.005, 0.02, 5)

    def cvar(w, level=0.975):
        loss = -x @ w
        q = np.quantile(loss, level)
        return loss[loss >= q].mean()

    w = min_cvar(x, 0.975)
    assert w.sum() == pytest.approx(1.0, abs=1e-8) and (w >= 0).all()
    assert cvar(w) <= cvar(equal_weight(5)) + 1e-6  # in-sample, up to quantile discretisation


# ----------------------------------------------------------------- walk-forward
def test_walk_forward_shapes_weights_and_turnover():
    r = _rets()
    res = walk_forward(r, lookback=250, rebalance_every=21, cost_bps=5)
    n_oos = len(r) - 250
    assert res.returns_net.shape == (n_oos, 5)
    for w in res.weights.values():
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)
        assert (w >= -1e-12).all()
    assert (res.returns_net <= res.returns_gross + 1e-15).all().all()
    assert res.turnover["EqualWeight"].iloc[1:].max() < 0.2  # only drift is traded


def test_walk_forward_costs_reduce_returns_exactly_by_traded_notional():
    r = _rets()
    free = walk_forward(r, lookback=250, cost_bps=0)
    paid = walk_forward(r, lookback=250, cost_bps=10)
    gap = (free.returns_net - paid.returns_net).sum()
    expected = 10 / 1e4 * free.turnover.sum()
    pd.testing.assert_series_equal(gap, expected, check_names=False, atol=1e-12)


def test_walk_forward_has_no_look_ahead():
    r = _rets()
    k = 500
    r2 = r.copy()
    r2.iloc[k:] = np.random.default_rng(0).standard_normal((len(r) - k, r.shape[1])) * 0.05
    a = walk_forward(r, lookback=250, rebalance_every=21)
    b = walk_forward(r2, lookback=250, rebalance_every=21)
    before = a.returns_net.index < r.index[k]
    pd.testing.assert_frame_equal(a.returns_net[before], b.returns_net[before])


def test_walk_forward_too_short_rejected():
    with pytest.raises(ValueError):
        walk_forward(_rets(n=200), lookback=250)


# ----------------------------------------------------------------- metrics + MBB
def test_sharpe_and_drawdown_known_values():
    r = np.array([0.01, -0.01, 0.01, -0.01])
    assert sharpe(r) == pytest.approx(0.0, abs=1e-12)
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_block_indices_are_contiguous_blocks_within_range():
    idx = moving_block_indices(100, 10, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() <= 99
    first = idx[:, :10]
    assert (np.diff(first, axis=1) == 1).all()
    with pytest.raises(ValueError):
        moving_block_indices(10, 20, 5, np.random.default_rng(0))


def test_default_block_length_order_n_one_third():
    assert default_block_length(4000) == 16 and default_block_length(1) == 1


def test_mbb_coverage_on_iid_data():
    rng = np.random.default_rng(0)
    mu, sd, t = 0.0004, 0.01, 500
    truth = mu / sd * np.sqrt(252)
    hits = 0
    for s in range(150):
        x = pd.DataFrame({"a": rng.normal(mu, sd, t)})
        ci = mbb_sharpe_ci(x, n_boot=300, seed=s)
        hits += ci.loc["a", "ci_low"] <= truth <= ci.loc["a", "ci_high"]
    assert 0.85 < hits / 150 < 0.99  # nominal 95%; calibration run gave 92.5%


def test_mbb_interval_wider_under_autocorrelation_than_iid_bootstrap():
    rng = np.random.default_rng(2)
    e, x = rng.standard_normal(3000), np.zeros(3000)
    for t in range(1, 3000):
        x[t] = 0.3 * x[t - 1] + e[t]
    df = pd.DataFrame({"a": x * 0.01 + 0.0003})
    iid = mbb_sharpe_ci(df, block=1, n_boot=1000, seed=1).iloc[0]
    blk = mbb_sharpe_ci(df, n_boot=1000, seed=1).iloc[0]
    assert blk["ci_high"] - blk["ci_low"] > 1.1 * (iid["ci_high"] - iid["ci_low"])


def test_paired_difference_vs_itself_is_zero_and_summary_columns():
    r = _rets()
    res = walk_forward(r, lookback=250)
    s = summarize(res, n_boot=200).set_index("strategy")
    assert s.loc["EqualWeight", "diff_vs_ew"] == 0.0
    assert s.loc["EqualWeight", "effective_n"] == pytest.approx(8.0)
    assert (s["sharpe_ci_low"] <= s["sharpe_net"]).all()
    assert (s["sharpe_net"] <= s["sharpe_ci_high"]).all()


def test_runner_end_to_end(tmp_path):
    from flatminima.data.pipeline import run as build_data
    from flatminima.portfolio.run import run

    path = tmp_path / "t.duckdb"
    build_data("synthetic", path, n_synth=600)
    out = run(path, lookback=252, n_boot=100, report=tmp_path / "r.md")
    assert len(out) == 5
    assert "Portfolio baselines" in (tmp_path / "r.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------- cap and risk-free
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 1000), n=st.integers(5, 14))
def test_capped_optimisers_respect_cap_and_budget(seed, n):
    x = np.random.default_rng(seed).standard_normal((250, n)) * np.linspace(0.004, 0.03, n)
    cap = 2.0 / n
    for w in (min_variance(sample_cov(x), cap=cap), min_cvar(x, 0.975, cap=cap)):
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= 0).all() and (w <= cap + 1e-7).all()


def test_cap_binds_and_infeasible_cap_rejected():
    c = np.diag([0.0001, 0.04, 0.04, 0.04])
    assert min_variance(c)[0] > 0.9
    assert min_variance(c, cap=0.4)[0] == pytest.approx(0.4, abs=1e-6)
    with pytest.raises(ValueError):
        min_variance(c, cap=0.2)
    with pytest.raises(ValueError):
        min_cvar(np.random.default_rng(0).standard_normal((100, 4)), cap=0.2)


def test_summarize_with_risk_free_uses_excess_returns():
    r = _rets()
    res = walk_forward(r, lookback=250)
    rf = pd.Series(0.03 / 252, index=res.returns_net.index)
    plain = summarize(res, n_boot=200).set_index("strategy")
    excess = summarize(res, n_boot=200, rf=rf).set_index("strategy")
    assert excess.attrs["excess"] and not plain.attrs["excess"]
    expected = float(sharpe((res.returns_net["EqualWeight"] - rf).to_numpy()))
    assert excess.loc["EqualWeight", "sharpe_net"] == pytest.approx(expected)
    assert (excess["sharpe_net"] < plain["sharpe_net"]).all()
    assert excess.loc["EqualWeight", "ann_return"] == pytest.approx(
        plain.loc["EqualWeight", "ann_return"]
    )
    with pytest.raises(ValueError):
        summarize(res, n_boot=50, rf=rf.iloc[:10])
