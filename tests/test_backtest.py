import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flatminima.backtest.run import backtest_series, run
from flatminima.backtest.tests import (
    acerbi_szekely,
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
    traffic_light_windows,
)
from flatminima.risk.fhs import fhs_var_es


# ---------------------------------------------------------------- Kupiec
@pytest.mark.parametrize("n,x", [(250, 7), (1000, 5), (500, 12), (4470, 45)])
def test_kupiec_matches_binomial_likelihood_ratio(n, x):
    """Independent implementation through scipy's binomial log-pmf."""
    hits = np.zeros(n, dtype=bool)
    hits[:x] = True
    p = 0.01
    lr = -2.0 * (stats.binom.logpmf(x, n, p) - stats.binom.logpmf(x, n, x / n))
    res = kupiec_pof(hits, 0.99)
    assert res.statistic == pytest.approx(lr, rel=1e-9)
    assert res.p_value == pytest.approx(stats.chi2.sf(lr, 1), rel=1e-9)


def test_kupiec_zero_exceptions_and_exact_coverage():
    assert kupiec_pof(np.zeros(250, dtype=bool), 0.99).statistic > 0  # 0*log(0) handled
    h = np.zeros(1000, dtype=bool)
    h[:10] = True  # exactly 1%
    assert kupiec_pof(h, 0.99).statistic == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- Christoffersen
def _markov_ll(h, p01, p11, p_init=None):
    ll = 0.0
    for a, b in zip(h[:-1], h[1:], strict=True):
        pr = (p11 if a else p01) if b else 1 - (p11 if a else p01)
        ll += np.log(pr)
    return ll


def test_christoffersen_independence_matches_explicit_markov_loglik():
    rng = np.random.default_rng(3)
    h = (rng.random(800) < 0.04).astype(int)
    h[100:104] = 1  # inject clustering
    a, b = h[:-1], h[1:]
    n01 = np.sum((a == 0) & (b == 1))
    n00 = np.sum((a == 0) & (b == 0))
    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    ll_alt = _markov_ll(h, n01 / (n00 + n01), n11 / (n10 + n11))
    pi = (n01 + n11) / (len(h) - 1)
    ll_null = _markov_ll(h, pi, pi)
    res = christoffersen_independence(h.astype(bool))
    assert res.statistic == pytest.approx(-2 * (ll_null - ll_alt), rel=1e-9)


def test_christoffersen_cc_is_sum_of_components():
    h = np.random.default_rng(1).random(600) < 0.02
    cc = christoffersen_cc(h, 0.99)
    assert cc.statistic == pytest.approx(
        kupiec_pof(h, 0.99).statistic + christoffersen_independence(h).statistic
    )
    assert cc.p_value == pytest.approx(stats.chi2.sf(cc.statistic, 2))


def _rejection_rate(test, sampler, reps, seed):
    rng = np.random.default_rng(seed)
    return np.mean([test(sampler(rng)).p_value < 0.05 for _ in range(reps)])


def test_size_and_power_of_kupiec():
    size = _rejection_rate(lambda h: kupiec_pof(h, 0.99), lambda r: r.random(1500) < 0.01, 3000, 0)
    power = _rejection_rate(lambda h: kupiec_pof(h, 0.99), lambda r: r.random(1500) < 0.02, 500, 1)
    assert 0.02 < size < 0.09  # nominal 5%; the chi2 approximation is discrete for rare events
    assert power > 0.8  # true rate 2% vs nominal 1%, N=1500


def _markov_hits(rng, n, p01=0.01, p11=0.3):
    h = np.zeros(n, dtype=bool)
    for t in range(1, n):
        h[t] = rng.random() < (p11 if h[t - 1] else p01)
    return h


def test_size_and_power_of_christoffersen_independence():
    size = _rejection_rate(christoffersen_independence, lambda r: r.random(1500) < 0.02, 2000, 2)
    power = _rejection_rate(christoffersen_independence, lambda r: _markov_hits(r, 3000), 150, 3)
    assert size < 0.09
    assert power > 0.5  # clustered exceptions (P(hit|hit)=0.3 vs 0.01) are detected


# ------------------------------------------------------- Basel traffic light
@pytest.mark.parametrize(
    "x,zone",
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (20, "red")],
)
def test_basel_zones_match_published_table(x, zone):
    assert basel_zone(x, 250, 0.99) == zone


def test_basel_cumulative_probabilities_match_published_values():
    cdf = stats.binom.cdf
    assert cdf(4, 250, 0.01) == pytest.approx(0.8922, abs=1e-4)
    assert cdf(5, 250, 0.01) == pytest.approx(0.9588, abs=1e-4)
    assert cdf(9, 250, 0.01) == pytest.approx(0.9997, abs=1e-4)


def test_traffic_light_windows_known_pattern():
    h = np.zeros(500, dtype=bool)
    h[10:16] = True  # 6 exceptions: yellow in every window containing them
    tl = traffic_light_windows(h, 250)
    assert tl["n_windows"] == 251 and tl["max_exceptions"] == 6
    assert tl["share_green"] + tl["share_yellow"] + tl["share_red"] == pytest.approx(1.0)
    # windows starting at 0..11 contain at least 5 of the 6 exceptions -> yellow
    assert tl["share_yellow"] == pytest.approx(12 / 251)
    with pytest.raises(ValueError):
        traffic_light_windows(h[:100], 250)


# ------------------------------------------------------- Acerbi-Szekely
def _h0_forecasts(rng, T=300, W=250, p=0.025, scale=1.0):
    """Forecasts whose realised P&L is drawn from the model's own predictive law (H0)."""
    z = np.sort(rng.standard_t(6, size=(T, W)) * np.sqrt(4 / 6), axis=1)
    sigma = rng.uniform(0.005, 0.02, T)
    mu = np.zeros(T)
    var, es = np.empty(T), np.empty(T)
    for t in range(T):
        var[t], es[t] = fhs_var_es(z[t], 0.0, sigma[t], 1 - p)
    u = rng.random(T)
    ret = np.array([np.interp(u[t] * (W - 1), np.arange(W), z[t]) for t in range(T)])
    ret = scale * (mu + sigma * ret)
    m = int(np.ceil(p * (W - 1))) + 1
    return dict(ret=ret, mu=mu, sigma=sigma, var=var, es=es, z_tail=z[:, :m], window=W, tail_prob=p)


def test_acerbi_szekely_size_under_null():
    """Monte Carlo p-values must be (approximately) uniform when H0 holds.

    A 1,500-replication study (run offline) gave rejection rates of 4.4% (Z1) and 5.7% (Z2)
    at the 5% level; this test uses fewer replications and correspondingly wider bands.
    """
    rng = np.random.default_rng(10)
    p1, p2 = [], []
    for k in range(500):
        f = _h0_forecasts(rng)
        if np.sum(f["ret"] < -f["var"]) == 0:
            continue
        r = acerbi_szekely(**f, n_sims=500, seed=k)
        p1.append(r.p_z1)
        p2.append(r.p_z2)
    for p in (np.array(p1), np.array(p2)):
        assert 0.02 < np.mean(p < 0.05) < 0.09
        assert 0.44 < p.mean() < 0.56


def test_acerbi_szekely_power_when_es_is_underestimated():
    rng = np.random.default_rng(11)
    f = _h0_forecasts(rng, T=800, scale=1.0)
    ok = acerbi_szekely(**f, n_sims=2000, seed=1)
    bad_f = dict(f, ret=f["ret"] * 1.5)  # realised losses 50% larger than forecast
    bad = acerbi_szekely(**bad_f, n_sims=2000, seed=1)
    assert ok.p_z2 > 0.01
    assert bad.p_z2 < 0.01 and bad.z2 < ok.z2


def test_acerbi_szekely_requires_exceptions():
    f = _h0_forecasts(np.random.default_rng(0))
    f["ret"] = np.abs(f["ret"])  # no losses at all
    with pytest.raises(ValueError):
        acerbi_szekely(**f, n_sims=10)


# ------------------------------------------------------- Runner
def test_backtest_runner_end_to_end(tmp_path):
    from flatminima.data import db
    from flatminima.data.pipeline import run as build_data
    from flatminima.risk.run import run as run_risk

    path = tmp_path / "t.duckdb"
    build_data("synthetic", path, n_synth=1100)
    run_risk(path, series=("SPY",), window=500, refit_every=100)
    report = tmp_path / "report.md"
    res, tl = run(path, report=report, n_sims=500)
    assert set(res["test"]) >= {"Kupiec POF", "Christoffersen CC", "Acerbi-Szekely Z2"}
    assert res["p_value"].between(0, 1).all()
    assert "Backtest results" in report.read_text(encoding="utf-8")
    con = db.connect(path)
    assert con.execute("SELECT count(*) FROM backtest_results_risk_forecasts").fetchone()[0] == len(
        res
    )
    assert len(tl) == 2


def test_backtest_series_columns():
    rng = np.random.default_rng(0)
    f = _h0_forecasts(rng, T=300, W=250)
    m = f["z_tail"].shape[1]
    df = pd.DataFrame(
        {
            "series": "X",
            "window": 250,
            "ret": f["ret"],
            "mu": f["mu"],
            "sigma": f["sigma"],
            "var_99": f["var"] * 1.2,
            "var_975": f["var"],
            "es_975": f["es"],
            "hs_var_99": f["var"] * 1.2,
        }
    )
    for k in range(m):
        df[f"zq_{k:02d}"] = f["z_tail"][:, k]
    res = backtest_series(df, n_sims=200)
    assert len(res) == 10 and res["reject_5pct"].dtype == bool


def test_pipeline_accepts_correct_model_and_rejects_underestimated_risk():
    """End-to-end validation of the tests on simulated data with known truth."""
    from flatminima.data.synthetic import simulate_garch_t
    from flatminima.risk.rolling import rolling_risk

    r, _ = simulate_garch_t(2200, 2e-6, 0.03, 0.90, 6.0, np.random.default_rng(21), gamma=0.08)
    s = pd.Series(r, index=pd.bdate_range("2010-01-04", periods=len(r)))
    fc = rolling_risk(s, window=600, refit_every=50).reset_index(names="date")
    fc.insert(0, "series", "SIM")
    fc["window"] = 600

    good = backtest_series(fc, n_sims=2000)
    fhs = good[good["method"] == "FHS-GJR"]
    assert not fhs["reject_5pct"].any(), fhs[fhs["reject_5pct"]]  # correct model: no rejection
    # Unfiltered historical simulation ignores volatility clustering: right frequency of
    # exceptions (Kupiec passes) but clustered exceptions (Christoffersen independence rejects).
    hist = good[good["method"] == "Historical (unfiltered)"].set_index("test")
    assert not hist.loc["Kupiec POF", "reject_5pct"]
    assert hist.loc["Christoffersen independence", "reject_5pct"]

    bad_fc = fc.copy()
    for col in ("var_99", "var_975", "es_975"):
        bad_fc[col] = bad_fc[col] * 0.65  # model that underestimates risk by 35%
    bad = backtest_series(bad_fc, n_sims=2000).set_index(["method", "target", "test"])
    assert bad.loc[("FHS-GJR", "VaR 99%", "Kupiec POF"), "reject_5pct"]
    assert bad.loc[("FHS-GJR", "VaR 97.5%", "Christoffersen CC"), "reject_5pct"]
    assert bad.loc[("FHS-GJR", "ES 97.5%", "Acerbi-Szekely Z2"), "reject_5pct"]
