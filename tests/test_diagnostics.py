import numpy as np
import pytest

from flatminima.data.synthetic import simulate_garch_t
from flatminima.risk.garch import conditional_variance
from flatminima.stats.diagnostics import arch_lm, jarque_bera, ljung_box, residual_diagnostics


def test_ljung_box_matches_statsmodels_reference():
    # Reference values produced once with statsmodels.stats.diagnostic.acorr_ljungbox.
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2000)
    x[1:] += 0.1 * x[:-1]
    q, p = ljung_box(x, 10)
    assert q == pytest.approx(50.85467519414129, rel=1e-9)
    assert p == pytest.approx(1.8576317895843502e-07, rel=1e-6)


def test_arch_lm_matches_statsmodels_reference():
    # Reference from statsmodels.stats.diagnostic.het_arch(z, nlags=10), same random stream.
    rng = np.random.default_rng(5)
    rng.standard_normal(2000)
    z = rng.standard_normal(3000)
    stat, p = arch_lm(z, 10)
    assert stat == pytest.approx(9.069357028818434, rel=1e-9)
    assert p == pytest.approx(0.5255342538952048, rel=1e-9)


def test_ljung_box_rejects_autocorrelation_and_accepts_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.5 * ar[t - 1] + noise[t]
    assert ljung_box(ar, 10)[1] < 1e-6
    assert ljung_box(noise, 10)[1] > 0.01


def test_arch_effects_detected_in_raw_returns_but_not_in_filtered_residuals():
    true = dict(omega=2e-6, alpha=0.08, beta=0.90, nu=8.0)
    r, _ = simulate_garch_t(5000, rng=np.random.default_rng(8), **true)
    assert arch_lm(r, 10)[1] < 1e-6  # raw returns: strong ARCH effects
    from flatminima.risk.garch import GarchParams

    p = GarchParams(0.0, true["omega"], true["alpha"], 0.0, true["beta"], true["nu"])
    s2, _ = conditional_variance(p, r)
    z = r / np.sqrt(s2)
    assert arch_lm(z, 10)[1] > 0.01  # filtered with the TRUE model: effects removed
    assert ljung_box(z**2, 10)[1] > 0.01


def test_jarque_bera_rejects_fat_tails():
    t = np.random.default_rng(2).standard_t(4, 5000)
    assert jarque_bera(t)[1] < 1e-6


def test_residual_diagnostics_table_and_input_checks():
    df = residual_diagnostics(np.random.default_rng(3).standard_normal(1000))
    assert list(df.columns) == ["test", "statistic", "p_value"] and len(df) == 4
    with pytest.raises(ValueError):
        ljung_box(np.ones(5), lags=10)
