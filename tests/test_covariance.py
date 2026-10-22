import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.covariance import empirical_covariance
from sklearn.covariance import ledoit_wolf as sk_lw

from flatminima.covariance.estimators import (
    ledoit_wolf,
    marchenko_pastur_edges,
    rmt_clean,
    sample_cov,
)


def _x(t=250, n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n)) @ rng.standard_normal((n, n)) * 0.01


def test_sample_cov_matches_sklearn():
    x = _x()
    np.testing.assert_allclose(sample_cov(x), empirical_covariance(x), atol=1e-15)


def test_ledoit_wolf_matches_sklearn_reference():
    x = _x(seed=1)
    cov, delta = ledoit_wolf(x)
    ref_cov, ref_delta = sk_lw(x)
    assert delta == pytest.approx(ref_delta, rel=1e-9)
    np.testing.assert_allclose(cov, ref_cov, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), t=st.integers(40, 300), n=st.integers(3, 25))
def test_estimators_are_symmetric_psd_with_bounded_shrinkage(seed, t, n):
    x = _x(t, n, seed)
    lw, delta = ledoit_wolf(x)
    rm, k = rmt_clean(x)
    assert 0.0 <= delta <= 1.0
    for c in (sample_cov(x), lw, rm):
        np.testing.assert_allclose(c, c.T, atol=1e-14)
        assert np.linalg.eigvalsh(c).min() > -1e-12
    assert 0 <= k <= n


def test_marchenko_pastur_bulk_contains_pure_noise_spectrum():
    n, t = 100, 400
    z = np.random.default_rng(0).standard_normal((t, n))
    ev = np.linalg.eigvalsh(np.corrcoef(z.T))
    lo, hi = marchenko_pastur_edges(n, t)
    assert (lo, hi) == pytest.approx((0.25, 2.25))
    assert ev.min() >= 0.95 * lo and ev.max() <= 1.05 * hi


def test_rmt_keeps_factor_and_preserves_variances_and_trace():
    rng = np.random.default_rng(3)
    t, n = 500, 30
    f = rng.standard_normal((t, 1)) * 0.01
    x = f @ rng.uniform(0.8, 1.2, (1, n)) + rng.standard_normal((t, n)) * 0.01
    cleaned, k = rmt_clean(x)
    assert k >= 1  # the common factor lies above the Marchenko-Pastur bulk
    np.testing.assert_allclose(np.diag(cleaned), np.diag(sample_cov(x)), rtol=1e-10)


def test_rmt_on_pure_noise_collapses_to_identity_correlation():
    z = np.random.default_rng(4).standard_normal((600, 20))
    cleaned, k = rmt_clean(z)
    d = np.sqrt(np.diag(cleaned))
    corr = cleaned / np.outer(d, d)
    assert k == 0
    np.testing.assert_allclose(corr, np.eye(20), atol=1e-10)


def test_input_validation():
    with pytest.raises(ValueError):
        sample_cov(np.ones(10))
