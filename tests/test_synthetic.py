import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatminima.data.synthetic import simulate_garch_t


def test_reproducible_with_seed():
    a, _ = simulate_garch_t(500, 1e-6, 0.08, 0.9, 6.0, np.random.default_rng(1))
    b, _ = simulate_garch_t(500, 1e-6, 0.08, 0.9, 6.0, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_unconditional_variance_matches_theory():
    alpha, beta, vol = 0.08, 0.90, 0.01
    omega = vol**2 * (1 - alpha - beta)
    r, _ = simulate_garch_t(400_000, omega, alpha, beta, 8.0, np.random.default_rng(7))
    assert r.std() == pytest.approx(vol, rel=0.06)


def test_standardised_residuals_have_unit_variance():
    r, sig = simulate_garch_t(200_000, 2e-6, 0.07, 0.9, 6.0, np.random.default_rng(3))
    assert (r / sig).std() == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(omega=-1.0, alpha=0.1, beta=0.8, nu=6.0),
        dict(omega=1e-6, alpha=0.2, beta=0.8, nu=6.0),
        dict(omega=1e-6, alpha=0.1, beta=0.8, nu=2.0),
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate_garch_t(10, rng=np.random.default_rng(0), **kwargs)


@settings(max_examples=25, deadline=None)
@given(
    alpha=st.floats(0.01, 0.3),
    beta=st.floats(0.3, 0.65),
    nu=st.floats(4.5, 30.0),
    seed=st.integers(0, 10_000),
)
def test_volatility_always_positive_and_finite(alpha, beta, nu, seed):
    r, sig = simulate_garch_t(300, 1e-6, alpha, beta, nu, np.random.default_rng(seed))
    assert np.all(sig > 0) and np.all(np.isfinite(r))
