import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatminima.risk.fhs import fhs_var_es, historical_var_es

_z = st.lists(st.floats(-6, 6, allow_nan=False), min_size=30, max_size=300)


@settings(max_examples=60, deadline=None)
@given(z=_z, level=st.floats(0.9, 0.995), mu=st.floats(-0.01, 0.01), sig=st.floats(0.001, 0.05))
def test_es_at_least_var(z, level, mu, sig):
    var, es = fhs_var_es(np.array(z), mu, sig, level)
    assert es >= var - 1e-12


@settings(max_examples=60, deadline=None)
@given(z=_z, sig=st.floats(0.001, 0.05))
def test_var_increases_with_confidence(z, sig):
    z = np.array(z)
    v975, _ = fhs_var_es(z, 0.0, sig, 0.975)
    v99, _ = fhs_var_es(z, 0.0, sig, 0.99)
    assert v99 >= v975 - 1e-12


@settings(max_examples=40, deadline=None)
@given(z=_z, c=st.floats(0.1, 10.0))
def test_scale_equivariance(z, c):
    z = np.array(z)
    v1, e1 = fhs_var_es(z, 0.0, 0.01, 0.975)
    v2, e2 = fhs_var_es(z, 0.0, 0.01 * c, 0.975)
    assert v2 == pytest.approx(c * v1, rel=1e-9, abs=1e-12)
    assert e2 == pytest.approx(c * e1, rel=1e-9, abs=1e-12)


def test_matches_gaussian_theory_for_large_sample():
    from scipy.stats import norm

    z = np.random.default_rng(0).standard_normal(400_000)
    var, es = fhs_var_es(z, 0.0, 1.0, 0.99)
    assert var == pytest.approx(norm.ppf(0.99), abs=0.03)
    assert es == pytest.approx(norm.pdf(norm.ppf(0.99)) / 0.01, abs=0.05)


def test_historical_baseline_properties():
    r = np.random.default_rng(1).standard_normal(5000) * 0.01
    v, e = historical_var_es(r, 0.975)
    assert e >= v > 0


@pytest.mark.parametrize("level", [0.5, 1.0, 0.2])
def test_level_validation(level):
    with pytest.raises(ValueError):
        fhs_var_es(np.ones(50), 0.0, 0.01, level)
    with pytest.raises(ValueError):
        historical_var_es(np.ones(50), level)
