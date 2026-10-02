import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatminima.data.synthetic import simulate_garch_t
from flatminima.risk.garch import GarchParams, conditional_variance, fit_garch, log_likelihood

TRUE_GJR = dict(omega=2e-6, alpha=0.03, gamma=0.08, beta=0.90, nu=6.0, mu=3e-4)
TRUE_GARCH = dict(omega=2e-6, alpha=0.08, gamma=0.0, beta=0.90, nu=6.0, mu=3e-4)


def _as_params(d):
    return GarchParams(d["mu"], d["omega"], d["alpha"], d["gamma"], d["beta"], d["nu"])


@pytest.mark.parametrize(
    "model,true,tol",
    [
        ("gjr", TRUE_GJR, dict(alpha=0.02, gamma=0.03, beta=0.03, nu=0.8)),
        ("garch", TRUE_GARCH, dict(alpha=0.02, beta=0.03, nu=0.8)),
    ],
)
def test_parameter_recovery(model, true, tol):
    """Mean of the MLE over replications must be close to the true parameters."""
    ests = []
    for seed in range(10):
        r, _ = simulate_garch_t(6000, rng=np.random.default_rng(seed), **true)
        f = fit_garch(r, model, compute_se=False)
        assert f.converged
        ests.append(f.params)
    for name, tolerance in tol.items():
        mean_est = np.mean([getattr(p, name) for p in ests])
        assert mean_est == pytest.approx(true[name], abs=tolerance), name


def test_fit_is_at_least_as_good_as_truth():
    r, _ = simulate_garch_t(5000, rng=np.random.default_rng(3), **TRUE_GJR)
    f = fit_garch(r, "gjr", compute_se=False)
    assert f.loglik >= log_likelihood(_as_params(TRUE_GJR), r) - 1e-6


def test_stationarity_and_positivity_of_fit():
    r, _ = simulate_garch_t(4000, rng=np.random.default_rng(4), **TRUE_GJR)
    p = fit_garch(r, "gjr", compute_se=False).params
    assert p.omega > 0 and p.persistence < 1 and p.nu > 2


def test_standard_errors_and_condition_number():
    r, _ = simulate_garch_t(6000, rng=np.random.default_rng(5), **TRUE_GJR)
    f = fit_garch(r, "gjr")
    assert all(np.isfinite(v) and v > 0 for v in f.std_errors.values())
    assert f.hessian_condition_number > 1.0


def test_warm_start_reaches_same_optimum():
    r, _ = simulate_garch_t(4000, rng=np.random.default_rng(6), **TRUE_GJR)
    cold = fit_garch(r, "gjr", compute_se=False)
    warm = fit_garch(r, "gjr", start=cold.params, compute_se=False)
    assert warm.loglik == pytest.approx(cold.loglik, abs=1e-3)


def test_information_criteria_formulae():
    r, _ = simulate_garch_t(3000, rng=np.random.default_rng(7), **TRUE_GARCH)
    f = fit_garch(r, "garch", compute_se=False)
    assert f.n_free_params == 5
    assert f.aic == pytest.approx(10 - 2 * f.loglik)
    assert f.bic == pytest.approx(5 * np.log(3000) - 2 * f.loglik)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        fit_garch(np.zeros(500), "arma")
    with pytest.raises(ValueError):
        fit_garch(np.zeros(50), "garch")


@settings(max_examples=30, deadline=None)
@given(
    alpha=st.floats(0.01, 0.2),
    gamma=st.floats(0.0, 0.2),
    beta=st.floats(0.3, 0.7),
    seed=st.integers(0, 10_000),
)
def test_filter_matches_naive_loop(alpha, gamma, beta, seed):
    """Property: the vectorised lfilter recursion equals the textbook for-loop."""
    r = np.random.default_rng(seed).standard_normal(200) * 0.01
    p = GarchParams(mu=1e-4, omega=1e-6, alpha=alpha, gamma=gamma, beta=beta, nu=8.0)
    s2, s2_next = conditional_variance(p, r)
    e = r - p.mu
    ref = np.empty(len(r) + 1)
    ref[0] = np.var(r)
    for t in range(1, len(r) + 1):
        ref[t] = (
            p.omega + (p.alpha + p.gamma * (e[t - 1] < 0)) * e[t - 1] ** 2 + p.beta * ref[t - 1]
        )
    np.testing.assert_allclose(s2, ref[:-1], rtol=1e-10)
    assert s2_next == pytest.approx(ref[-1], rel=1e-10)
