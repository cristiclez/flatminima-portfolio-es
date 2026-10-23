import math

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from flatminima.data.synthetic import simulate_panel  # noqa: E402
from flatminima.portfolio.optimizers import min_cvar  # noqa: E402
from flatminima.portfolio.torch_alloc import (  # noqa: E402
    DT,
    FEATURE_WARMUP,
    DirectStrategy,
    PolicyStrategy,
    TrainConfig,
    analyse_fit,
    build_features,
    capped_softmax,
    cvar,
    fit_direct,
    fit_policy,
    risk_loss,
)
from flatminima.portfolio.walkforward import walk_forward  # noqa: E402


# ------------------------------------------------------------------ capped softmax
@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(2, 15),
    mult=st.floats(1.0, 6.0),
    seed=st.integers(0, 10_000),
    scale=st.floats(0.1, 8.0),
)
def test_capped_softmax_is_a_feasible_monotone_allocation(n, mult, seed, scale):
    cap = min(1.0, mult / n)
    z = torch.as_tensor(np.random.default_rng(seed).standard_normal(n) * scale, dtype=DT)
    w = capped_softmax(z, cap)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(w.min()) >= 0.0 and float(w.max()) <= cap * (1 + 1e-9)
    order = torch.argsort(z)
    assert (torch.diff(w[order]) >= -1e-12).all()  # higher score never gets less weight


def test_capped_softmax_special_cases_and_validation():
    z = torch.tensor([0.3, -1.0, 2.0, 0.5], dtype=DT)
    assert torch.allclose(capped_softmax(z, None), torch.softmax(z, -1))
    assert torch.allclose(capped_softmax(z, 1.0), torch.softmax(z, -1))
    assert torch.allclose(capped_softmax(z, 0.25), torch.full_like(z, 0.25))
    with pytest.raises(ValueError):
        capped_softmax(z, 0.2)


def test_capped_softmax_matches_hand_computation():
    p = torch.tensor([0.7, 0.2, 0.1], dtype=DT)
    w = capped_softmax(torch.log(p), 0.5)
    # asset 0 is locked at 0.5; the remaining 0.5 is shared 2:1
    np.testing.assert_allclose(w.numpy(), [0.5, 0.5 * 2 / 3, 0.5 / 3], atol=1e-12)


def test_capped_softmax_batched_equals_rowwise_and_has_finite_gradient():
    z = torch.randn(5, 9, dtype=DT, requires_grad=True)
    w = capped_softmax(z, 0.2)
    rows = torch.stack([capped_softmax(z[i], 0.2) for i in range(5)])
    assert torch.allclose(w, rows)
    (w * torch.arange(9, dtype=DT)).sum().backward()
    assert torch.isfinite(z.grad).all()
    z2 = z.detach().clone().requires_grad_(True)
    capped_softmax(z2, 0.2).sum().backward()
    assert z2.grad.abs().max() < 1e-12  # the budget is constant


# ------------------------------------------------------------------ CVaR
def _cvar_numpy(loss, level):
    x = np.sort(loss)[::-1]
    m = (1 - level) * len(x)
    k = int(math.floor(m + 1e-9))
    frac = m - k
    return (x[:k].sum() + (frac * x[k] if frac > 1e-9 else 0.0)) / m


@pytest.mark.parametrize("t", [400, 756, 1000])
def test_cvar_matches_numpy_and_the_rockafellar_uryasev_optimum(t):
    loss = np.random.default_rng(t).standard_normal(t) * 0.01
    got = float(cvar(torch.as_tensor(loss, dtype=DT), 0.975))
    assert got == pytest.approx(_cvar_numpy(loss, 0.975), rel=1e-12)
    a, u = cp.Variable(), cp.Variable(t)
    prob = cp.Problem(cp.Minimize(a + cp.sum(u) / ((1 - 0.975) * t)), [u >= loss - a, u >= 0])
    prob.solve(solver=cp.CLARABEL)
    assert got == pytest.approx(prob.value, rel=1e-6)


def test_risk_loss_combines_cvar_and_mean_return():
    r = torch.as_tensor(np.random.default_rng(0).standard_normal(400) * 0.01, dtype=DT)
    base = risk_loss(r, 0.975, 0.0)
    assert float(risk_loss(r, 0.975, 2.0)) == pytest.approx(float(base) - 2.0 * float(r.mean()))


# ------------------------------------------------------------------ training
def _scenarios(n=28, t=400, seed=5):
    df = simulate_panel([f"A{i}" for i in range(n)], n=t + 1, start="2010-01-04", seed=seed)
    lr = np.log(df.pivot(index="date", columns="ticker", values="adj_close")).diff().dropna()
    return lr.to_numpy()


def test_direct_adam_reproduces_the_convex_min_cvar_solution():
    r = np.expm1(_scenarios())
    ref = min_cvar(r, 0.975, cap=0.10)
    fit = fit_direct(r, TrainConfig("adam", lr=0.05, grad_budget=200), cap=0.10)

    def obj(w):
        return float(cvar(torch.as_tensor(-(r @ w), dtype=DT), 0.975))

    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-8) and fit.weights.max() <= 0.10 + 1e-9
    assert obj(fit.weights) <= obj(ref) * 1.01  # within 1% of the convex optimum
    assert obj(fit.weights) >= obj(ref) * (1 - 1e-6)  # and never below it (it is the optimum)


def test_gradient_budget_is_equal_across_optimisers():
    assert TrainConfig("adam", grad_budget=200).n_steps == 200
    assert TrainConfig("sam", grad_budget=200).n_steps == 100
    with pytest.raises(ValueError):
        _ = TrainConfig("sgd").n_steps


def test_sam_with_zero_rho_and_double_budget_equals_adam():
    r = np.expm1(_scenarios(t=300))
    a = fit_direct(r, TrainConfig("adam", grad_budget=60), cap=0.2)
    s = fit_direct(r, TrainConfig("sam", rho=0.0, grad_budget=120), cap=0.2)
    np.testing.assert_allclose(a.weights, s.weights, atol=1e-10)


def test_sam_changes_the_solution_when_rho_is_positive():
    r = np.expm1(_scenarios(t=300))
    a = fit_direct(r, TrainConfig("adam", grad_budget=60), cap=0.2)
    s = fit_direct(r, TrainConfig("sam", rho=0.3, grad_budget=120), cap=0.2)
    assert np.abs(a.weights - s.weights).max() > 1e-4


# ------------------------------------------------------------------ features and policy
def test_features_are_past_only_standardised_and_nan_before_warmup():
    r = _scenarios(t=400)
    f = build_features(r)
    assert f.shape == (len(r), 28, 4)
    assert np.isnan(f[:FEATURE_WARMUP]).any(axis=(1, 2)).all()
    assert not np.isnan(f[FEATURE_WARMUP:]).any()
    np.testing.assert_allclose(f[FEATURE_WARMUP:].mean(axis=1), 0.0, atol=1e-10)
    r2 = r.copy()
    r2[300:] = np.random.default_rng(0).standard_normal((len(r) - 300, 28)) * 0.05
    np.testing.assert_allclose(build_features(r2)[:300], f[:300], equal_nan=True)  # no look-ahead


def _hetero_returns(t=500, n=28, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.linspace(0.004, 0.02, n)
    return rng.standard_normal((t, 1)) * 0.006 + rng.standard_normal((t, n)) * scale, scale


def test_policy_beats_equal_weight_in_sample_and_tilts_to_low_volatility_on_average():
    r, scale = _hetero_returns(t=882)
    x = torch.as_tensor(np.expm1(r)[FEATURE_WARMUP + 1 :], dtype=DT)
    eq = float(risk_loss(x @ torch.full((28,), 1 / 28, dtype=DT), 0.975))
    tilts = []
    for seed in (1, 2, 3):
        fit, _, w = fit_policy(
            r, TrainConfig("adam", lr=0.05, grad_budget=200), cap=0.10, seed=seed
        )
        assert fit.loss < 0.95 * eq
        assert w.sum() == pytest.approx(1.0, abs=1e-8) and w.max() <= 0.10 + 1e-9
        tilts.append(float(w @ scale))
    # a single seed can miss the tilt (noisy fit); averaged over seeds it is clearly below 1/N
    assert np.mean(tilts) < scale.mean() - 0.0005


def test_policy_is_permutation_equivariant_and_reproducible():
    r, _ = _hetero_returns(t=330)
    cfg = TrainConfig("sam", rho=0.1, lr=0.05, grad_budget=40)
    _, _, w = fit_policy(r, cfg, cap=0.2, seed=3)
    perm = np.random.default_rng(1).permutation(28)
    _, _, wp = fit_policy(r[:, perm], cfg, cap=0.2, seed=3)
    np.testing.assert_allclose(wp, w[perm], atol=1e-6)
    _, _, w2 = fit_policy(r, cfg, cap=0.2, seed=3)
    np.testing.assert_allclose(w, w2, atol=1e-12)
    _, _, w3 = fit_policy(r, cfg, cap=0.2, seed=4)
    assert np.abs(w - w3).max() > 1e-6


def test_policy_rejects_short_windows():
    with pytest.raises(ValueError):
        fit_policy(np.zeros((100, 5)), TrainConfig(), cap=0.5)


def test_analyse_fit_reports_curvature():
    r, _ = _hetero_returns(t=330)
    fit, _, _ = fit_policy(r, TrainConfig("adam", grad_budget=60), cap=0.2, seed=0)
    a = analyse_fit(fit, rho_ref=0.05)
    assert a["n_params"] == 49 and a["lambda_max"] >= a["lambda_2"]
    assert a["sharpness"] >= 0.0 and np.isfinite(a["trace"])


# ------------------------------------------------------------------ walk-forward integration
def test_strategies_run_inside_walk_forward_with_valid_weights_and_logs():
    df = simulate_panel([f"A{i}" for i in range(12)], n=1000, start="2010-01-04", seed=2)
    lr = np.log(df.pivot(index="date", columns="ticker", values="adj_close")).diff().dropna()
    cfg = TrainConfig("sam", rho=0.1, grad_budget=30)
    strategies = {
        "Direct": DirectStrategy(cfg, cap=0.2, est_window=300, analyse_every=1),
        "Policy": PolicyStrategy(cfg, cap=0.2, retrain_every=3, analyse_every=1),
    }
    res = walk_forward(lr, strategies, lookback=420, rebalance_every=21)
    n_calls = len(res.weights["Policy"])
    for name in strategies:
        w = res.weights[name]
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)
        assert w.max() <= 0.2 + 1e-9 and w.min() >= -1e-12
    assert strategies["Policy"].n_trained == math.ceil(n_calls / 3)
    assert len(strategies["Policy"].log) == strategies["Policy"].n_trained
    assert len(strategies["Direct"].log) == n_calls
