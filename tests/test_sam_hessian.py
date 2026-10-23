import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flatminima.optim.hessian import (  # noqa: E402
    hutchinson_trace,
    lanczos,
    make_hvp,
    power_iteration,
    sharpness,
)
from flatminima.optim.sam import SAM  # noqa: E402

DT = torch.float64


def _quadratic(eigs, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((len(eigs), len(eigs))))
    a = torch.tensor(q @ np.diag(eigs) @ q.T, dtype=DT)
    return 0.5 * (a + a.T)


# ------------------------------------------------------------------ Hessian tools
def test_lanczos_power_and_trace_recover_known_spectrum():
    eigs = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    a = _quadratic(eigs)
    w = torch.randn(6, dtype=DT, requires_grad=True)
    hvp, dim = make_hvp(lambda: 0.5 * w @ a @ w, [w])
    np.testing.assert_allclose(lanczos(hvp, dim, k=6), eigs, rtol=1e-9)
    partial = lanczos(hvp, dim, k=4)  # 4 of 6 steps: the dominant value converges first
    assert partial[0] == pytest.approx(10.0, rel=1e-3)
    assert partial[1] == pytest.approx(5.0, rel=0.05)
    assert power_iteration(hvp, dim, iters=500) == pytest.approx(10.0, rel=1e-5)
    assert hutchinson_trace(hvp, dim, n_probes=3000) == pytest.approx(eigs.sum(), rel=0.05)


def test_hvp_matches_explicit_hessian_for_nonquadratic_loss():
    def f(x):
        return (torch.sin(x) * x**2).sum() + (x[0] * x[1] * x[2]) ** 2 + 0.1 * (x**4).sum()

    x = torch.tensor([0.7, -1.1, 0.4, 0.9], dtype=DT, requires_grad=True)
    H = torch.autograd.functional.hessian(f, x.detach())
    hvp, dim = make_hvp(lambda: f(x), [x])
    ref = np.sort(np.linalg.eigvalsh(H.numpy()))[::-1]
    np.testing.assert_allclose(lanczos(hvp, dim, k=4), ref, rtol=1e-8, atol=1e-10)
    v = torch.randn(4, dtype=DT)
    np.testing.assert_allclose(hvp(v).detach().numpy(), (H @ v).numpy(), rtol=1e-9, atol=1e-10)


def test_hvp_handles_multiple_parameter_tensors():
    p1 = torch.tensor([1.0, 2.0], dtype=DT, requires_grad=True)
    p2 = torch.tensor([[0.5, -0.3]], dtype=DT, requires_grad=True)
    hvp, dim = make_hvp(lambda: (p1**2).sum() * 3 + (p2**2).sum(), [p1, p2])
    assert dim == 4
    # Unblocked Lanczos returns one Ritz value per DISTINCT eigenvalue (multiplicity is lost).
    np.testing.assert_allclose(lanczos(hvp, dim, k=4), [6.0, 2.0], atol=1e-9)


def test_sharpness_of_quadratic_matches_closed_form_and_restores_params():
    a = _quadratic(np.array([4.0, 1.0]))
    w = torch.tensor([0.3, -0.2], dtype=DT, requires_grad=True)
    before = w.detach().clone()
    rho = 0.1
    g = a @ before
    e = rho * g / g.norm()
    expected = float(0.5 * (before + e) @ a @ (before + e) - 0.5 * before @ a @ before)
    got = sharpness(lambda: 0.5 * w @ a @ w, [w], rho)
    assert got == pytest.approx(expected, rel=1e-9)
    assert torch.equal(w.detach(), before)


# ------------------------------------------------------------------ SAM
def test_sam_step_matches_closed_form_on_quadratic():
    a = _quadratic(np.array([3.0, 1.0, 0.5]), seed=2)
    w = torch.tensor([1.0, -0.5, 0.8], dtype=DT, requires_grad=True)
    w0, lr, rho = w.detach().clone(), 0.05, 0.2
    base = torch.optim.SGD([w], lr=lr)

    def closure():
        base.zero_grad()
        loss = 0.5 * w @ a @ w
        loss.backward()
        return loss

    SAM(base, rho).step(closure)
    g = a @ w0
    eps = rho * g / g.norm()
    expected = w0 - lr * (a @ (w0 + eps))
    np.testing.assert_allclose(w.detach().numpy(), expected.numpy(), rtol=1e-10)


def test_sam_with_zero_rho_equals_adam():
    a = _quadratic(np.array([2.0, 1.0, 0.3, 0.1]), seed=3)
    outs = []
    for use_sam in (False, True):
        w = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=DT, requires_grad=True)
        base = torch.optim.Adam([w], lr=0.05)

        def closure(w=w, base=base):
            base.zero_grad()
            loss = 0.5 * w @ a @ w
            loss.backward()
            return loss

        opt = SAM(base, 0.0)
        for _ in range(25):
            opt.step(closure) if use_sam else (closure(), base.step())
        outs.append(w.detach().clone())
    np.testing.assert_allclose(outs[0].numpy(), outs[1].numpy(), rtol=1e-12)


def test_sam_rejects_negative_rho():
    w = torch.zeros(2, dtype=DT, requires_grad=True)
    with pytest.raises(ValueError):
        SAM(torch.optim.SGD([w], lr=0.1), rho=-1.0)


def _train_product_model(use_sam, rho=0.2, steps=6000, lr=0.01):
    """L = 0.5 (a b - 1)^2: a manifold of global minima with Hessian lambda_max = a^2 + b^2."""
    ab = torch.tensor([3.0, 0.2], dtype=DT, requires_grad=True)
    base = torch.optim.SGD([ab], lr=lr)

    def loss_fn():
        return 0.5 * (ab[0] * ab[1] - 1.0) ** 2

    def closure():
        base.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt = SAM(base, rho) if use_sam else None
    for _ in range(steps):
        opt.step(closure) if opt else (closure(), base.step())
    hvp, dim = make_hvp(loss_fn, [ab])
    return float(loss_fn().detach()), float(lanczos(hvp, dim, k=2)[0])


def test_sam_reaches_a_flatter_minimum_than_sgd_on_degenerate_minima():
    loss_sgd, lam_sgd = _train_product_model(False)
    loss_sam, lam_sam = _train_product_model(True)
    assert loss_sgd < 1e-3 and loss_sam < 1e-3  # both are minima
    assert lam_sgd > 8.0  # SGD keeps the sharp initial imbalance (a^2 - b^2 is conserved)
    assert lam_sam < 0.5 * lam_sgd  # SAM drifts to the balanced, flat point (lambda_max -> 2)
