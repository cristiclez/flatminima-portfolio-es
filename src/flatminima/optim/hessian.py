"""Curvature tools based on Hessian-vector products (no explicit Hessian is formed).

``make_hvp`` builds H v by double back-propagation (Pearlmutter, 1994). On top of it:

* ``lanczos``: top Ritz values of H (Lanczos with full re-orthogonalisation),
* ``power_iteration``: dominant-magnitude eigenvalue,
* ``hutchinson_trace``: trace estimate with Rademacher probes,
* ``sharpness``: one-step SAM-style sharpness  L(w + rho g/||g||) - L(w).

Run in float64: with float32 the Hessian spectrum of a flat minimum is dominated by noise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

HVP = Callable[[torch.Tensor], torch.Tensor]


def _flatten(ts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def make_hvp(loss_fn: Callable[[], torch.Tensor], params: list[torch.Tensor]) -> tuple[HVP, int]:
    """Return (hvp, dim) where ``hvp(v) = H v`` for the Hessian of ``loss_fn`` w.r.t. ``params``."""
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_g = _flatten(list(grads))

    def hvp(v: torch.Tensor) -> torch.Tensor:
        out = torch.autograd.grad(flat_g @ v, params, retain_graph=True, allow_unused=True)
        return _flatten(
            [torch.zeros_like(p) if o is None else o for o, p in zip(out, params, strict=True)]
        )

    return hvp, flat_g.numel()


def lanczos(hvp: HVP, dim: int, k: int = 20, seed: int = 0, dtype=torch.float64) -> np.ndarray:
    """Top Ritz values (descending) from ``k`` Lanczos steps with full re-orthogonalisation.

    Unblocked Lanczos finds one Ritz value per distinct eigenvalue: repeated eigenvalues appear
    once, so the result can have fewer than ``k`` entries.
    """
    k = min(k, dim)
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=gen, dtype=dtype)
    q = q / q.norm()
    Q = [q]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(k):
        w = hvp(Q[j])
        a = float(w @ Q[j])
        alphas.append(a)
        w = w - a * Q[j] - (betas[-1] * Q[j - 1] if j > 0 else 0.0)
        for qi in Q:  # full re-orthogonalisation
            w = w - (w @ qi) * qi
        b = float(w.norm())
        if j == k - 1 or b < 1e-10:
            break
        betas.append(b)
        Q.append(w / b)
    m = len(alphas)
    T = np.diag(alphas) + np.diag(betas[: m - 1], 1) + np.diag(betas[: m - 1], -1)
    return np.sort(np.linalg.eigvalsh(T))[::-1]


def power_iteration(
    hvp: HVP, dim: int, iters: int = 200, tol: float = 1e-8, seed: int = 0, dtype=torch.float64
) -> float:
    """Eigenvalue of largest magnitude by power iteration (Rayleigh quotient)."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=gen, dtype=dtype)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        new = float(v @ w)
        nrm = float(w.norm())
        if nrm == 0.0:
            return 0.0
        v = w / nrm
        if abs(new - lam) < tol * max(1.0, abs(new)):
            return new
        lam = new
    return lam


def hutchinson_trace(
    hvp: HVP, dim: int, n_probes: int = 100, seed: int = 0, dtype=torch.float64
) -> float:
    """Hutchinson estimate of tr(H) with Rademacher probes: E[z' H z] = tr(H)."""
    gen = torch.Generator().manual_seed(seed)
    acc = 0.0
    for _ in range(n_probes):
        z = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).to(dtype)
        acc += float(z @ hvp(z))
    return acc / n_probes


def sharpness(loss_fn: Callable[[], torch.Tensor], params: list[torch.Tensor], rho: float) -> float:
    """L(w + rho g/||g||) - L(w): the first-order SAM sharpness at ``params`` (restored after)."""
    loss0 = loss_fn()
    grads = torch.autograd.grad(loss0, params)
    norm = torch.sqrt(sum((g**2).sum() for g in grads)) + 1e-12
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, g in zip(params, grads, strict=True):
            p.add_(rho * g / norm)
        loss1 = float(loss_fn())
        for p, s in zip(params, saved, strict=True):
            p.copy_(s)  # exact restoration (adding then subtracting is not exact in floating point)
    return loss1 - float(loss0.detach())
