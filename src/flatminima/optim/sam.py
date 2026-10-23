"""Sharpness-Aware Minimization (Foret, Kleiner, Mobahi and Neyshabur, ICLR 2021).

SAM solves  min_w  max_{||e||_2 <= rho} L(w + e)  with the first-order approximation of the
inner maximiser  e(w) = rho * grad L(w) / ||grad L(w)||_2.  One SAM step is:

    1. g  = grad L(w);              e = rho * g / ||g||
    2. g' = grad L(w + e)           (gradient at the perturbed point)
    3. w <- w - lr * update(g')     (w is restored to its unperturbed value first)

``update`` is whatever the wrapped base optimiser does (Adam here), so SAM costs two gradient
evaluations per step. With ``rho = 0`` it reduces exactly to the base optimiser.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class SAM:
    """Wrap a ``torch.optim.Optimizer`` with the SAM two-pass update.

    ``closure`` must clear the gradients, evaluate the loss, call ``backward()`` and return it.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05) -> None:
        if rho < 0:
            raise ValueError("rho must be non-negative")
        self.base = base_optimizer
        self.rho = float(rho)

    @property
    def params(self) -> list[torch.Tensor]:
        return [p for g in self.base.param_groups for p in g["params"] if p.requires_grad]

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        loss = closure()  # gradient at w
        params = [p for p in self.params if p.grad is not None]
        norm = torch.sqrt(sum((p.grad**2).sum() for p in params))
        scale = self.rho / (norm + 1e-12)
        eps = []
        with torch.no_grad():
            for p in params:
                e = p.grad * scale
                p.add_(e)
                eps.append(e)
        closure()  # gradient at w + e
        with torch.no_grad():
            for p, e in zip(params, eps, strict=True):
                p.sub_(e)
        self.base.step()
        return loss.detach()
