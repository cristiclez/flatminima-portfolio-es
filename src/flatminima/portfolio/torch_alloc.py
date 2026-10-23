"""Differentiable long-only allocators in PyTorch, trained with Adam or SAM.

Two allocators share the objective, the weight map and the training loop:

* ``fit_direct``: free logits theta in R^N, w = capped_softmax(theta) (a control: for a
  convex problem like minimum CVaR it should reproduce the cvxpy solution).
* ``fit_policy``: w_t = capped_softmax(f_theta(x_t)) where f is a small MLP shared across
  assets that maps past-only, cross-sectionally standardised features (trailing volatility and
  momentum) to scores. This is a model trained on a time window, so out-of-sample
  generalisation (and hence flat minima) is meaningful.

Objective (both): empirical CVaR at ``level`` of the portfolio daily loss, minus ``lam`` times
the mean return (``lam = 0``: pure tail-risk minimisation, comparable with the convex
minimum-CVaR baseline). Everything runs in float64.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from flatminima.optim.hessian import hutchinson_trace, lanczos, make_hvp, sharpness
from flatminima.optim.sam import SAM

DT = torch.float64
FEATURE_NAMES = ("vol21", "vol63", "ret21", "ret126")
FEATURE_WARMUP = 125  # first row with all features defined (126-day window)


# ------------------------------------------------------------------ weights and loss
def capped_softmax(z: torch.Tensor, cap: float | None = None) -> torch.Tensor:
    """Softmax over the last axis projected onto {w >= 0, sum w = 1, w <= cap}.

    Assets whose weight would exceed ``cap`` are locked at ``cap`` and the remaining mass is
    shared among the others in proportion to their softmax weights (repeated until no weight
    exceeds the cap; at most N rounds). Piecewise smooth, so autograd applies.
    """
    p = torch.softmax(z, dim=-1)
    if cap is None:
        return p
    n = z.shape[-1]
    if cap * n < 1.0 - 1e-12:
        raise ValueError(f"cap={cap} is infeasible for {n} assets (need cap >= 1/N)")
    locked = torch.zeros_like(p, dtype=torch.bool)
    w = p
    for _ in range(n):
        free = ~locked
        room = (1.0 - locked.sum(-1, keepdim=True).to(p.dtype) * cap).clamp_min(0.0)
        denom = (p * free).sum(-1, keepdim=True).clamp_min(1e-300)
        w = torch.where(locked, torch.full_like(p, cap), p * room / denom)
        newly = (w > cap * (1.0 + 1e-12)) & free
        if not bool(newly.any()):
            break
        locked = locked | newly
    return w


def cvar(losses: torch.Tensor, level: float = 0.975) -> torch.Tensor:
    """Empirical CVaR (expected shortfall) of ``losses`` over the last axis.

    Uses the Rockafellar-Uryasev weighting: the worst floor(alpha T) losses with weight 1 and
    the next one with the fractional remainder, divided by alpha T (alpha = 1 - level). It
    therefore equals the optimum of the cvxpy formulation for any sample size T.
    """
    t = losses.shape[-1]
    m = (1.0 - level) * t
    if m <= 0:
        raise ValueError("sample too small for this level")
    k = int(math.floor(m + 1e-9))
    frac = m - k
    if frac > 1e-9:
        top = torch.topk(losses, k + 1, dim=-1).values
        return (top[..., :k].sum(-1) + frac * top[..., k]) / m
    return torch.topk(losses, k, dim=-1).values.sum(-1) / m


def risk_loss(port_ret: torch.Tensor, level: float = 0.975, lam: float = 0.0) -> torch.Tensor:
    """CVaR of portfolio losses minus ``lam`` times the mean portfolio return."""
    return cvar(-port_ret, level) - lam * port_ret.mean()


# ------------------------------------------------------------------ training
@dataclass(frozen=True)
class TrainConfig:
    """Optimiser settings. ``grad_budget`` counts gradient evaluations, so SAM (two per step)
    takes ``grad_budget // 2`` steps and Adam ``grad_budget`` steps: equal compute."""

    optimizer: str = "adam"  # "adam" | "sam"
    rho: float = 0.05
    lr: float = 0.05
    grad_budget: int = 200

    @property
    def n_steps(self) -> int:
        if self.optimizer == "adam":
            return self.grad_budget
        if self.optimizer == "sam":
            return self.grad_budget // 2
        raise ValueError("optimizer must be 'adam' or 'sam'")


def train(
    params: list[torch.Tensor], loss_fn: Callable[[], torch.Tensor], cfg: TrainConfig
) -> float:
    """Minimise ``loss_fn`` over ``params``; returns the final loss at the unperturbed point."""
    base = torch.optim.Adam(params, lr=cfg.lr)
    sam = SAM(base, cfg.rho) if cfg.optimizer == "sam" else None

    def closure() -> torch.Tensor:
        base.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for _ in range(cfg.n_steps):
        if sam is not None:
            sam.step(closure)
        else:
            closure()
            base.step()
    with torch.no_grad():
        return float(loss_fn())


@dataclass
class Fit:
    """A trained allocator with what is needed to analyse its loss surface."""

    weights: np.ndarray
    loss: float
    params: list[torch.Tensor]
    loss_fn: Callable[[], torch.Tensor] = field(repr=False)


def analyse_fit(fit: Fit, rho_ref: float = 0.05, k: int = 12, n_probes: int = 30) -> dict:
    """Curvature summary at the trained point: top Hessian eigenvalues, trace, sharpness."""
    hvp, dim = make_hvp(fit.loss_fn, fit.params)
    ritz = lanczos(hvp, dim, k=min(k, dim))
    return {
        "train_loss": fit.loss,
        "lambda_max": float(ritz[0]),
        "lambda_2": float(ritz[1]) if len(ritz) > 1 else float("nan"),
        "trace": hutchinson_trace(hvp, dim, n_probes=n_probes),
        "sharpness": sharpness(fit.loss_fn, fit.params, rho_ref),
        "n_params": dim,
    }


# ------------------------------------------------------------------ direct allocator
def fit_direct(
    simple_returns: np.ndarray,
    cfg: TrainConfig,
    cap: float | None = None,
    level: float = 0.975,
    lam: float = 0.0,
) -> Fit:
    """Free-logit allocator on a window of simple returns (T x N); starts at equal weights."""
    r = torch.as_tensor(np.asarray(simple_returns), dtype=DT)
    theta = torch.zeros(r.shape[1], dtype=DT, requires_grad=True)

    def loss_fn() -> torch.Tensor:
        return risk_loss(r @ capped_softmax(theta, cap), level, lam)

    loss = train([theta], loss_fn, cfg)
    w = capped_softmax(theta.detach(), cap).numpy()
    return Fit(w, loss, [theta], loss_fn)


# ------------------------------------------------------------------ policy allocator
def build_features(log_returns: np.ndarray) -> np.ndarray:
    """Past-only features (T x N x F), cross-sectionally standardised at every date.

    Row s uses returns up to and including s. Rows before ``FEATURE_WARMUP`` are NaN.
    """
    df = pd.DataFrame(np.asarray(log_returns, dtype=float))
    ann = math.sqrt(252.0)
    raw = [
        df.rolling(21).std() * ann,
        df.rolling(63).std() * ann,
        df.rolling(21).sum(),
        df.rolling(126).sum(),
    ]
    x = np.stack([f.to_numpy() for f in raw], axis=-1)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


class PolicyNet(nn.Module):
    """Shared per-asset MLP: features (.., N, F) -> scores (.., N)."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 8) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.hidden(x))).squeeze(-1)


def fit_policy(
    window_log_returns: np.ndarray,
    cfg: TrainConfig,
    cap: float | None = None,
    hidden: int = 8,
    seed: int = 0,
    level: float = 0.975,
    lam: float = 0.0,
) -> tuple[Fit, PolicyNet, np.ndarray]:
    """Train the policy on a window of log returns; returns (fit, net, weights for the next day).

    Training pairs are (features at s, simple returns at s+1) for every s in the window with a
    complete feature set. The returned weights use the features at the LAST row.
    """
    r = np.asarray(window_log_returns, dtype=float)
    feats = build_features(r)
    simple = np.expm1(r)
    if len(r) < FEATURE_WARMUP + 40:
        raise ValueError("window too short for the feature warm-up")
    s = np.arange(FEATURE_WARMUP, len(r) - 1)
    x = torch.as_tensor(feats[s], dtype=DT)
    y = torch.as_tensor(simple[s + 1], dtype=DT)
    torch.manual_seed(seed)
    net = PolicyNet(hidden=hidden).to(DT)
    params = list(net.parameters())

    def loss_fn() -> torch.Tensor:
        w = capped_softmax(net(x), cap)
        return risk_loss((w * y).sum(-1), level, lam)

    loss = train(params, loss_fn, cfg)
    with torch.no_grad():
        last = torch.as_tensor(feats[-1], dtype=DT)
        w_next = capped_softmax(net(last), cap).numpy()
    return Fit(w_next, loss, params, loss_fn), net, w_next


# ------------------------------------------------------------------ walk-forward strategies
class DirectStrategy:
    """Walk-forward wrapper: re-fits the free-logit allocator at every rebalance."""

    def __init__(
        self,
        cfg: TrainConfig,
        cap: float | None,
        est_window: int = 756,
        analyse_every: int = 0,
        rho_ref: float = 0.05,
    ) -> None:
        self.cfg, self.cap, self.est_window = cfg, cap, est_window
        self.analyse_every, self.rho_ref = analyse_every, rho_ref
        self.calls = 0
        self.log: list[dict] = []

    def __call__(self, window: np.ndarray) -> np.ndarray:
        fit = fit_direct(np.expm1(window[-self.est_window :]), self.cfg, self.cap)
        if self.analyse_every and self.calls % self.analyse_every == 0:
            self.log.append({"call": self.calls, **analyse_fit(fit, self.rho_ref)})
        self.calls += 1
        return fit.weights


class PolicyStrategy:
    """Walk-forward wrapper: re-trains the policy every ``retrain_every`` rebalances and
    re-computes the weights from fresh features at every rebalance."""

    def __init__(
        self,
        cfg: TrainConfig,
        cap: float | None,
        retrain_every: int = 3,
        hidden: int = 8,
        seed: int = 0,
        analyse_every: int = 0,
        rho_ref: float = 0.05,
    ) -> None:
        self.cfg, self.cap, self.retrain_every = cfg, cap, retrain_every
        self.hidden, self.seed = hidden, seed
        self.analyse_every, self.rho_ref = analyse_every, rho_ref
        self.calls = 0
        self.n_trained = 0
        self.net: PolicyNet | None = None
        self.log: list[dict] = []

    def __call__(self, window: np.ndarray) -> np.ndarray:
        if self.net is None or self.calls % self.retrain_every == 0:
            fit, self.net, w = fit_policy(
                window, self.cfg, self.cap, self.hidden, self.seed + self.n_trained
            )
            if self.analyse_every and self.n_trained % self.analyse_every == 0:
                self.log.append({"call": self.calls, **analyse_fit(fit, self.rho_ref)})
            self.n_trained += 1
        else:
            feats = build_features(window)
            with torch.no_grad():
                w = capped_softmax(self.net(torch.as_tensor(feats[-1], dtype=DT)), self.cap).numpy()
        self.calls += 1
        return w
