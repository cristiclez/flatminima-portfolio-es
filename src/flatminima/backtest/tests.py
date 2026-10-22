"""VaR / ES backtests: Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely.

References
----------
Kupiec (1995), "Techniques for verifying the accuracy of risk measurement models".
Christoffersen (1998), "Evaluating interval forecasts".
Basel Committee (1996), "Supervisory framework for the use of backtesting ...".
Acerbi and Szekely (2014), "Backtesting Expected Shortfall", Risk Magazine.

``hits`` are boolean arrays with True on days when the realised loss exceeded the VaR.
Rejection of the null means the model is NOT adequate; "passing" means non-rejection at the
chosen significance level, which is weaker than proving the model correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy


@dataclass(frozen=True)
class TestResult:
    """Statistic and p-value of a backtest."""

    name: str
    statistic: float
    p_value: float
    n_obs: int
    n_exceptions: int

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _bern_ll(n_fail: float, n_ok: float, p: float) -> float:
    """Bernoulli log-likelihood with the 0*log(0)=0 convention."""
    return float(xlogy(n_ok, 1.0 - p) + xlogy(n_fail, p))


def kupiec_pof(hits: np.ndarray, coverage: float) -> TestResult:
    """Kupiec proportion-of-failures test (unconditional coverage).

    H0: exception probability equals ``1 - coverage``. LR ~ chi2(1) asymptotically.
    """
    h = np.asarray(hits, dtype=bool)
    n, x = len(h), int(h.sum())
    p = 1.0 - coverage
    lr = -2.0 * (_bern_ll(x, n - x, p) - _bern_ll(x, n - x, x / n))
    lr = max(lr, 0.0)
    return TestResult("Kupiec POF", lr, float(stats.chi2.sf(lr, 1)), n, x)


def christoffersen_independence(hits: np.ndarray) -> TestResult:
    """Christoffersen test of independence of exceptions (first-order Markov alternative)."""
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll0 = _bern_ll(n01 + n11, n00 + n10, pi)
    ll1 = _bern_ll(n01, n00, pi01) + _bern_ll(n11, n10, pi11)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return TestResult(
        "Christoffersen independence", lr, float(stats.chi2.sf(lr, 1)), len(h), int(h.sum())
    )


def christoffersen_cc(hits: np.ndarray, coverage: float) -> TestResult:
    """Conditional coverage: LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    pof = kupiec_pof(hits, coverage)
    ind = christoffersen_independence(hits)
    lr = pof.statistic + ind.statistic
    return TestResult(
        "Christoffersen CC", lr, float(stats.chi2.sf(lr, 2)), pof.n_obs, pof.n_exceptions
    )


def basel_zone(n_exceptions: int, n_obs: int = 250, coverage: float = 0.99) -> str:
    """Basel traffic-light zone from the binomial cumulative probability.

    green: cumulative probability < 95%; yellow: < 99.99%; red: otherwise. For 250
    observations at 99% this gives green 0-4, yellow 5-9, red 10+ exceptions.
    """
    cdf = float(stats.binom.cdf(n_exceptions, n_obs, 1.0 - coverage))
    if cdf < 0.95:
        return "green"
    return "yellow" if cdf < 0.9999 else "red"


def traffic_light_windows(
    hits: np.ndarray, window: int = 250, coverage: float = 0.99
) -> dict[str, float]:
    """Zone classification over all rolling ``window``-day windows of exceptions."""
    h = np.asarray(hits, dtype=int)
    if len(h) < window:
        raise ValueError("series shorter than the window")
    counts = np.convolve(h, np.ones(window, dtype=int), mode="valid")
    zones = np.array([basel_zone(int(c), window, coverage) for c in counts])
    return {
        "n_windows": float(len(counts)),
        "share_green": float(np.mean(zones == "green")),
        "share_yellow": float(np.mean(zones == "yellow")),
        "share_red": float(np.mean(zones == "red")),
        "max_exceptions": float(counts.max()),
        "last_window_exceptions": float(counts[-1]),
    }


@dataclass(frozen=True)
class ESTestResult:
    """Acerbi-Szekely Z1/Z2 with one-sided Monte Carlo p-values (H1: ES underestimated)."""

    z1: float
    z2: float
    p_z1: float
    p_z2: float
    n_obs: int
    n_exceptions: int
    n_sims: int


def acerbi_szekely(
    ret: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    z_tail: np.ndarray,
    window: int,
    tail_prob: float,
    n_sims: int = 10_000,
    seed: int = 0,
    batch: int = 500,
) -> ESTestResult:
    """Acerbi-Szekely Z1 and Z2 tests for ES at tail probability ``tail_prob``.

    With P&L X_t, exceedance I_t = 1[X_t < -VaR_t] and positive-loss ES_t:

        Z1 = sum_t X_t I_t / (N_T ES_t) + 1,     Z2 = sum_t X_t I_t / (T p ES_t) + 1,

    both with expectation 0 under H0 (forecast distributions correct); clearly negative
    values mean ES is underestimated. Significance is by Monte Carlo under H0: for each day
    the P&L is drawn from that day's own predictive distribution, i.e. the FHS empirical
    distribution mu_t + sigma_t * z, where only the lowest ``z_tail`` order statistics of the
    estimation window (size ``window``) are needed because only exceedances enter Z1/Z2.
    p-values are (1 + #{Z_sim <= Z_obs}) / (1 + n_sims).
    """
    ret, mu, sigma, var, es = (np.asarray(a, dtype=float) for a in (ret, mu, sigma, var, es))
    tails = np.asarray(z_tail, dtype=float)
    T = len(ret)
    p = tail_prob

    obs = ret < -var
    n_exc = int(obs.sum())
    if n_exc == 0:
        raise ValueError("no exceptions: Z1 undefined")
    s_obs = float(np.sum(ret[obs] / es[obs]))
    z1_obs = s_obs / n_exc + 1.0
    z2_obs = s_obs / (T * p) + 1.0

    rng = np.random.default_rng(seed)
    z1_sim, z2_sim = [], []
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        U = rng.random((b, T))
        si, ti = np.nonzero(U < p)
        pos = U[si, ti] * (window - 1)
        lo = pos.astype(int)
        frac = pos - lo
        zq = tails[ti, lo] * (1.0 - frac) + tails[ti, lo + 1] * frac
        x = mu[ti] + sigma[ti] * zq
        w = x / es[ti]
        s = np.bincount(si, weights=w, minlength=b)
        n = np.bincount(si, minlength=b)
        ok = n > 0
        z1_sim.append(s[ok] / n[ok] + 1.0)
        z2_sim.append(s / (T * p) + 1.0)
        done += b
    z1s, z2s = np.concatenate(z1_sim), np.concatenate(z2_sim)
    p1 = (1.0 + float(np.sum(z1s <= z1_obs))) / (1.0 + len(z1s))
    p2 = (1.0 + float(np.sum(z2s <= z2_obs))) / (1.0 + len(z2s))
    return ESTestResult(z1_obs, z2_obs, p1, p2, T, n_exc, n_sims)
