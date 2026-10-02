# flatminima-portfolio-es

**Portfolio Risk Engine & Sharpness-Aware Allocator** (work in progress)

A risk engine (GARCH / filtered historical simulation, VaR and Expected Shortfall with
regulatory-style backtesting) and a differentiable portfolio allocator trained with
Sharpness-Aware Minimization (SAM), benchmarked against Adam and convex baselines.

> **Status: v0.1 in progress.** Only the items marked "done" below exist. Nothing here
> claims regulatory (FRTB) compliance; "FRTB-style" means ES at 97.5% with backtesting.

## Status

| Component | Status |
|---|---|
| DuckDB data pipeline (ingest, quality checks, log returns) | done |
| Synthetic GARCH-t simulator with known ground truth | done |
| Tests (pytest + Hypothesis) and CI (ruff, pytest, coverage) | done (data layer) |
| GARCH / GJR-GARCH (Student-t) MLE, residual diagnostics, filtered historical simulation, rolling VaR 99% / ES 97.5% | done (validated on simulated data) |
| Backtesting: Kupiec, Christoffersen, Acerbi-Székely, Basel traffic light | planned (week 3) |
| Covariance: sample, Ledoit-Wolf, Random Matrix Theory | planned (week 4) |
| Convex baselines (cvxpy), walk-forward evaluation, Moving Block Bootstrap | planned (week 4) |
| Differentiable allocator, Adam vs SAM, Hessian-spectrum analysis | planned (week 5) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install            # editable install with dev + data extras
make data-synthetic     # offline demo on SYNTHETIC data (labelled as such)
make data               # real data from Yahoo Finance (needs internet)
make risk               # rolling VaR/ES forecasts (after `make data`)
make test               # pytest
make lint               # ruff
```

On Windows (PowerShell): `.venv\Scripts\activate`, and run the `python -m ...` commands from the Makefile directly if `make` is unavailable.

## Data

- **Real:** daily adjusted closes of 28 liquid US-listed ETFs from 2005 via Yahoo Finance
  (`yfinance`). Data is for personal/research use under Yahoo's terms and is **never
  committed**; `make data` rebuilds it.
- **Synthetic:** GARCH(1,1)-t simulations with known parameters, used to validate
  estimators and tests and to run the pipeline offline. Any result computed on synthetic
  data is labelled as such.

## Known limitations

- Survivorship bias: the universe contains only ETFs that exist today.
- Yahoo adjusted prices can contain errors; the quality checks flag, but do not fix, them.
- One extreme observation is flagged and deliberately retained: EWZ, 2020-03-16
  (log return approx. -0.26), consistent with same-day falls in SPY and EEM (COVID-19 crash).

## Risk engine (week 2)

**Model.** GARCH(1,1) and GJR-GARCH(1,1) with unit-variance Student-t innovations, exact
maximum likelihood (SLSQP, covariance-stationarity constraint), standard errors from the
numerical observed Hessian, and Hessian condition number reported per fit.

**Forecasts.** Rolling window of 1,000 days, refit every 20 days (warm start), one-step-ahead
filtered historical simulation (Barone-Adesi et al., 1999): VaR at 99% and 97.5%, and ES at
97.5% (the FRTB-style level). Losses are positive numbers on a unit-notional position, with
log returns as the P&L proxy. Unfiltered historical simulation is kept as a baseline.

**What has been verified (all on simulated data with known truth).**

- Parameter recovery: the mean MLE over 10 replications matches the true GARCH and GJR
  parameters within stated tolerances (`tests/test_garch.py`).
- Cross-check against the `arch` package (`scripts/validate_against_arch.py`): alpha, beta and
  gamma agree to about 1e-4; nu to about 1e-2.
- Ljung-Box and ARCH-LM implementations match `statsmodels` to machine precision.
- No look-ahead: changing returns from day k onwards leaves all earlier forecasts unchanged.
- Properties (Hypothesis): ES >= VaR at the same level, VaR increasing in confidence,
  scale equivariance, vectorised variance filter equals the textbook loop.

**Not yet done.** Formal backtests (week 3) and results on real market data are pending;
no real-data performance claims are made here.
