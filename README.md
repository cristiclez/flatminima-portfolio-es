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
| GARCH / GJR fitting and filtered historical simulation (VaR, ES 97.5%) | planned (week 2) |
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
make test               # pytest
make lint               # ruff
```

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