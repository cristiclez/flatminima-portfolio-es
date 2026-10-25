# CV evidence

Generated 2026-10-25 at commit `b755d9d`. Every number below is recomputed from the database by `python -m flatminima.report.evidence`; none is typed by hand.

Rule: a CV line may use only what is listed here, with its limits.

## BT-1000

- Claim: Rolling FHS-GJR-GARCH-t backtest, 1000-day window: 20 tests run, 4 rejections at 5% (not multiple-testing corrected).
  - EW: 45 VaR-99% exceptions in 4477 out-of-sample days (1.01% vs 1% nominal); Basel zones over rolling 250-day windows: green 89.6%, yellow 10.4%, red 0.0%
  - SPY: 45 VaR-99% exceptions in 4477 out-of-sample days (1.01% vs 1% nominal); Basel zones over rolling 250-day windows: green 95.6%, yellow 4.4%, red 0.0%
- Reproduce: `make risk && make backtest`
- Source: table `backtest_results_risk_forecasts`, `risk_forecasts`
- Do not say: Do not call this FRTB compliance; it is ES at 97.5% with backtesting ('FRTB-style').

## BT-500

- Claim: Rolling FHS-GJR-GARCH-t backtest, 500-day window: 20 tests run, 10 rejections at 5% (not multiple-testing corrected).
  - EW: 63 VaR-99% exceptions in 4977 out-of-sample days (1.27% vs 1% nominal); Basel zones over rolling 250-day windows: green 87.9%, yellow 12.1%, red 0.0%
  - SPY: 70 VaR-99% exceptions in 4977 out-of-sample days (1.41% vs 1% nominal); Basel zones over rolling 250-day windows: green 78.3%, yellow 21.7%, red 0.0%
- Reproduce: `make robustness`
- Source: table `backtest_results_risk_forecasts_w500`, `risk_forecasts_w500`
- Do not say: Do not call this FRTB compliance; it is ES at 97.5% with backtesting ('FRTB-style').

## SPECB-Direct-SAM-0.05

- Claim: Direct-SAM-0.05 minus Direct-Adam: Sharpe difference -0.02 (95% MBB CI -0.05 to +0.01), block 17, 5000 resamples, paired.
- Reproduce: `make data && make riskfree && make experiments`
- Source: table `specb_returns` (excess returns, net of 5 bps costs)
- Do not say: Specification B is post hoc. An interval that includes zero is no evidence of an improvement.

## SPECB-Direct-SAM-0.2

- Claim: Direct-SAM-0.2 minus Direct-Adam: Sharpe difference -0.02 (95% MBB CI -0.06 to +0.01), block 17, 5000 resamples, paired.
- Reproduce: `make data && make riskfree && make experiments`
- Source: table `specb_returns` (excess returns, net of 5 bps costs)
- Do not say: Specification B is post hoc. An interval that includes zero is no evidence of an improvement.

## SPECB-Policy-SAM-0.05

- Claim: Policy-SAM-0.05 minus Policy-Adam: Sharpe difference -0.02 (95% MBB CI -0.07 to +0.03), block 17, 5000 resamples, paired.
- Reproduce: `make data && make riskfree && make experiments`
- Source: table `specb_returns` (excess returns, net of 5 bps costs)
- Do not say: Specification B is post hoc. An interval that includes zero is no evidence of an improvement.

## SPECB-Policy-SAM-0.2

- Claim: Policy-SAM-0.2 minus Policy-Adam: Sharpe difference -0.02 (95% MBB CI -0.07 to +0.03), block 17, 5000 resamples, paired.
- Reproduce: `make data && make riskfree && make experiments`
- Source: table `specb_returns` (excess returns, net of 5 bps costs)
- Do not say: Specification B is post hoc. An interval that includes zero is no evidence of an improvement.

## SPECB-CURVATURE

- Claim: Median largest Hessian eigenvalue of the in-sample CVaR loss at the trained point, over walk-forward refits.
  - Direct-Adam: median lambda_max 3.61e-05
  - Direct-SAM-0.05: median lambda_max 4.79e-05
  - Direct-SAM-0.2: median lambda_max 5.15e-05
  - Policy-Adam: median lambda_max 0.000397
  - Policy-SAM-0.05: median lambda_max 0.000359
  - Policy-SAM-0.2: median lambda_max 0.000371
- Reproduce: `make experiments`
- Source: table `specb_hessian` (Lanczos on Hessian-vector products)
- Do not say: Do not claim SAM finds flatter minima unless the ratio is clearly below 1 for both allocators; no uncertainty interval is available.
