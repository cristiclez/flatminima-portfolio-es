# Portfolio baselines (walk-forward)

Data source: `yahoo` (28 ETFs). Out-of-sample: 2006-01-04 to 2026-10-12 (5225 days). Lookback 252 days, rebalance every 21 days, long-only, fully invested, cost 5.0 bps per unit traded notional. Sharpe = annualised mean/std of net returns (no risk-free rate); intervals: Moving Block Bootstrap, block 17, 5000 resamples, 95% percentile. Not multiple-testing corrected.

| Strategy | Ann. ret | Ann. vol | Sharpe (95% CI) | vs EW (95% CI) | Max DD | Turnover/yr | Weight std | Eff. N |
|---|---|---|---|---|---|---|---|---|
| EqualWeight | 9.3% | 14.7% | 0.63 (0.23, 1.06) | +0.00 (+0.00, +0.00) | -43.6% | 0.34 | 0.000 | 28.0 |
| MinVar-Sample | 2.0% | 1.4% | 1.41 (0.99, 1.83) | +0.77 (+0.30, +1.22) | -5.9% | 0.15 | 0.180 | 1.1 |
| MinVar-LedoitWolf | 2.8% | 2.5% | 1.13 (0.61, 1.66) | +0.49 (+0.11, +0.90) | -8.5% | 0.88 | 0.123 | 2.4 |
| MinVar-RMT | 2.0% | 1.5% | 1.36 (0.93, 1.79) | +0.72 (+0.26, +1.16) | -5.8% | 0.16 | 0.180 | 1.1 |
| MinCVaR-97.5 | 2.2% | 1.5% | 1.44 (1.02, 1.86) | +0.80 (+0.34, +1.23) | -6.0% | 0.37 | 0.179 | 1.1 |

    
## Reading these results

- Sharpe ratios are annualised mean/std of net returns **without subtracting a risk-free rate**.
- The minimum-variance and minimum-CVaR portfolios are extremely concentrated (effective number of assets about 1.1). The sample-covariance minimum-variance portfolio held on average 97% in SHY (1-3 year US Treasuries), i.e. it is essentially a cash-like position, and its Sharpe ratio is not comparable with the equal-weight portfolio. The "vs EW" intervals must not be read as evidence that these strategies add value.
- The Sharpe intervals of the three covariance estimators overlap; no estimator is shown to be better.
- No conclusion about SAM or robustness is drawn from this table.