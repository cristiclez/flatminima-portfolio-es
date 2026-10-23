# Specification B: Adam vs SAM allocators (post hoc)

> **Specification B was defined after specification A's results were seen** (`docs/portfolio_baselines.md`), because A degenerated into a Treasury-ETF position. It is post hoc. A is kept unchanged. Nothing below is a claim of out-of-sample alpha.

Data: `yahoo`; risk-free rate: FRED DTB3 (1954-01-04 to 2026-10-21), approximated as DTB3/100/252 per day. All returns are excess log returns. Out-of-sample: 2008-07-08 to 2026-10-12 (4595 days). Lookback 882 days (756 for estimation, 126 for the policy features), rebalance every 21 days, long-only, cap 10% per asset, cost 5 bps per unit traded. Adam and SAM use the same number of gradient evaluations (200); SAM rho preregistered in [0.05, 0.2]. Policy networks: seeds [0, 1, 2]; the table below uses seed 0. Sharpe intervals: Moving Block Bootstrap, block 17, 5000 resamples. Not multiple-testing corrected.

## Main table

| strategy | ann_excess_return | ann_vol | Sharpe (95% CI) | oos_cvar_975 | max_drawdown | ann_turnover | effective_n |
|---|---|---|---|---|---|---|---|
| EqualWeight | 8.1% | 15.0% | 0.54 (0.12, 0.99) | 2.99% | -38.3% | 0.34 | 28.0 |
| MinVar-Sample(cap) | 4.7% | 6.2% | 0.76 (0.25, 1.27) | 1.14% | -19.0% | 0.45 | 10.8 |
| MinCVaR-97.5(cap) | 4.5% | 6.2% | 0.72 (0.23, 1.23) | 1.14% | -19.1% | 0.79 | 10.7 |
| Direct-Adam | 4.4% | 6.3% | 0.70 (0.20, 1.21) | 1.15% | -19.2% | 0.60 | 11.1 |
| Direct-SAM-0.05 | 4.3% | 6.3% | 0.68 (0.19, 1.19) | 1.17% | -19.6% | 0.49 | 11.5 |
| Direct-SAM-0.2 | 4.3% | 6.3% | 0.68 (0.18, 1.19) | 1.17% | -19.5% | 0.50 | 11.5 |
| Policy-Adam | 4.0% | 6.8% | 0.59 (0.11, 1.08) | 1.29% | -18.6% | 4.85 | 11.2 |
| Policy-SAM-0.05 | 3.8% | 6.8% | 0.56 (0.09, 1.07) | 1.29% | -18.2% | 4.55 | 11.2 |
| Policy-SAM-0.2 | 3.8% | 6.8% | 0.57 (0.09, 1.06) | 1.29% | -18.5% | 4.53 | 11.2 |

`oos_cvar_975` = mean daily loss over the worst 2.5% of out-of-sample days.

## Paired Sharpe difference, SAM minus Adam (main seed)

| comparison | diff | lo | hi |
|---|---|---|---|
| Direct-SAM-0.05 - Direct-Adam | -0.02 | -0.05 | +0.01 |
| Direct-SAM-0.2 - Direct-Adam | -0.02 | -0.06 | +0.01 |
| Policy-SAM-0.05 - Policy-Adam | -0.02 | -0.07 | +0.03 |
| Policy-SAM-0.2 - Policy-Adam | -0.02 | -0.07 | +0.03 |

## Robustness across policy seeds

| strategy | sharpe_mean | sharpe_sd | cvar_mean | cvar_sd | n |
|---|---|---|---|---|---|
| Policy-Adam | 0.58 | 0.03 | 1.304% | 0.009% | 3 |
| Policy-SAM-0.05 | 0.56 | 0.01 | 1.301% | 0.009% | 3 |
| Policy-SAM-0.2 | 0.57 | 0.01 | 1.292% | 0.004% | 3 |

## Loss-surface curvature at the trained point (medians over refits)

| strategy | train_loss | lambda_max | trace | sharpness | n |
|---|---|---|---|---|---|
| Direct-Adam | 0.01004 | 3.61e-05 | 7.11e-05 | 5.18e-06 | 37 |
| Direct-SAM-0.05 | 0.01016 | 4.79e-05 | 0.000115 | 8.33e-06 | 37 |
| Direct-SAM-0.2 | 0.01016 | 5.15e-05 | 0.000108 | 7.76e-06 | 37 |
| Policy-Adam | 0.01020 | 0.000397 | 0.00074 | 1.77e-05 | 39 |
| Policy-SAM-0.05 | 0.01036 | 0.000359 | 0.000694 | 1.39e-05 | 39 |
| Policy-SAM-0.2 | 0.01048 | 0.000371 | 0.000134 | 9.28e-06 | 39 |

Curvature is measured on the training loss (in-sample CVaR) at the fitted parameters; sharpness is the loss increase for a one-step ascent of radius 0.05.

## Reading these results

- Direct-SAM-0.05: median lambda_max is 1.33x that of Direct-Adam (not flatter by this measure).
- Direct-SAM-0.2: median lambda_max is 1.43x that of Direct-Adam (not flatter by this measure).
- Policy-SAM-0.05: median lambda_max is 0.90x that of Policy-Adam (flatter by this measure).
- Policy-SAM-0.2: median lambda_max is 0.93x that of Policy-Adam (flatter by this measure).
- Direct-SAM-0.05 - Direct-Adam: Sharpe difference -0.02 (95% CI -0.05 to +0.01); the interval includes zero.
- Direct-SAM-0.2 - Direct-Adam: Sharpe difference -0.02 (95% CI -0.06 to +0.01); the interval includes zero.
- Policy-SAM-0.05 - Policy-Adam: Sharpe difference -0.02 (95% CI -0.07 to +0.03); the interval includes zero.
- Policy-SAM-0.2 - Policy-Adam: Sharpe difference -0.02 (95% CI -0.07 to +0.03); the interval includes zero.
- The gradient-based allocators are trained on the in-sample CVaR; the baselines are not tuned on out-of-sample data and nothing here was selected on the out-of-sample result, but specification B itself was chosen after seeing A.
- One out-of-sample path, one market, 28 ETFs from 2005 (survivorship bias). Intervals are not multiple-testing corrected.
- Only the claims that the numbers above support may be quoted; if SAM shows no out-of-sample gain, that is the result.


## Interpretation (added by hand)

- SAM showed no out-of-sample benefit over Adam in either allocator: all four paired Sharpe differences are -0.02 with intervals that include zero, and the policy results are stable across three seeds.
- The "flatter" wording above for the policy network refers to ratios of 0.90 and 0.93 of a median lambda_max with no uncertainty interval; it should not be read as evidence that SAM finds flatter minima. For the direct allocator SAM ended at a sharper point (1.33x and 1.43x).
- The Hutchinson trace uses 30 probes and is noisy; the 5x drop in trace for Policy-SAM-0.2 with an almost unchanged lambda_max should not be interpreted.
- A possible explanation (not tested here): the direct problem has a single basin, so SAM has no flat-versus-sharp choice to make, and with equal gradient budget it takes half as many steps.
- The capped optimised portfolios have lower volatility and drawdown than equal weight, but no Sharpe improvement over it is demonstrated (all intervals include zero) and their excess return is lower.