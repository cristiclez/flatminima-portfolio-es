.PHONY: install data data-synthetic risk backtest robustness portfolio riskfree experiments figures evidence test lint cov
PY ?= python

install:
	$(PY) -m pip install -e ".[dev,data,torch,plots]"

data:            ## Real data (Yahoo Finance); needs internet and the `data` extra
	$(PY) -m flatminima.data.pipeline --source yahoo

data-synthetic:  ## Fully offline synthetic panel (for demos and CI)
	$(PY) -m flatminima.data.pipeline --source synthetic

risk:            ## Rolling VaR/ES forecasts on the stored data (run `make data` first)
	$(PY) -m flatminima.risk.run

backtest:        ## Kupiec / Christoffersen / Acerbi-Szekely / Basel traffic light -> docs/backtest_results.md
	$(PY) -m flatminima.backtest.run

robustness:      ## Same with a 500-day window (out-of-sample period then covers 2008)
	$(PY) -m flatminima.risk.run --window 500 --table risk_forecasts_w500
	$(PY) -m flatminima.backtest.run --table risk_forecasts_w500 --report docs/backtest_w500.md

portfolio:       ## Walk-forward baselines (min-var with sample/LW/RMT, min-CVaR) -> docs/portfolio_baselines.md
	$(PY) -m flatminima.portfolio.run --cost-bps 5

riskfree:       ## 3-month T-bill (FRED DTB3) into the database; run after `make data`
	$(PY) -m flatminima.data.riskfree

experiments:    ## Spec B: Adam vs SAM allocators + Hessian analysis -> docs/portfolio_specB.md (needs torch)
	$(PY) -m flatminima.portfolio.experiment_b

figures:        ## PNG figures -> docs/figures (needs matplotlib)
	$(PY) -m flatminima.report.figures

evidence:       ## docs/CV_EVIDENCE.md, numbers recomputed from the database, with commit hash
	$(PY) -m flatminima.report.evidence

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing
