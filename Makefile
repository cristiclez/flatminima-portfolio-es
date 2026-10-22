.PHONY: install data data-synthetic risk backtest robustness portfolio test lint cov
PY ?= python

install:
	$(PY) -m pip install -e ".[dev,data]"

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

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing
