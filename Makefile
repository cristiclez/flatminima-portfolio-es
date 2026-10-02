.PHONY: install data data-synthetic test lint cov
PY ?= python

install:
	$(PY) -m pip install -e ".[dev,data]"

data:            ## Real data (Yahoo Finance); needs internet and the `data` extra
	$(PY) -m flatminima.data.pipeline --source yahoo

data-synthetic:  ## Fully offline synthetic panel (for demos and CI)
	$(PY) -m flatminima.data.pipeline --source synthetic

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing
