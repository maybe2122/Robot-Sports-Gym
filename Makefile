PYTHON ?= python

.PHONY: test lint verify run

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

verify: test
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus --headless --duration 0.1

run:
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus

