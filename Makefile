PYTHON ?= python
ISAAC_PYTHON ?= python

.PHONY: test lint verify test-isaac verify-all run run-isaac

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

verify: test
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus --headless --duration 0.1

test-isaac:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --headless --device cpu --scene campus --duration 0.1

verify-all: verify test-isaac

run:
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus

run-isaac:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --scene campus
