PYTHON ?= python
ISAAC_PYTHON ?= python

.PHONY: test lint verify test-isaac verify-all evaluate evaluate-isaac benchmark run run-isaac

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

verify: test
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus --headless --duration 0.1

test-isaac:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --headless --device cpu --scene campus --duration 0.1

verify-all: verify test-isaac

evaluate:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.evaluation_cli \
		--output reports/mujoco-fidelity.json --markdown reports/mujoco-fidelity.md

evaluate-isaac:
	mkdir -p reports/isaac
	for sport in tennis table_tennis football badminton basketball; do \
		PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --headless --device cpu \
			--scene $$sport --evaluate --report reports/isaac/$$sport.json; \
	done
	PYTHONPATH=src $(PYTHON) -m multisport_sim.evaluation_cli \
		--aggregate reports/isaac/*.json --output reports/isaac-fidelity.json \
		--markdown reports/isaac-fidelity.md

benchmark:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.benchmark_cli \
		--level L1 --split dev --controller scripted \
		--report reports/table-tennis-l1.json \
		--markdown reports/table-tennis-l1.md

run:
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus

run-isaac:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --scene campus
