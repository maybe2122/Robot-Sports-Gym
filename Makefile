PYTHON ?= python
ISAAC_PYTHON ?= python
POLICY ?= multisport_sim.benchmark.robot_controllers:ScriptedInterceptController
POLICY_ID ?= $(POLICY)
SPLIT ?= test
TRACK ?= state
SUBMISSION_OUT ?= submission
SPORT ?= table_tennis

.PHONY: test lint verify check-reports test-isaac verify-all evaluate evaluate-isaac benchmark \
	benchmark-robot benchmark-g1 baselines baselines-g1 tennis tennis-baselines \
	shot-bank calibrate submission \
	demo-squash verify-squash-demo \
	watch-squash-demo run run-isaac

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

check-reports:
	PYTHONPATH=src $(PYTHON) scripts/check_baseline_reports.py

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
	for sport in tennis table_tennis football badminton basketball squash; do \
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

# One embodied run.  The scripted intercept baseline is not a submission; it
# bounds what a fixed-base Panda can do on this bank.
benchmark-robot:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.benchmark_cli \
		--robot panda --controller intercept --level L2 --split dev \
		--report reports/table-tennis-panda-l2.json \
		--markdown reports/table-tennis-panda-l2.md

# Every reference baseline over every level of both splits.  Needs the Panda
# asset; see docs/ROBOT_LAYER.md for MULTISPORT_MENAGERIE_PATH.
baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --out reports

# The same sweep on the second embodiment.  Separate target, separate report:
# the two robots share a task family, not a results table row.
baselines-g1:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --robot g1 --out reports

# One embodied run on the Unitree G1.
benchmark-g1:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.benchmark_cli \
		--robot g1 --controller intercept --level L1 --split dev \
		--report reports/table-tennis-g1-l1.json \
		--markdown reports/table-tennis-g1-l1.md

# A reproducible result package for one policy (BENCHMARK_SPEC section 9).
# POLICY names 'module:attribute'; videos need the optional pillow extra.
submission:
	PYTHONPATH=src $(PYTHON) scripts/package_submission.py \
		--policy $(POLICY) --policy-id $(POLICY_ID) \
		--split $(SPLIT) --track $(TRACK) --out $(SUBMISSION_OUT)

# One tennis run with the mocap fixture.  Tennis has no embodied task yet.
tennis:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.benchmark_cli \
		--sport tennis --level L2 --split test --controller scripted \
		--report reports/tennis-l2.json --markdown reports/tennis-l2.md

# Both tennis baselines over every level of the test split.
tennis-baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_tennis_baselines.py --out reports

# Regenerate a statistically sufficient shot bank.  This REPLACES the bank's
# files and digests, so never run it against a bank that has published scores.
shot-bank:
	PYTHONPATH=src $(PYTHON) scripts/generate_shot_bank.py --sport $(SPORT)

# Re-derive the robot-side constants (strike plane, base pose, ready qpos).
calibrate:
	PYTHONPATH=src $(PYTHON) scripts/calibrate_reachability.py \
		--report reports/table-tennis-panda-reachability.json

demo-squash:
	mkdir -p reports/isaac
	PYTHONPATH=src $(ISAAC_PYTHON) scripts/capture_squash_demo.py \
		--headless --device cpu --duration 9 \
		--output docs/images/shot-skill/squash-serve-score-demo.gif \
		--report docs/images/shot-skill/squash-serve-score-demo.json
	$(MAKE) verify-squash-demo PYTHON=$(ISAAC_PYTHON)

verify-squash-demo:
	PYTHONPATH=src $(PYTHON) -m multisport_sim.squash_demo \
		--gif docs/images/shot-skill/squash-serve-score-demo.gif \
		--report docs/images/shot-skill/squash-serve-score-demo.json

watch-squash-demo:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli \
		--scene squash --squash-demo --duration 9 \
		--demo-report reports/isaac/squash-demo.json

run:
	PYTHONPATH=src $(PYTHON) -m multisport_sim --scene campus

run-isaac:
	PYTHONPATH=src $(ISAAC_PYTHON) -m multisport_sim.isaac_cli --scene campus
