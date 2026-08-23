# Robot Sports Gym

[简体中文](README.md) | English

Robot Sports Gym (RSG) is an in-development, cross-embodiment platform for training and evaluating robot perception, planning, control, robustness, and sim-to-real performance across tennis, table tennis, football, badminton, basketball, and squash under shared tasks, physics specifications, and metrics. The current repository provides the asset-free **MuJoCo + Isaac Sim/PhysX** physics foundation with regulation-scale scenes and sport-specific dynamics.

> **Status: Alpha.** The scenes, ball dynamics, rebound-fidelity reports, an experimental table-tennis Shot Skill harness, and a MuJoCo Gymnasium fixture are operational. Real robot adapters, Isaac Lab RL environments, the remaining canonical tasks, and reference policies are planned. Do not describe the current release as a completed robot ball-sports benchmark.

## Documentation

- [Draft robot benchmark specification](docs/BENCHMARK_SPEC.md)
- [Table-tennis Shot Skill harness](docs/TABLE_TENNIS_SHOT_SKILL.md)
- [Roadmap](docs/ROADMAP.md)
- [Reproducibility requirements](docs/REPRODUCIBILITY.md)
- [Physics and fidelity evaluation](docs/PHYSICS.md)
- [Isaac Sim backend](docs/ISAAC_SIM.md)
- [Contributing](CONTRIBUTING.md) · [Governance](GOVERNANCE.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

## Implemented scenes

| Sport | Scene and equipment | Physics |
|---|---|---|
| Tennis | Regulation court, lines, net, two rackets, ball | ITF-scale rebound, drag, spin/Magnus force |
| Table tennis | 2.74 × 1.525 × 0.76 m table, mesh net, two paddles, ball | Table-specific contact, drag and spin |
| Football | 105 × 68 m pitch, goals and ball | Grass friction/rolling resistance, drag and curve |
| Badminton | Singles/doubles court, net, two rackets, shuttle | Orientation-dependent drag and stabilizing torque |
| Basketball | 28 × 15 m court, two framed backboards, rims/nets, ball | Hardwood contact, FIBA-scale rebound and spin |
| Squash | 9.75 × 6.40 m enclosed singles court, four walls, service boxes, ball | Floor/wall contact, drag and spin |

The `campus` scene loads all sports at once. Each sport also has a standalone close-up scene. Shared dimensions, mass, and aerodynamic parameters are defined in `src/multisport_sim/specs.py`.

## Backend render gallery

These images are rendered directly from the current repository code using each standalone scene's default camera; they are not concept art. Click an image to view it at its original resolution on GitHub.

| Sport | MuJoCo | Isaac Sim / PhysX |
|:---:|:---:|:---:|
| Tennis | [![MuJoCo tennis scene](docs/images/rendered/mujoco/tennis.png)](docs/images/rendered/mujoco/tennis.png) | [![Isaac Sim tennis scene](docs/images/rendered/isaac/tennis.png)](docs/images/rendered/isaac/tennis.png) |
| Table tennis | [![MuJoCo table tennis scene](docs/images/rendered/mujoco/table_tennis.png)](docs/images/rendered/mujoco/table_tennis.png) | [![Isaac Sim table tennis scene](docs/images/rendered/isaac/table_tennis.png)](docs/images/rendered/isaac/table_tennis.png) |
| Football | [![MuJoCo football scene](docs/images/rendered/mujoco/football.png)](docs/images/rendered/mujoco/football.png) | [![Isaac Sim football scene](docs/images/rendered/isaac/football.png)](docs/images/rendered/isaac/football.png) |
| Badminton | [![MuJoCo badminton scene](docs/images/rendered/mujoco/badminton.png)](docs/images/rendered/mujoco/badminton.png) | [![Isaac Sim badminton scene](docs/images/rendered/isaac/badminton.png)](docs/images/rendered/isaac/badminton.png) |
| Basketball | [![MuJoCo basketball scene](docs/images/rendered/mujoco/basketball.png)](docs/images/rendered/mujoco/basketball.png) | [![Isaac Sim basketball scene](docs/images/rendered/isaac/basketball.png)](docs/images/rendered/isaac/basketball.png) |
| Squash | [![MuJoCo squash scene](docs/images/rendered/mujoco/squash.png)](docs/images/rendered/mujoco/squash.png) | [![Isaac Sim squash scene](docs/images/rendered/isaac/squash.png)](docs/images/rendered/isaac/squash.png) |

## MuJoCo quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"

multisport-sim --scene campus
multisport-sim --scene badminton --headless --duration 3 --wind 2 0 0
multisport-sim --scene squash
```

In the native viewer, press `Space` to relaunch and `R` to reset.

## Isaac Sim quick start

Use a Python environment containing a compatible Isaac Sim and Isaac Lab installation:

```bash
/path/to/isaac/python -m pip install -e . --no-deps

multisport-isaac --scene campus
multisport-isaac --headless --device cpu --scene tennis --duration 0.1
multisport-isaac --headless --device cpu --scene squash --duration 0.1

# Complete rally: A serves, B returns, second bounce awards B the point (0:0 -> 0:1)
multisport-isaac --scene squash --squash-demo --duration 9 \
  --demo-report reports/isaac/squash-demo.json

# Rebuild both the JSON report and GIF from the same real PhysX rally
make demo-squash ISAAC_PYTHON=/path/to/isaac/python

# Verify the existing JSON/GIF evidence without launching Isaac
make verify-squash-demo PYTHON=.venv/bin/python

multisport-isaac --headless --device cpu --scene tennis \
  --evaluate --report reports/isaac/tennis.json
```

PhysX produces the front-wall contacts and two floor bounces; the demo script only applies
the serve and return racket impulses. The JSON report includes the complete event timeline and
sets `complete: true` only after the observed second bounce awards the point. The GIF generator
also verifies the fixed event order, final 0:1 score, frame count, and frame-to-frame changes,
and refuses to overwrite the asset if validation fails.
The standalone verifier needs neither Isaac nor Pillow and can audit the event timeline and GIF
container metadata in a regular Python environment or CI. `make demo-squash` writes both committed
evidence files under `docs/images/shot-skill/`.

![Isaac Sim complete squash scoring rally](docs/images/shot-skill/squash-serve-score-demo.gif)

The validated local combination is Isaac Sim 5.0.0, Isaac Lab 0.46.2, Python 3.11, and CPU PhysX. See [docs/ISAAC_SIM.md](docs/ISAAC_SIM.md) for lifecycle and headless-exit details.

## Table-tennis Shot Skill

The experimental `table-tennis-return-v0` task provides fixed Shot Banks, MuJoCo ball launch, true racket-contact events, legal-return and placement judging, bucketed metrics, and JSON/Markdown reports. Its scripted mocap paddle is a test fixture, not an eligible robot policy.

```bash
multisport-benchmark --level L1 --split dev --controller scripted \
  --report reports/table-tennis-l1.json --markdown reports/table-tennis-l1.md

multisport-benchmark --level L1 --split dev --controller noop
```

See the [Shot Skill documentation](docs/TABLE_TENNIS_SHOT_SKILL.md) for the coordinate frame, L0–L5 criteria, report schema, fixed-bank integrity checks, and robot-adapter contract.

## Quantitative fidelity

The `multisport-fidelity-v1` suite runs real drop simulations and records measured rebound, reference intervals, absolute/relative error, tolerance utilization, effective restitution, pass/fail, and a continuous score.

```bash
make evaluate
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=/path/to/python
```

Committed baselines are available for [MuJoCo](reports/mujoco-fidelity.md) and [Isaac Sim](reports/isaac-fidelity.md). These scores cover first-rebound contact dynamics only; they are not a claim of total simulator realism.

## Toward a robot benchmark

The experimental table-tennis Shot Skill validates fixed launches, true-contact events, judging, and report schemas with a MuJoCo fixture; it is not yet a robot environment or submission track. The proposed first public release contains five single-episode tasks: tennis return, table-tennis return, football kick-to-target, badminton serve, and basketball shooting. The draft protocol defines state, vision, robustness, and sim-to-real tracks; raw metrics; seeding; robot adapters; and reproducible submission artifacts.

The repository remains a physics foundation until at least one versioned Gymnasium environment, one vectorized Isaac Lab environment, licensed robot assets, fixed evaluation splits, and reproducible reference policies are available.

## Development

```bash
make lint
make test
make verify
make benchmark
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Physics changes require primary references, numerical regression tests, and refreshed fidelity reports.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The project is licensed under the [MIT License](LICENSE); third-party robot assets, datasets, and weights must carry their own compatible licenses and provenance.
