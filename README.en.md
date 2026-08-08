# MultiSport Physics Sim

[简体中文](README.md) | English

MultiSport Physics Sim is a standalone, asset-free collection of programmatic sports scenes for **MuJoCo** and **Isaac Sim/PhysX**. It currently includes regulation-scale tennis, table tennis, football, badminton, and basketball environments with sport-specific rigid-contact and aerodynamic models.

> **Status: Alpha / physics foundation.** The scenes, ball dynamics, and rebound-fidelity reports are operational. Robots, Gymnasium/Isaac Lab RL environments, canonical tasks, and reference policies are planned. Do not describe the current release as a completed robot ball-sports benchmark.

## Documentation

- [Draft robot benchmark specification](docs/BENCHMARK_SPEC.md)
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

The `campus` scene loads all sports at once. Each sport also has a standalone close-up scene. Shared dimensions, mass, and aerodynamic parameters are defined in `src/multisport_sim/specs.py`.

## MuJoCo quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"

multisport-sim --scene campus
multisport-sim --scene badminton --headless --duration 3 --wind 2 0 0
```

In the native viewer, press `Space` to relaunch and `R` to reset.

## Isaac Sim quick start

Use a Python environment containing a compatible Isaac Sim and Isaac Lab installation:

```bash
/path/to/isaac/python -m pip install -e . --no-deps

multisport-isaac --scene campus
multisport-isaac --headless --device cpu --scene tennis --duration 0.1
multisport-isaac --headless --device cpu --scene tennis \
  --evaluate --report reports/isaac/tennis.json
```

The validated local combination is Isaac Sim 5.0.0, Isaac Lab 0.46.2, Python 3.11, and CPU PhysX. See [docs/ISAAC_SIM.md](docs/ISAAC_SIM.md) for lifecycle and headless-exit details.

## Quantitative fidelity

The `multisport-fidelity-v1` suite runs real drop simulations and records measured rebound, reference intervals, absolute/relative error, tolerance utilization, effective restitution, pass/fail, and a continuous score.

```bash
make evaluate
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=/path/to/python
```

Committed baselines are available for [MuJoCo](reports/mujoco-fidelity.md) and [Isaac Sim](reports/isaac-fidelity.md). These scores cover first-rebound contact dynamics only; they are not a claim of total simulator realism.

## Toward a robot benchmark

The proposed first release contains five single-episode tasks: tennis return, table-tennis return, football kick-to-target, badminton serve, and basketball shooting. The draft protocol defines state, vision, robustness, and sim-to-real tracks; raw metrics; seeding; robot adapters; and reproducible submission artifacts.

The repository remains a physics foundation until at least one versioned Gymnasium environment, one vectorized Isaac Lab environment, licensed robot assets, fixed evaluation splits, and reproducible reference policies are available.

## Development

```bash
make lint
make test
make verify
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Physics changes require primary references, numerical regression tests, and refreshed fidelity reports.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The project is licensed under the [MIT License](LICENSE); third-party robot assets, datasets, and weights must carry their own compatible licenses and provenance.

