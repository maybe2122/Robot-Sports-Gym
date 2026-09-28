# Robot Sports Gym

[简体中文](README.md) | English

Robot Sports Gym (RSG) is an in-development, cross-embodiment platform for training and evaluating robot perception, planning, control, robustness, and sim-to-real performance across tennis, table tennis, football, badminton, basketball, and squash under shared tasks, physics specifications, and metrics. The current repository provides the asset-free **MuJoCo + Isaac Sim/PhysX** physics foundation with regulation-scale scenes and sport-specific dynamics.

> **Status: Alpha (v0.3.0).** All five canonical single-shot tasks run in MuJoCo with fixed shot banks (100 test episodes per level), versioned Gymnasium environments, frozen L0–L5 criteria, reference baselines and learned PPO baselines over 5 seeds for every task. Table tennis additionally runs on an actuated Franka Panda and a Unitree G1 with state and vision tracks, and its Isaac Lab environment has been run and compared episode by episode with MuJoCo. Every shot bank is still `experimental` and no leaderboard is open: this is a benchmark candidate, not a certified benchmark. What is missing is listed in [docs/TODO.md](docs/TODO.md).

## What v0.3.0 delivers

| Sport | Task | Performed by | Rules | Docs |
|---|---|---|---|---|
| Table tennis | `table-tennis-return-*` | mocap fixture, **Franka Panda**, **Unitree G1** (fixed base / free standing); state and vision tracks | ITTF legal return | [table tennis](docs/TABLE_TENNIS_SHOT_SKILL.md) |
| Tennis | `tennis-return-v0` | mocap racket fixture | ITF singles court | [tennis](docs/TENNIS.md) |
| Badminton | `badminton-serve-v0` | mocap racket fixture | BWF serve (1.15 m, diagonal court) | [badminton](docs/BADMINTON.md) |
| Football | `football-kick-v0` | mocap boot fixture | IFAB: whole ball over the line | [launch tasks](docs/LAUNCH_TASKS.md) |
| Basketball | `basketball-shoot-v0` | mocap launcher fixture | FIBA: down through the ring | [launch tasks](docs/LAUNCH_TASKS.md) |

- **Every task**: a versioned Gymnasium environment (11 in total, all pass `check_env`), a simulation-verified shot bank (train 200 / dev 50 / test 100 per level), frozen L0–L5 criteria, and reports with raw episodes, Wilson intervals and buckets.
- **Baselines**: floors and reference controllers for every task, plus learned PPO baselines (5 seeds × 8000 episodes) with untrained- and random-primitive controls: table tennis L2 0% → **98%**, basketball 2% → **36%**, badminton placement 19% → **36%**; tennis only matches the prior and football placement gets *worse* (49% → 9%) — reported as is. One summary: [`reports/cross-task-summary.md`](reports/cross-task-summary.md).
- **Two backends**: the Isaac Lab table-tennis environment runs on CPU and GPU PhysX and matches MuJoCo shot by shot (flight 3.5 mm, verdicts 98.7%). The same Panda runs on both: kinematics agree to 0.5 µm, interception transfers (90% shot-by-shot agreement), returning does not (26% vs 0%) — a contact-model gap that needs measured data ([report](reports/panda-backend-parity.md)).
- **Release engineering**: versioned JSON Schemas, an auditable result-package format ([`docs/LEADERBOARD.md`](docs/LEADERBOARD.md)), [API reference](docs/API.md), [license manifest](THIRD_PARTY_LICENSES.md), `uv.lock` verified on Python 3.10 and 3.13, green CI.
- **Not yet**: robot versions of tennis/badminton/football/basketball (measured: neither the Panda nor the wrist-paddle G1 can swing fast enough to serve), real-world calibration data (M4), an open leaderboard, external reproduction (M6). See [`docs/TODO.md`](docs/TODO.md).

## Humanoid vision-driven table tennis

A free-pelvis Unitree G1 can now return simulated balls using depth only, stereo RGB, or aligned RGB-D.
An ankle feedback controller maintains standing while the waist and racket arm follow visual estimates.
This is a standing, single-return baseline; walking, continuous rallies, and hardware deployment are not included.

```bash
MUJOCO_GL=glfw .venv/bin/python -m multisport_sim.benchmark_cli \
  --config configs/humanoid-rgbd.json --viewer
```

See [setup, camera configuration, policy interfaces, and limitations](docs/HUMANOID_PLAY.md).

## Documentation

- [API reference](docs/API.md) · [Policy interface](docs/POLICY_INTERFACE.md) · [License manifest](THIRD_PARTY_LICENSES.md)
- Tasks: [table tennis](docs/TABLE_TENNIS_SHOT_SKILL.md) · [tennis](docs/TENNIS.md) · [badminton serve](docs/BADMINTON.md) · [football kick & basketball shoot](docs/LAUNCH_TASKS.md)
- [Cross-task baseline summary](reports/cross-task-summary.md) · [Submission packages](docs/SUBMISSION.md) · [Leaderboard audit process](docs/LEADERBOARD.md)
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
uv sync --extra test          # reproduces the locked development environment (uv.lock)
source .venv/bin/activate     # or: uv venv && uv pip install -e ".[test]"

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

### Embodied task `table-tennis-return-panda-v1`

A versioned task that runs *alongside* `table-tennis-return-v0`, not after it: same judge, rules and reward terms, while an actuated 7-DoF Franka Panda hits the ball instead of a teleported mocap blade. The Panda task defaults to the statistically sufficient `table_tennis/return-v1` bank; the fixture can replay that bank for controlled comparisons while retaining v0 for frozen rule-engine regression. The action is seven joint-position setpoints in radians (the arm's own limits, not normalized), the observation is 33 numbers, safety breaches are checked every physics step and terminate the episode into the denominator, and actuator mechanical work is integrated as `energy_joule`.

The arm asset is not vendored. It is resolved from `MULTISPORT_MENAGERIE_PATH`, then `MUJOCO_MENAGERIE_PATH`, then `~/mujoco_menagerie`:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
git -C ~/mujoco_menagerie sparse-checkout add franka_emika_panda
```

```bash
# Reference baselines: hold / random / intercept.  None is a submission.
multisport-benchmark --robot panda --controller intercept --level L2 --split dev

# The full table: four baselines x two splits x L0-L5
make baselines PYTHON=/path/to/python
```

To score **your own trained policy** -- any callable `obs(33) -> action(7)`, given as `module:policy` or as a zero-argument factory returning one:

```bash
python scripts/eval_policy.py --policy my_pkg.eval:load_policy --policy-id my-sac-v3 \
  --split test --levels all --out reports/my-sac-v3
```

Each level produces a full JSON/Markdown report plus one cross-difficulty summary table. The observation layout is defined once, in `multisport_sim.benchmark.envs.panda_observation_vector`, so the Gymnasium environment `MultiSportRobot/TableTennisReturn-Panda-v1` and the offline scoring path read the same numbers in the same order.

### Vision track

A second observation track on the same task: **the policy sees only cameras**. The robot, the action, the judge, the shot bank, the reward and the thresholds are all unchanged; the one difference is that the observation object has **no `ball` field**. That rule is enforced by construction, not promised in a docstring.

The declared suite is two 320x240 / 120 Hz cameras on a 4.2 m stereo baseline either side of the table, plus blade IMU, contact, joint torque and frame transform. Camera poses come from the `CameraSpec` and are written into the model, so the pose a report names is the pose that was rendered.

```bash
# The built-in vision baseline: the identical swing, driven by triangulation instead of truth
multisport-benchmark --robot panda --track vision --level L1 --split dev

# Score your own vision policy
python scripts/eval_policy.py --policy my_pkg.eval:load_policy --policy-id my-vision \
  --track vision --split test --levels all --out reports/my-vision
```

The policy receives a `VisionObservation`: `obs.sensors.camera("ball_camera_left").rgb` is a `(240, 320, 3)` uint8 image and `obs.robot.joint_positions` is proprioception. Train against `MultiSportRobot/TableTennisReturn-Panda-Vision-v1`, whose dictionary observation includes `frame_age_s` -- the cameras run at 120 Hz under a 200 Hz control loop, so every other step holds the previous frame. Training and scoring call the same packing function.

The reference pipeline (colour segmentation, largest blob, two-ray triangulation, least-squares velocity) localizes the ball to a **median 0.81 cm (p90 1.02 cm) with 8.1% of frames undetected**, measured over all 540 stereo frames of the dev split. Running the identical swing from it scores 100% hit rate at L1 -- level with the state track -- and 50% at L2. **That gap is the cost of perception, measured rather than assumed.**

See [`docs/VISION_TRACK.md`](docs/VISION_TRACK.md).

### Full metrics and result packages

Reports now carry all eight raw metrics from `BENCHMARK_SPEC` section 7. The three that were missing are `contact_error` (how far off the blade's centre the strike landed, plus blade speed at contact), `robustness_gap` (`valid_return_rate` on L2-L3 minus L4-L5; L1 ends at the strike by construction and is reported per level only), and `inference_latency_ms` (the policy's own `act` call, nothing else).

```bash
# A reproducible package: manifest / config / metrics / policy / videos / environment
make submission POLICY=my_pkg:load_policy POLICY_ID=my-sac-v3
```

The manifest **admits when the working tree is dirty**, videos are the first N successes *and* the first N failures by shot id (cherry-picking is not available), weights are content-hashed with sha256, and a package shipped without weights says so instead of looking complete. See [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

### Tennis, and shot banks large enough to conclude something

A second sport, `tennis-return-v0`, now runs on **the same judge, the same L0-L5 thresholds and the same report schema**. Only geometry and scale differ: a 23.77 x 8.23 m singles court where the ground itself is the landing surface, a three-second episode, and a ball arriving at 17-34 m/s.

```bash
multisport-benchmark --sport tennis --level L2 --split test --controller scripted
make tennis-baselines PYTHON=/path/to/python
```

The fixed sets are large enough now. `scripts/generate_shot_bank.py` produces `table_tennis/return-v1` and `tennis/return-v0`, each **train 1200 / dev 300 / test 600 -- 100 episodes per level** -- with non-overlapping seeds *and* provably disjoint shots. Every shot was launched in the real scene and kept only when the judge called it a legal incoming ball; `short`, `deep` and `edge` come from the **measured** first bounce; and every bucket an L4/L5 pass criterion names is filled by stratified sampling rather than by luck.

```bash
# Score on the statistically sufficient bank
multisport-benchmark --robot panda --level L2 --split test
```

L5 of `return-v1` also declares real perturbations -- observation noise, observation and action latency, domain randomization -- all stated in the manifest, all drawn from the episode seed so a failure replays exactly, and all applied **only to what the policy sees, never to what the judge sees**. **`return-v0` is unchanged to the byte**, so every score ever produced on it remains valid.

See [`docs/TENNIS.md`](docs/TENNIS.md) and [`docs/SHOT_BANKS.md`](docs/SHOT_BANKS.md).

Baseline scores are in [`reports/table-tennis-panda-baselines.md`](reports/table-tennis-panda-baselines.md); the calibration, reachability findings, and safety envelope are in [`docs/ROBOT_LAYER.md`](docs/ROBOT_LAYER.md). The Panda table uses `table_tennis/return-v1`: 50 dev and 100 test shots per level, with per-bucket Wilson intervals. The smallest L4/L5 bucket still has only 14 shots, so a worst-bucket point estimate must not be cited without its interval.

## The five canonical tasks

| Task | Gymnasium id | Family | Success | Baseline docs |
|---|---|---|---|---|
| `table-tennis-return-v0` / `-panda-v1` / `-g1-v1` | `MultiSportRobot/TableTennisReturn-*` | return | legal return onto the opponent's half | [table tennis](docs/TABLE_TENNIS_SHOT_SKILL.md) |
| `tennis-return-v0` | `MultiSportRobot/TennisReturn-v0` | return | legal return into the singles court | [tennis](docs/TENNIS.md) |
| `badminton-serve-v0` | `MultiSportRobot/BadmintonServe-v0` | launch | BWF-legal serve into the diagonal service court | [badminton](docs/BADMINTON.md) |
| `football-kick-v0` | `MultiSportRobot/FootballKick-v0` | launch | the whole ball over the goal line (IFAB) | [launch tasks](docs/LAUNCH_TASKS.md) |
| `basketball-shoot-v0` | `MultiSportRobot/BasketballShoot-v0` | launch | down through the ring (FIBA) | [launch tasks](docs/LAUNCH_TASKS.md) |

```bash
multisport-benchmark --sport badminton --level L2 --split test --controller scripted
multisport-benchmark --sport football  --level L3 --split test --controller scripted
multisport-benchmark --sport basketball --level L2 --split test \
    --learned-policy baselines/learned/basketball/seed0.zip
make cross-task        # one table: every task x level x controller
```

Launch tasks start with the object on the robot's side (released, resting or rolling) and put the task frame's origin at the goal, so the robot is always at x < 0 launching toward +x. L3 placement targets are task information and reach the policy through `info["target"]`, the `TargetObservation` wrapper, and `reset(target=...)`.

**Learned baselines.** `scripts/train_launch_policies.py` trains a PPO policy (Stable-Baselines3) over a single swing primitive — face speed, elevation and yaw offset — from observable geometry only, 5 seeds per task (all five sports) on the train split, scored on the test split through the same report path as every other baseline, next to two controls: the untrained primitive (every parameter mid-range) and a random primitive. Learning clearly helps table tennis, basketball and badminton placement, only matches the prior on tennis, and hurts football placement — see [docs/LEARNED_BASELINES.md](docs/LEARNED_BASELINES.md). Weights, training curves, wall time and hardware are committed under `baselines/learned/`; results in `reports/learned-*-baselines.md`.

**Backend parity.** The Isaac Lab table-tennis environment was run on Isaac Sim 5.0 / Isaac Lab 0.46.2 (CPU and GPU PhysX) and compared shot by shot with MuJoCo on the 300-shot dev split: ball flight agrees to 3.5 mm (median), judge verdicts to 99%, while the post-bounce apex differs by ~50 mm (contact models; M4 calibration). The run found and fixed a doubled aerodynamic force in the Isaac environment. See [the parity report](reports/table-tennis-backend-parity.md).

**Auditable results.** `scripts/audit_submission.py` recomputes every verdict and metric of a result package from its raw episodes, checks that each level contains exactly the bank's shots, verifies bank digests and weight hashes, and can replay the package. See [docs/LEADERBOARD.md](docs/LEADERBOARD.md). Every published file format has a versioned JSON Schema in `multisport_sim.benchmark.schemas`.

## Quantitative fidelity

The `multisport-fidelity-v1` suite runs real drop simulations and records measured rebound, reference intervals, absolute/relative error, tolerance utilization, effective restitution, pass/fail, and a continuous score.

```bash
make evaluate
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=/path/to/python
```

Committed baselines are available for [MuJoCo](reports/mujoco-fidelity.md) and [Isaac Sim](reports/isaac-fidelity.md). These scores cover first-rebound contact dynamics only; they are not a claim of total simulator realism.

## What this release is not

- **Not a leaderboard.** Every shot bank is `experimental`; the audit tooling exists, the table does not.
- **Not embodied for every sport.** Tennis, badminton, football and basketball use mocap fixtures; only table tennis has robot arms/humanoids. The biped asset football needs is not in (M2).
- **Not validated for robots on Isaac.** The Isaac environment covers the table-tennis fixture (CPU and GPU PhysX); the Panda/G1 adapters and the other sports have no Isaac environment yet.
- **Not calibrated against the real world.** Fidelity covers first-rebound tests; there is no measured trajectory, impact or robot data yet (M4), and no external reproduction (M6).

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
