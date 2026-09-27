# API reference

The public API of `multisport_sim.benchmark` (v0.3.0). Everything listed here is
exported from `multisport_sim.benchmark` unless a module path is given; anything
not listed is internal and may change without notice. Docstrings in the source
are the detailed reference; this page is the map.

## Tasks

| Task id | Config constant | Gymnasium id | Judge | Bank |
|---|---|---|---|---|
| `table-tennis-return-v0` | `TABLE_TENNIS_RETURN_V0` | `MultiSportRobot/TableTennisReturn-v0` | `TableTennisReturnJudge` | `table_tennis/return-v0` |
| `table-tennis-return-panda-v1` | `TABLE_TENNIS_RETURN_PANDA_V1` | `MultiSportRobot/TableTennisReturn-Panda-v1` (+ `-Vision-v1`) | `TableTennisReturnJudge` | `table_tennis/return-v1` |
| `table-tennis-return-g1-v1` | `TABLE_TENNIS_RETURN_G1_V1` | `MultiSportRobot/TableTennisReturn-G1-v1` (+ `-Vision-v1`) | `TableTennisReturnJudge` | `table_tennis/return-v1` |
| `table-tennis-return-g1-standing-v2` | `task_config.TABLE_TENNIS_RETURN_STANDING_G1_V2` | `MultiSportRobot/TableTennisReturn-G1-Standing-v2` (+ `-Vision-v2`) | `TableTennisReturnJudge` | `table_tennis/return-v1` |
| `tennis-return-v0` | `TENNIS_RETURN_V0` | `MultiSportRobot/TennisReturn-v0` | `TennisReturnJudge` | `tennis/return-v0` |
| `badminton-serve-v0` | `BADMINTON_SERVE_V0` | `MultiSportRobot/BadmintonServe-v0` | `BadmintonServeJudge` | `badminton/serve-v0` |
| `football-kick-v0` | `FOOTBALL_KICK_V0` | `MultiSportRobot/FootballKick-v0` | `FootballKickJudge` | `football/kick-v0` |
| `basketball-shoot-v0` | `BASKETBALL_SHOOT_V0` | `MultiSportRobot/BasketballShoot-v0` | `BasketballShootJudge` | `basketball/shoot-v0` |

The Panda and G1 tasks need the robot asset from MuJoCo Menagerie (see
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)); the others need nothing
beyond the base install.

### Registry

```python
from multisport_sim.benchmark import get_task, task_ids, iter_tasks, register_task, TaskEntry

task_ids()                         # every registered task id
entry = get_task("football-kick-v0")
entry.config                       # the shared ShotTaskConfig
entry.make_judge()                 # a fresh judge for one episode
entry.make_backend()               # the MuJoCo backend
```

`TaskEntry(config, judge_factory, backend_factory, env_entry_point,
vision_env_entry_point=None)` is how a new task is added; `register_task(entry)`
publishes it to the CLI, the runner and `register_envs()`.

### Task configuration

`ShotTaskConfig` is the backend-neutral definition every backend must agree on:
`task_id`, `env_id`, `sport`, `bank_resource`, `split`, `control_hz`,
`timeout_s`, `frame: TaskFrame`, `workspace`, `ball_limits`, `reward`. Methods:
`decimation(physics_dt)`, `max_physics_steps(physics_dt)`, `action_bounds()`,
`observation_bounds()`, `reward_for(result)`, `to_dict()`.

`TaskFrame(origin_xyz)` places the task frame in a simulator's world frame (a
translation; the axis convention is `CoordinateConvention`, fixed). Net-sport
tasks put the origin at the court centre, launch tasks at their goal.

## Environments

```python
import gymnasium as gym
from multisport_sim.benchmark import register_envs, TargetObservation

register_envs()                    # idempotent; importing the package already does it
env = TargetObservation(gym.make("MultiSportRobot/BadmintonServe-v0"))
observation, info = env.reset(seed=0)
info["target"]                     # {"center": [u, v], "radius_m": r} or None
```

Fixture tasks: action `[x, y, z, qw, qx, qy, qz]` (effector pose in the task
frame), observation 16 values (ball position, velocity, spin; effector pose).
Robot tasks: joint-space actions and a named observation layout
(`config.observation_layout()`); see [`POLICY_INTERFACE.md`](POLICY_INTERFACE.md).
`reset(options={"shot_id": ...})` plays one named shot. `info["episode_result"]`
on the last step is the judged `EpisodeResult` as a dict.

## Shot banks

```python
from multisport_sim.benchmark import ShotBank

bank = ShotBank.from_resource(split="test", task="badminton/serve-v0")
bank.digest, bank.manifest_digest, bank.manifest["levels"]
bank.filter(level="L3")            # tuple of ShotSpec
bank.get("badminton-serve-v0-test-l3-0001")
```

Banks are validated on load (strict JSON, sorted ids, digests, level ranges,
launch side, target plane, bucket coverage) and raise `ShotBankError`.
`ShotSpec` is one shot: `shot_id`, `sport`, `level`, `position`,
`linear_velocity`, `angular_velocity`, `tags`, `target: TargetSpec | None`.

## Judges

All judges implement the `ShotJudge` protocol: `reset(shot)`, `update(time_s=,
ball=, contacts=)` once per physics step, `abort(failure_reason=)` for adapter
failures, and `done` / `result`. Contacts are `SemanticContact`s in the shared
vocabulary (`ball`, `robot_racket`, `table`, `net`, `floor`, `post`,
`goal_net`, `rim`, `backboard`).

| Judge | Family | Success (`valid_return`) |
|---|---|---|
| `NetReturnJudge` (`TableTennisReturnJudge`, `TennisReturnJudge`) | return | struck on own half, crosses the net, first lands on the opponent's surface |
| `LaunchJudge` (base) | launch | defined by the sport subclass |
| `BadmintonServeJudge` | launch | struck below 1.15 m, crosses the net, lands in the diagonal singles service court |
| `FootballKickJudge` | launch | the whole ball crosses the goal line between the posts and under the bar |
| `BasketballShootJudge` | launch | the ball's centre comes down through the ring |

`EpisodeResult` fields and failure reasons are fixed by
[`schemas/episode-result.v0.json`](../src/multisport_sim/benchmark/schemas/episode-result.v0.json).

## Running and scoring

```python
from multisport_sim.benchmark import run_shots, RunConfig, assess_level, robustness_gap

output = run_shots(backend, controller, shots, config=RunConfig.from_task_config(task))
output.results                     # one EpisodeResult per shot, in input order
assess_level(output.results, bank.manifest, level="L2")   # the frozen pass criteria
robustness_gap(all_results)        # valid_return_rate on L2-L3 minus L4-L5
```

A controller is any object with `reset(shot, *, seed=None)` and
`act(observation) -> action | None`. `build_benchmark_report(...)` assembles the
full report; the CLI writes it:

```bash
multisport-benchmark --sport football --level L3 --split test --controller scripted \
    --report out.json --markdown out.md
multisport-benchmark --sport basketball --level L2 --learned-policy baselines/learned/basketball/seed0.zip
multisport-benchmark --robot panda --level L2 --policy my_pkg:load_policy
```

## Learned baselines

`multisport_sim.benchmark.learning`: `SwingPrimitive` (one straight-line swing
chosen from three parameters), `PrimitiveLaunchEnv` (a one-decision Gymnasium
environment over the real rollout), `LearnedLaunchController` and
`load_learned_controller(sport, path)`. Training:
`scripts/train_launch_policies.py` (needs the `train` extra).

## Schemas

`multisport_sim.benchmark.schemas`: `SCHEMAS` (artifact name to file),
`load(name)`, `validate(name, document)` (needs `jsonschema`). Every published
file format has a versioned schema whose `$id` never changes once released.

## Scripts

| Script | Purpose |
|---|---|
| `generate_shot_bank.py`, `generate_launch_banks.py` | build shot banks |
| `run_baselines.py`, `run_fixture_baselines.py`, `cross_task_report.py` | baseline tables and the cross-task summary |
| `train_launch_policies.py` | learned launch baselines |
| `package_submission.py`, `audit_submission.py` | result packages and their audit ([`LEADERBOARD.md`](LEADERBOARD.md)) |
| `backend_parity.py` | MuJoCo vs Isaac Lab trajectory comparison |
| `calibrate_reachability.py`, `calibrate_serve.py` | fixture and robot calibration |
