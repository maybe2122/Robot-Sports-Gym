# License manifest

Every asset, dataset and set of weights that Robot Sports Gym ships or loads, with
its license and where it comes from. The project itself is MIT-licensed
([`LICENSE`](LICENSE)); nothing below changes that, but each entry tells you what
you may do with that particular piece.

## 1. Robot assets (loaded, not vendored)

The repository contains **no robot model files**. They are loaded at run time from
a local checkout of MuJoCo Menagerie (`MULTISPORT_MENAGERIE_PATH`,
`MUJOCO_MENAGERIE_PATH` or `~/mujoco_menagerie`), and every report records the
asset's upstream URL, license and the list of in-memory modifications
(`multisport_sim.benchmark.assets.license_manifest()`).

| Asset | Upstream | License | Used by | Local modifications (in memory only) |
|---|---|---|---|---|
| Franka Emika Panda | [mujoco_menagerie/franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) | Apache-2.0 | `table-tennis-return-panda-v1` | attached with prefix `rb_`, paddle added, pedestal mount |
| Unitree G1 | [mujoco_menagerie/unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) | BSD-3-Clause | `table-tennis-return-g1-v1`, `-standing-v2` | attached with prefix `g1_`, paddle added, support platform, free pelvis retained (standing task) |

## 2. Scenes and fixtures (generated, in this repository)

All courts, tables, nets, goals, baskets, balls and benchmark fixtures are
generated procedurally by `multisport_sim.scene` and `multisport_sim.isaac_scene`
from regulation dimensions. They are part of this repository: **MIT**.

## 3. Datasets (generated, in this repository)

| Dataset | Path | How it was made | License |
|---|---|---|---|
| Shot banks (`table_tennis/return-v0`, `-v1`, `tennis/return-v0`, `badminton/serve-v0`, `football/kick-v0`, `basketball/shoot-v0`) | `src/multisport_sim/benchmark/data/` | Sampled with documented seeds and verified in MuJoCo by `scripts/generate_shot_bank.py` / `scripts/generate_launch_banks.py` (`return-v0` hand-written) | MIT |
| Baseline tables and reports | `reports/` | Produced by the scripts named in each file | MIT |
| Rendered images and demo GIFs | `docs/images/` | Rendered from this repository's scenes | MIT |

No real-world measurement data is included yet (M4).

## 4. Weights (trained, in this repository)

| Weights | Path | How they were made | License |
|---|---|---|---|
| Learned launch baselines (PPO, 5 seeds per task) | `baselines/learned/<sport>/seed<N>.zip` | `scripts/train_launch_policies.py`; each `seed<N>.json` records hyperparameters, wall time and hardware | MIT |

## 5. Software dependencies (installed, not bundled)

| Package | License | Needed for |
|---|---|---|
| MuJoCo | Apache-2.0 | everything (`dependencies`) |
| Gymnasium | MIT | environments (`dependencies`) |
| NumPy | BSD-3-Clause | everything (`dependencies`) |
| pytest, ruff, jsonschema | MIT / MIT / MIT | `test` extra |
| Stable-Baselines3 | MIT | `train` extra |
| PyTorch | BSD-3-Clause | `train` extra |
| Pillow | MIT-CMU (HPND) | `submission` extra (video export) |
| NVIDIA Isaac Sim | NVIDIA Isaac Sim additional software and materials license (proprietary) | optional Isaac backend; **not** a dependency, never redistributed |
| Isaac Lab | BSD-3-Clause | optional Isaac backend; not a dependency |

The exact versions used for any result are written to its package's
`environment.txt` (`scripts/package_submission.py`) and pinned for development in
[`uv.lock`](uv.lock).
