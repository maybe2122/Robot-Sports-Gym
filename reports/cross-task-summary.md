# Cross-task baseline summary

Test split, primary metric per level (bold = meets the frozen threshold). Regenerate every input with `make baselines-all`; this page with `python scripts/cross_task_report.py`.

**No row is a submission.** Every controller here is a reference: `hold`/`noop` and `random` are floors, the scripted ones read privileged state.

| Task | Embodiment | Controller | L0 | L1 | L2 | L3 | L4 | L5 | Passed | Gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Table tennis return (`table-tennis-return-panda-v1`) | Franka Panda (joint space) | `hold` | **100%** | 0% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Table tennis return (`table-tennis-return-panda-v1`) | Franka Panda (joint space) | `random` | 91% | 2% | 0% | 0% | 0% | 0% | 0/6 | -0% |
| Table tennis return (`table-tennis-return-panda-v1`) | Franka Panda (joint space) | `intercept` | **100%** | **100%** | 16% | 0% | 0% | 12% | 2/6 | -4% |
| Table tennis return (`table-tennis-return-panda-v1`) | Franka Panda (joint space) | `vision` | 99% | 89% | 5% | 0% | 0% | 9% | 0/6 | -5% |
| Table tennis return (`table-tennis-return-g1-v1`) | Unitree G1, fixed base | `hold` | **100%** | 35% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Table tennis return (`table-tennis-return-g1-v1`) | Unitree G1, fixed base | `random` | 0% | 0% | 0% | 0% | 0% | 0% | 0/6 | +0% |
| Table tennis return (`table-tennis-return-g1-v1`) | Unitree G1, fixed base | `intercept` | **100%** | 85% | 5% | 0% | 0% | 4% | 1/6 | -2% |
| Tennis return (`tennis-return-v0`) | mocap racket fixture | `noop` | **100%** | 0% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Tennis return (`tennis-return-v0`) | mocap racket fixture | `scripted` | **100%** | **99%** | 59% | 1% | 22% | 0% | 2/6 | +31% |
| Badminton serve (`badminton-serve-v0`) | mocap racket fixture | `noop` | **100%** | 0% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Badminton serve (`badminton-serve-v0`) | mocap racket fixture | `scripted` | **100%** | **100%** | **100%** | **78%** | **100%** | **79%** | 6/6 | +6% |
| Football kick (`football-kick-v0`) | mocap boot fixture | `noop` | **100%** | 0% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Football kick (`football-kick-v0`) | mocap boot fixture | `scripted` | **100%** | **100%** | **100%** | **100%** | **100%** | 18% | 5/6 | +37% |
| Basketball shoot (`basketball-shoot-v0`) | mocap launcher fixture | `noop` | **100%** | 0% | 0% | 0% | 0% | 0% | 1/6 | +0% |
| Basketball shoot (`basketball-shoot-v0`) | mocap launcher fixture | `scripted` | **100%** | **100%** | **100%** | 65% | 36% | 12% | 3/6 | +52% |

Primary metrics: L0 `incoming_valid_rate`, L1 `hit_rate`, L2 `valid_return_rate`, L3 `target_rate`, L4/L5 the worst pass bucket's `valid_return_rate`. For the launch tasks `valid_return` means a legal serve, a goal or a made basket. Gap is `valid_return_rate` on L2-L3 minus L4-L5.
