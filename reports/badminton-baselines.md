# Badminton serve baselines

`badminton-serve-v0` into the BWF singles service court, split `test` (digest `2ca5c08e7074`), MuJoCo mocap fixture.
Regenerate with `python scripts/run_fixture_baselines.py --sport badminton`.

**Neither row is a submission.** `noop` is the floor; `scripted` reads the privileged shuttle state and the shot's target and swings a kinematic racket along a straight line at a speed read from a measured carry table (`scripts/calibrate_serve.py`); it bounds what the fixture can do, not what a robot could.  There is no embodied badminton task yet.

| Level | Controller | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| L0 | `noop` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L0 | `scripted` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L1 | `noop` | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% |
| L1 | `scripted` | 100 | `hit_rate` | 100% | 96%–100% | 90% | PASS | 100% | 0% | 0% |
| L2 | `noop` | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 0% | 0% | 0% |
| L2 | `scripted` | 100 | `valid_return_rate` | 100% | 96%–100% | 80% | PASS | 100% | 100% | 0% |
| L3 | `noop` | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% |
| L3 | `scripted` | 100 | `target_rate` | 78% | 69%–85% | 70% | PASS | 100% | 100% | 78% |
| L4 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–13% | 60% | FAIL | 0% | 0% | 0% |
| L4 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 100% | 87%–100% | 60% | PASS | 100% | 100% | 0% |
| L5 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–10% | 50% | FAIL | 0% | 0% | 0% |
| L5 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 79% | 63%–90% | 50% | PASS | 98% | 87% | 0% |

## Robustness gap

- `noop`: 0% on L2-L3 (200 episodes) minus 0% on L4-L5 (200) = **+0%**; L4 0% (100), L5 0% (100)
- `scripted`: 100% on L2-L3 (200 episodes) minus 94% on L4-L5 (200) = **+6%**; L4 100% (100), L5 87% (100)

L4 and L5 are independently stratified challenge distributions; only L5
applies the declared observation/action/domain perturbations. Their point
estimates therefore need not be monotonic, which is why both are retained.
