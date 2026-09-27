# Football kick baselines

`football-kick-v0` at the IFAB goal, split `test` (digest `dbf7ed16a8a3`), MuJoCo mocap fixture.
Regenerate with `python scripts/run_fixture_baselines.py --sport football`.

**Neither row is a submission.** `noop` is the floor; `scripted` reads the privileged ball state and the shot's target, solves the launch against the drag model and inverts the boot's measured contact law; it bounds what the fixture can do, not what a robot could.  There is no embodied football task yet (the biped asset is not in).

| Level | Controller | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| L0 | `noop` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L0 | `scripted` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L1 | `noop` | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% |
| L1 | `scripted` | 100 | `hit_rate` | 100% | 96%–100% | 90% | PASS | 100% | 0% | 0% |
| L2 | `noop` | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 0% | 0% | 0% |
| L2 | `scripted` | 100 | `valid_return_rate` | 100% | 96%–100% | 80% | PASS | 100% | 100% | 0% |
| L3 | `noop` | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% |
| L3 | `scripted` | 100 | `target_rate` | 100% | 96%–100% | 70% | PASS | 100% | 100% | 100% |
| L4 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–13% | 60% | FAIL | 0% | 0% | 0% |
| L4 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 100% | 87%–100% | 60% | PASS | 100% | 100% | 0% |
| L5 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 50% | FAIL | 0% | 0% | 0% |
| L5 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 18% | 9%–34% | 50% | FAIL | 88% | 26% | 0% |

## Robustness gap

- `noop`: 0% on L2-L3 (200 episodes) minus 0% on L4-L5 (200) = **+0%**; L4 0% (100), L5 0% (100)
- `scripted`: 100% on L2-L3 (200 episodes) minus 63% on L4-L5 (200) = **+37%**; L4 100% (100), L5 26% (100)

L4 and L5 are independently stratified challenge distributions; only L5
applies the declared observation/action/domain perturbations. Their point
estimates therefore need not be monotonic, which is why both are retained.
