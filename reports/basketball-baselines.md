# Basketball shoot baselines

`basketball-shoot-v0` at the FIBA basket, split `test` (digest `d4fedb70ba89`), MuJoCo mocap fixture.
Regenerate with `python scripts/run_fixture_baselines.py --sport basketball`.

**Neither row is a submission.** `noop` is the floor; `scripted` reads the privileged ball state and the shot's target, solves an arc through the rim against the drag model and inverts the launcher plate's measured contact law; it bounds what the fixture can do, not what a robot could.

| Level | Controller | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| L0 | `noop` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L0 | `scripted` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L1 | `noop` | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% |
| L1 | `scripted` | 100 | `hit_rate` | 100% | 96%–100% | 90% | PASS | 100% | 0% | 0% |
| L2 | `noop` | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 0% | 0% | 0% |
| L2 | `scripted` | 100 | `valid_return_rate` | 100% | 96%–100% | 80% | PASS | 100% | 100% | 0% |
| L3 | `noop` | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% |
| L3 | `scripted` | 100 | `target_rate` | 65% | 55%–74% | 70% | FAIL | 100% | 100% | 65% |
| L4 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–13% | 60% | FAIL | 0% | 0% | 0% |
| L4 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 36% | 20%–55% | 60% | FAIL | 100% | 84% | 0% |
| L5 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–10% | 50% | FAIL | 0% | 0% | 0% |
| L5 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 12% | 5%–27% | 50% | FAIL | 100% | 13% | 0% |

## Robustness gap

- `noop`: 0% on L2-L3 (200 episodes) minus 0% on L4-L5 (200) = **+0%**; L4 0% (100), L5 0% (100)
- `scripted`: 100% on L2-L3 (200 episodes) minus 48% on L4-L5 (200) = **+52%**; L4 84% (100), L5 13% (100)

L4 and L5 are independently stratified challenge distributions; only L5
applies the declared observation/action/domain perturbations. Their point
estimates therefore need not be monotonic, which is why both are retained.
