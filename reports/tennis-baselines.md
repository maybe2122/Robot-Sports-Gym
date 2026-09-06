# Tennis return baselines

`tennis-return-v0` on the regulation singles court, split `test` (digest `b0614a72ee8e`), MuJoCo mocap fixture.
Regenerate with `python scripts/run_tennis_baselines.py`.

**Neither row is a submission.** `noop` is the floor; `scripted` reads
privileged ball state and moves a kinematic racket, so it bounds what
the fixture can do rather than what a robot could.  Tennis has no
embodied task yet: there is no arm holding the racket.

| Level | Controller | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| L0 | `noop` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L0 | `scripted` | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% |
| L1 | `noop` | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% |
| L1 | `scripted` | 100 | `hit_rate` | 99% | 95%–100% | 90% | PASS | 99% | 0% | 0% |
| L2 | `noop` | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 0% | 0% | 0% |
| L2 | `scripted` | 100 | `valid_return_rate` | 59% | 49%–68% | 80% | FAIL | 95% | 59% | 0% |
| L3 | `noop` | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% |
| L3 | `scripted` | 100 | `target_rate` | 1% | 0%–5% | 70% | FAIL | 100% | 68% | 1% |
| L4 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–9% | 60% | FAIL | 0% | 0% | 0% |
| L4 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 22% | 12%–37% | 60% | FAIL | 90% | 56% | 0% |
| L5 | `noop` | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–5% | 50% | FAIL | 0% | 0% | 0% |
| L5 | `scripted` | 100 | `worst_bucket_valid_return_rate` | 32% | 22%–43% | 50% | FAIL | 80% | 46% | 0% |

## Robustness gap

- `noop`: 0% on L1-L3 (300 episodes) minus 0% on L4-L5 (200) = **+0%**; L4 0% (100), L5 0% (100)
- `scripted`: 42% on L1-L3 (300 episodes) minus 51% on L4-L5 (200) = **-9%**; L4 56% (100), L5 46% (100)

L4 and L5 are independently stratified challenge distributions; only L5
applies the declared observation/action/domain perturbations. Their point
estimates therefore need not be monotonic, which is why both are retained.
