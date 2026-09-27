# Embodied table-tennis baselines

`table-tennis-return-g1-v1` on a fixed-base Unitree G1 (waist + right arm, 10 joints), MuJoCo backend.
Shot bank `table_tennis/return-v1`: dev has 50 and test 100 episodes per level.
Regenerate with `python scripts/run_baselines.py --robot g1`.

**None of these rows is a submission.** `hold` and `random` are floors;
`intercept` is a scripted controller reading privileged ball state, so its
score bounds what this embodiment can do, it does not represent a policy.

## split `dev` (digest `f17c0241aec2`)

| Level | Controller | Track | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |
|---|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| L0 | `hold` | state | 50 | `incoming_valid_rate` | 100% | 93%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 0.7 |
| L0 | `random` | state | 50 | `incoming_valid_rate` | 0% | 0%–7% | 100% | FAIL | 0% | 0% | 0% | 50 | 21.6 |
| L0 | `intercept` | state | 50 | `incoming_valid_rate` | 100% | 93%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 9.7 |
| L1 | `hold` | state | 50 | `hit_rate` | 28% | 17%–42% | 90% | FAIL | 28% | 0% | 0% | 0 | 0.7 |
| L1 | `random` | state | 50 | `hit_rate` | 0% | 0%–7% | 90% | FAIL | 0% | 0% | 0% | 50 | 21.6 |
| L1 | `intercept` | state | 50 | `hit_rate` | 84% | 71%–92% | 90% | FAIL | 84% | 0% | 0% | 0 | 16.8 |
| L2 | `hold` | state | 50 | `valid_return_rate` | 0% | 0%–7% | 80% | FAIL | 16% | 0% | 0% | 0 | 0.7 |
| L2 | `random` | state | 50 | `valid_return_rate` | 0% | 0%–7% | 80% | FAIL | 0% | 0% | 0% | 50 | 21.6 |
| L2 | `intercept` | state | 50 | `valid_return_rate` | 2% | 0%–10% | 80% | FAIL | 96% | 2% | 0% | 0 | 25.4 |
| L3 | `hold` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 38% | 4% | 0% | 0 | 0.7 |
| L3 | `random` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 0% | 0% | 0% | 50 | 21.6 |
| L3 | `intercept` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 98% | 0% | 0% | 0 | 25.1 |
| L4 | `hold` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 60% | FAIL | 10% | 0% | 0% | 0 | 0.7 |
| L4 | `random` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 60% | FAIL | 0% | 0% | 0% | 50 | 21.6 |
| L4 | `intercept` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–22% | 60% | FAIL | 66% | 4% | 0% | 0 | 24.2 |
| L5 | `hold` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 50% | FAIL | 10% | 0% | 0% | 0 | 0.7 |
| L5 | `random` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 50% | FAIL | 0% | 0% | 0% | 50 | 24.3 |
| L5 | `intercept` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–35% | 50% | FAIL | 22% | 8% | 0% | 1 | 21.8 |

## split `test` (digest `146bf82e111b`)

| Level | Controller | Track | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |
|---|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| L0 | `hold` | state | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 0.7 |
| L0 | `random` | state | 100 | `incoming_valid_rate` | 0% | 0%–4% | 100% | FAIL | 0% | 0% | 0% | 100 | 25.2 |
| L0 | `intercept` | state | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 9.6 |
| L1 | `hold` | state | 100 | `hit_rate` | 35% | 26%–45% | 90% | FAIL | 35% | 0% | 0% | 0 | 0.7 |
| L1 | `random` | state | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% | 100 | 25.2 |
| L1 | `intercept` | state | 100 | `hit_rate` | 85% | 77%–91% | 90% | FAIL | 85% | 0% | 0% | 0 | 16.6 |
| L2 | `hold` | state | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 16% | 0% | 0% | 0 | 0.7 |
| L2 | `random` | state | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 0% | 0% | 0% | 100 | 25.2 |
| L2 | `intercept` | state | 100 | `valid_return_rate` | 5% | 2%–11% | 80% | FAIL | 91% | 5% | 0% | 0 | 25.2 |
| L3 | `hold` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 53% | 0% | 0% | 0 | 0.7 |
| L3 | `random` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% | 100 | 25.2 |
| L3 | `intercept` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 96% | 2% | 0% | 0 | 25.3 |
| L4 | `hold` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 60% | FAIL | 15% | 0% | 0% | 0 | 0.7 |
| L4 | `random` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 60% | FAIL | 0% | 0% | 0% | 100 | 25.2 |
| L4 | `intercept` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–12% | 60% | FAIL | 62% | 6% | 0% | 0 | 23.8 |
| L5 | `hold` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 50% | FAIL | 8% | 0% | 0% | 0 | 0.7 |
| L5 | `random` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 50% | FAIL | 0% | 0% | 0% | 100 | 26.5 |
| L5 | `intercept` | state | 100 | `worst_bucket_valid_return_rate` | 4% | 2%–10% | 50% | FAIL | 22% | 4% | 0% | 0 | 22.7 |

## Robustness gap

`valid_return_rate` on L1-L3 minus the same rate on L4-L5. L4 and L5
are also shown separately: they are independently stratified challenge
distributions (and only L5 applies declared perturbations), so their
empirical rates are not expected to be monotonic.

| Split | Controller | L1-L3 | L4 | L5 | L4-L5 | Gap |
|---|---|---:|---:|---:|---:|---:|
| `dev` | `hold` | 1% (150) | 0% (50) | 0% (50) | 0% (100) | +1% |
| `dev` | `random` | 0% (150) | 0% (50) | 0% (50) | 0% (100) | +0% |
| `dev` | `intercept` | 1% (150) | 4% (50) | 8% (50) | 6% (100) | -5% |
| `test` | `hold` | 0% (300) | 0% (100) | 0% (100) | 0% (200) | +0% |
| `test` | `random` | 0% (300) | 0% (100) | 0% (100) | 0% (200) | +0% |
| `test` | `intercept` | 2% (300) | 6% (100) | 4% (100) | 5% (200) | -3% |
