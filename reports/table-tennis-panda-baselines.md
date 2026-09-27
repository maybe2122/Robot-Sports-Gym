# Embodied table-tennis baselines

`table-tennis-return-panda-v1` on a Franka Panda, MuJoCo backend.
Shot bank `table_tennis/return-v1`: dev has 50 and test 100 episodes per level.
Regenerate with `python scripts/run_baselines.py --robot panda`.

**None of these rows is a submission.** `hold` and `random` are floors;
`intercept` is a scripted controller reading privileged ball state, so its
score bounds what this embodiment can do, it does not represent a policy.
`vision` runs the identical swing from the declared stereo pair with no
privileged state, so the gap between the two is the cost of perception.

## split `dev` (digest `f17c0241aec2`)

| Level | Controller | Track | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |
|---|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| L0 | `hold` | state | 50 | `incoming_valid_rate` | 100% | 93%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 0.2 |
| L0 | `random` | state | 50 | `incoming_valid_rate` | 90% | 79%–96% | 100% | FAIL | 0% | 0% | 0% | 5 | 66.4 |
| L0 | `intercept` | state | 50 | `incoming_valid_rate` | 100% | 93%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 31.2 |
| L0 | `vision` | vision | 50 | `incoming_valid_rate` | 100% | 93%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 51.1 |
| L1 | `hold` | state | 50 | `hit_rate` | 0% | 0%–7% | 90% | FAIL | 0% | 0% | 0% | 0 | 0.2 |
| L1 | `random` | state | 50 | `hit_rate` | 4% | 1%–13% | 90% | FAIL | 4% | 0% | 0% | 12 | 145.6 |
| L1 | `intercept` | state | 50 | `hit_rate` | 100% | 93%–100% | 90% | PASS | 100% | 0% | 0% | 0 | 49.2 |
| L1 | `vision` | vision | 50 | `hit_rate` | 82% | 69%–90% | 90% | FAIL | 82% | 0% | 0% | 4 | 74.0 |
| L2 | `hold` | state | 50 | `valid_return_rate` | 2% | 0%–10% | 80% | FAIL | 8% | 2% | 0% | 0 | 0.2 |
| L2 | `random` | state | 50 | `valid_return_rate` | 0% | 0%–7% | 80% | FAIL | 2% | 0% | 0% | 11 | 127.5 |
| L2 | `intercept` | state | 50 | `valid_return_rate` | 26% | 16%–40% | 80% | FAIL | 94% | 26% | 0% | 0 | 62.9 |
| L2 | `vision` | vision | 50 | `valid_return_rate` | 12% | 6%–24% | 80% | FAIL | 68% | 12% | 0% | 4 | 91.2 |
| L3 | `hold` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 2% | 0% | 0% | 0 | 0.2 |
| L3 | `random` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 4% | 0% | 0% | 10 | 132.1 |
| L3 | `intercept` | state | 50 | `target_rate` | 0% | 0%–7% | 70% | FAIL | 100% | 10% | 0% | 0 | 63.6 |
| L3 | `vision` | vision | 50 | `target_rate` | 2% | 0%–10% | 70% | FAIL | 80% | 10% | 2% | 1 | 96.1 |
| L4 | `hold` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 60% | FAIL | 12% | 0% | 0% | 0 | 0.2 |
| L4 | `random` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 60% | FAIL | 12% | 0% | 0% | 10 | 121.4 |
| L4 | `intercept` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–22% | 60% | FAIL | 60% | 30% | 0% | 5 | 79.1 |
| L4 | `vision` | vision | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–22% | 60% | FAIL | 42% | 4% | 0% | 8 | 105.2 |
| L5 | `hold` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 50% | FAIL | 10% | 0% | 0% | 0 | 0.2 |
| L5 | `random` | state | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–8% | 50% | FAIL | 4% | 0% | 0% | 9 | 110.6 |
| L5 | `intercept` | state | 50 | `worst_bucket_valid_return_rate` | 14% | 3%–51% | 50% | FAIL | 36% | 22% | 0% | 7 | 78.7 |
| L5 | `vision` | vision | 50 | `worst_bucket_valid_return_rate` | 0% | 0%–35% | 50% | FAIL | 20% | 6% | 0% | 11 | 81.8 |

## split `test` (digest `146bf82e111b`)

| Level | Controller | Track | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |
|---|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| L0 | `hold` | state | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 0.2 |
| L0 | `random` | state | 100 | `incoming_valid_rate` | 91% | 84%–95% | 100% | FAIL | 0% | 0% | 0% | 9 | 65.8 |
| L0 | `intercept` | state | 100 | `incoming_valid_rate` | 100% | 96%–100% | 100% | PASS | 0% | 0% | 0% | 0 | 30.7 |
| L0 | `vision` | vision | 100 | `incoming_valid_rate` | 99% | 95%–100% | 100% | FAIL | 0% | 0% | 0% | 1 | 51.2 |
| L1 | `hold` | state | 100 | `hit_rate` | 0% | 0%–4% | 90% | FAIL | 0% | 0% | 0% | 0 | 0.2 |
| L1 | `random` | state | 100 | `hit_rate` | 2% | 1%–7% | 90% | FAIL | 2% | 0% | 0% | 18 | 149.4 |
| L1 | `intercept` | state | 100 | `hit_rate` | 100% | 96%–100% | 90% | PASS | 100% | 0% | 0% | 0 | 48.5 |
| L1 | `vision` | vision | 100 | `hit_rate` | 89% | 81%–94% | 90% | FAIL | 89% | 0% | 0% | 2 | 73.1 |
| L2 | `hold` | state | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 7% | 0% | 0% | 0 | 0.2 |
| L2 | `random` | state | 100 | `valid_return_rate` | 0% | 0%–4% | 80% | FAIL | 2% | 0% | 0% | 16 | 134.7 |
| L2 | `intercept` | state | 100 | `valid_return_rate` | 16% | 10%–24% | 80% | FAIL | 94% | 16% | 0% | 1 | 64.9 |
| L2 | `vision` | vision | 100 | `valid_return_rate` | 5% | 2%–11% | 80% | FAIL | 70% | 5% | 0% | 11 | 88.5 |
| L3 | `hold` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 0% | 0% | 0% | 0 | 0.2 |
| L3 | `random` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 6% | 0% | 0% | 15 | 133.0 |
| L3 | `intercept` | state | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 99% | 17% | 0% | 0 | 61.4 |
| L3 | `vision` | vision | 100 | `target_rate` | 0% | 0%–4% | 70% | FAIL | 78% | 13% | 0% | 4 | 85.7 |
| L4 | `hold` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 60% | FAIL | 19% | 0% | 0% | 0 | 0.2 |
| L4 | `random` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 60% | FAIL | 4% | 0% | 0% | 16 | 125.8 |
| L4 | `intercept` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–12% | 60% | FAIL | 59% | 28% | 0% | 6 | 87.9 |
| L4 | `vision` | vision | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–12% | 60% | FAIL | 44% | 17% | 0% | 22 | 87.8 |
| L5 | `hold` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–4% | 50% | FAIL | 16% | 0% | 0% | 0 | 0.2 |
| L5 | `random` | state | 100 | `worst_bucket_valid_return_rate` | 0% | 0%–22% | 50% | FAIL | 4% | 1% | 0% | 10 | 110.8 |
| L5 | `intercept` | state | 100 | `worst_bucket_valid_return_rate` | 12% | 7%–20% | 50% | FAIL | 31% | 13% | 0% | 11 | 83.8 |
| L5 | `vision` | vision | 100 | `worst_bucket_valid_return_rate` | 9% | 4%–16% | 50% | FAIL | 34% | 12% | 0% | 16 | 90.3 |

## Robustness gap

`valid_return_rate` on L2-L3 minus the same rate on L4-L5. L4 and L5
are also shown separately: they are independently stratified challenge
distributions (and only L5 applies declared perturbations), so their
empirical rates are not expected to be monotonic.

| Split | Controller | L2-L3 | L4 | L5 | L4-L5 | Gap |
|---|---|---:|---:|---:|---:|---:|
| `dev` | `hold` | 1% (100) | 0% (50) | 0% (50) | 0% (100) | +1% |
| `dev` | `random` | 0% (100) | 0% (50) | 0% (50) | 0% (100) | +0% |
| `dev` | `intercept` | 18% (100) | 30% (50) | 22% (50) | 26% (100) | -8% |
| `dev` | `vision` | 11% (100) | 4% (50) | 6% (50) | 5% (100) | +6% |
| `test` | `hold` | 0% (200) | 0% (100) | 0% (100) | 0% (200) | +0% |
| `test` | `random` | 0% (200) | 0% (100) | 1% (100) | 0% (200) | -0% |
| `test` | `intercept` | 16% (200) | 28% (100) | 13% (100) | 20% (200) | -4% |
| `test` | `vision` | 9% (200) | 17% (100) | 12% (100) | 14% (200) | -5% |
