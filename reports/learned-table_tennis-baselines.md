# Learned baseline: table_tennis

PPO over the swing primitive (`multisport_sim.benchmark.learning`), 5 seeds x 8000 training episodes on the train split (L1-L4), scored on every level of the test split. Regenerate with `python scripts/train_launch_policies.py --sport table_tennis`.

The policy sees observable geometry only (contact distance and height, ball velocity, target) and knows nothing the simulator was calibrated with. `untrained-primitive` sets every parameter to the middle of its range (the untrained policy's mean); `random-primitive` draws them uniformly.

| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | Untrained primitive | Random primitive |
|---|---|---:|---:|---:|---:|---:|
| L0 | `incoming_valid_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L1 | `hit_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L2 | `valid_return_rate` | 98% | 97%–99% | 5/5 | 0% | 13% |
| L3 | `target_rate` | 25% | 19%–30% | 0/5 | 0% | 4% |
| L4 | `worst_bucket_valid_return_rate` | 87% | 84%–89% | 5/5 | 0% | 7% |
| L5 | `worst_bucket_valid_return_rate` | 42% | 37%–48% | 0/5 | 14% | 9% |

## Training

| Seed | Wall time | First 128 | Last 128 |
|---:|---:|---:|---:|
| 0 | 9.1 min | 0.23 | 0.43 |
| 1 | 9.0 min | 0.24 | 0.35 |
| 2 | 8.9 min | 0.19 | 0.45 |
| 3 | 9.1 min | 0.19 | 0.44 |
| 4 | 8.8 min | 0.28 | 0.36 |

Hardware: {'cpu': 'x86_64', 'cpu_count': 32, 'python': '3.13.9', 'torch': '2.14.0+cpu'}. Reward per episode: `0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success`.
