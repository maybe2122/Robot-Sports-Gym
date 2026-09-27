# Learned baseline: tennis

PPO over the swing primitive (`multisport_sim.benchmark.learning`), 5 seeds x 8000 training episodes on the train split (L1-L4), scored on every level of the test split. Regenerate with `python scripts/train_launch_policies.py --sport tennis`.

The policy sees observable geometry only (contact distance and height, ball velocity, target) and knows nothing the simulator was calibrated with. `untrained-primitive` sets every parameter to the middle of its range (the untrained policy's mean); `random-primitive` draws them uniformly.

| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | Untrained primitive | Random primitive |
|---|---|---:|---:|---:|---:|---:|
| L0 | `incoming_valid_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L1 | `hit_rate` | 99% | 99%–99% | 5/5 | 99% | 99% |
| L2 | `valid_return_rate` | 60% | 58%–62% | 0/5 | 58% | 22% |
| L3 | `target_rate` | 2% | 1%–2% | 0/5 | 2% | 0% |
| L4 | `worst_bucket_valid_return_rate` | 21% | 20%–23% | 0/5 | 22% | 5% |
| L5 | `worst_bucket_valid_return_rate` | 0% | 0%–0% | 0/5 | 0% | 0% |

## Training

| Seed | Wall time | First 128 | Last 128 |
|---:|---:|---:|---:|
| 0 | 14.9 min | 0.31 | 0.36 |
| 1 | 14.8 min | 0.28 | 0.28 |
| 2 | 14.8 min | 0.29 | 0.26 |
| 3 | 14.7 min | 0.25 | 0.30 |
| 4 | 14.8 min | 0.36 | 0.28 |

Hardware: {'cpu': 'x86_64', 'cpu_count': 32, 'python': '3.13.9', 'torch': '2.14.0+cpu'}. Reward per episode: `0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success`.
