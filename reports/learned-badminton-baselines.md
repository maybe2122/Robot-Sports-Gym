# Learned baseline: badminton

PPO over the swing primitive (`multisport_sim.benchmark.learning`), 5 seeds x 8000 training episodes on the train split (L1-L4), scored on every level of the test split. Regenerate with `python scripts/train_launch_policies.py --sport badminton`.

The policy sees observable geometry only (contact distance and height, ball velocity, target) and knows nothing the simulator was calibrated with. `untrained-primitive` sets every parameter to the middle of its range (the untrained policy's mean); `random-primitive` draws them uniformly.

| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | Untrained primitive | Random primitive |
|---|---|---:|---:|---:|---:|---:|
| L0 | `incoming_valid_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L1 | `hit_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L2 | `valid_return_rate` | 100% | 100%–100% | 5/5 | 100% | 62% |
| L3 | `target_rate` | 36% | 36%–37% | 0/5 | 19% | 8% |
| L4 | `worst_bucket_valid_return_rate` | 100% | 100%–100% | 5/5 | 100% | 36% |
| L5 | `worst_bucket_valid_return_rate` | 80% | 78%–82% | 5/5 | 79% | 36% |

## Training

| Seed | Wall time | First 128 | Last 128 |
|---:|---:|---:|---:|
| 0 | 32.4 min | 0.66 | 0.84 |
| 1 | 32.6 min | 0.59 | 0.91 |
| 2 | 31.8 min | 0.59 | 0.82 |
| 3 | 32.3 min | 0.65 | 0.86 |
| 4 | 32.8 min | 0.59 | 0.74 |

Hardware: {'cpu': 'x86_64', 'cpu_count': 32, 'python': '3.13.9', 'torch': '2.14.0+cpu'}. Reward per episode: `0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success`.
