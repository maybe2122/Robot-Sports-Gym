# Learned baseline: football

PPO over the swing primitive (`multisport_sim.benchmark.learning`), 5 seeds x 8000 training episodes on the train split (L1-L4), scored on every level of the test split. Regenerate with `python scripts/train_launch_policies.py --sport football`.

The policy sees observable geometry only (contact distance and height, ball velocity, target) and knows nothing the simulator was calibrated with. `untrained-primitive` sets every parameter to the middle of its range (the untrained policy's mean); `random-primitive` draws them uniformly.

| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | Untrained primitive | Random primitive |
|---|---|---:|---:|---:|---:|---:|
| L0 | `incoming_valid_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L1 | `hit_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L2 | `valid_return_rate` | 100% | 100%–100% | 5/5 | 100% | 78% |
| L3 | `target_rate` | 9% | 6%–11% | 0/5 | 49% | 9% |
| L4 | `worst_bucket_valid_return_rate` | 100% | 100%–100% | 5/5 | 100% | 56% |
| L5 | `worst_bucket_valid_return_rate` | 27% | 25%–28% | 0/5 | 21% | 15% |

## Training

| Seed | Wall time | First 128 | Last 128 |
|---:|---:|---:|---:|
| 0 | 9.4 min | 0.74 | 0.83 |
| 1 | 9.5 min | 0.72 | 0.86 |
| 2 | 9.2 min | 0.74 | 0.80 |
| 3 | 9.3 min | 0.67 | 0.86 |
| 4 | 9.4 min | 0.60 | 0.78 |

Hardware: {'cpu': 'x86_64', 'cpu_count': 32, 'python': '3.13.9', 'torch': '2.14.0+cpu'}. Reward per episode: `0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success`.
