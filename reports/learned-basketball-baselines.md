# Learned baseline: basketball

PPO over the swing primitive (`multisport_sim.benchmark.learning`), 5 seeds x 8000 training episodes on the train split (L1-L4), scored on every level of the test split. Regenerate with `python scripts/train_launch_policies.py --sport basketball`.

The policy sees observable geometry only (contact distance and height, ball velocity, target) and knows nothing the simulator was calibrated with. `untrained-primitive` sets every parameter to the middle of its range (the untrained policy's mean); `random-primitive` draws them uniformly.

| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | Untrained primitive | Random primitive |
|---|---|---:|---:|---:|---:|---:|
| L0 | `incoming_valid_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L1 | `hit_rate` | 100% | 100%–100% | 5/5 | 100% | 100% |
| L2 | `valid_return_rate` | 36% | 24%–49% | 0/5 | 2% | 4% |
| L3 | `target_rate` | 9% | 7%–12% | 0/5 | 0% | 2% |
| L4 | `worst_bucket_valid_return_rate` | 2% | 0%–6% | 0/5 | 4% | 0% |
| L5 | `worst_bucket_valid_return_rate` | 7% | 0%–18% | 0/5 | 0% | 0% |

## Training

| Seed | Wall time | First 128 | Last 128 |
|---:|---:|---:|---:|
| 0 | 10.9 min | 0.12 | 0.16 |
| 1 | 10.8 min | 0.13 | 0.21 |
| 2 | 11.2 min | 0.14 | 0.12 |
| 3 | 11.0 min | 0.13 | 0.12 |
| 4 | 11.3 min | 0.13 | 0.13 |

Hardware: {'cpu': 'x86_64', 'cpu_count': 32, 'python': '3.13.9', 'torch': '2.14.0+cpu'}. Reward per episode: `0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success`.
