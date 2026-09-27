# Backend parity: table-tennis ball flight and bounce

Bank `table_tennis/return-v1` split `dev`, 300 shots, blade parked. Reference mujoco 3.13.0 vs candidate isaacsim 5.0.0.0 / isaaclab 0.46.2 (cuda:0). Generated 2026-09-27T17:30:11+00:00.

| Quantity | median | p95 | max |
|---|---:|---:|---:|
| Flight divergence before first bounce (mm) | 3.4 | 4.3 | 286.9 |
| First-bounce time difference (ms) | 0.0 | 5.0 | 10.0 |
| First-bounce position difference (mm) | 1.5 | 19.3 | 42.5 |
| Divergence after bounce (mm) | 74.7 | 211.0 | 265.4 |
| Post-bounce apex height, reference (mm) | 1163.9 | 1330.4 | 1403.5 |
| Post-bounce apex height, candidate (mm) | 1179.8 | 1276.8 | 1310.4 |
| Post-bounce apex height difference (mm) | 49.2 | 102.4 | 507.9 |

Judge agreement: incoming_valid 98.7%, failure_reason 99.3%.

## Verdict disagreements

| shot | tags | MuJoCo | Isaac |
|---|---|---|---|
| tt-return-v1-dev-l4-0020 | backhand, deep, normal-speed, topspin | miss (valid=True) | miss (valid=False) |
| tt-return-v1-dev-l5-0003 | backhand, deep, edge, fast, held-out, underspin | miss (valid=True) | out (valid=False) |
| tt-return-v1-dev-l5-0015 | backhand, deep, fast, held-out, topspin | miss (valid=True) | miss (valid=False) |
| tt-return-v1-dev-l5-0046 | deep, edge, fast, forehand, held-out, topspin | miss (valid=True) | out (valid=False) |
