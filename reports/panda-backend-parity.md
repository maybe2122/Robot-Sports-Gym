# Panda on two backends: the same controller, the same shots

`scripted-panda-intercept-v1` on `table_tennis/return-v1` split `dev`. Reference MuJoCo (Menagerie Panda); candidate Isaac Lab / PhysX (Franka USD). Regenerate with `scripts/isaac_panda_rollout.py`.

| Level | n | Hit MuJoCo | Hit Isaac | Hit agreement | Return MuJoCo | Return Isaac | Return agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 50 | 100% | 90% | 90% | 0% | 0% | 100% |
| L2 | 50 | 94% | 92% | 90% | 26% | 0% | 74% |
**Reading.** Interception transfers between the backends: the same controller hits 90–92% of the same shots on Isaac against 94–100% on MuJoCo, and the two agree shot by shot on 90% of them. Returning does not: 26% of L2 shots come back legally on MuJoCo and none on Isaac. Blade speed at contact matches (0.6–0.9 m/s on both), and after calibrating the PhysX table friction (horizontal speed kept through a bounce: 77% → 90%, MuJoCo 98%) and the blade's rubber material (restitution 0.84, friction 0.85), Isaac's outgoing forward speed is still ~2 m/s where MuJoCo's successful returns leave at 3.5–4.7 m/s. MuJoCo's blade contact is a soft 23 ms constraint; PhysX resolves the same nominal coefficients as a rigid impulse. Reconciling them is contact calibration against measured data (M4), not a parameter to tune until the numbers agree.
