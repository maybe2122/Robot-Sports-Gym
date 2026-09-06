# M2 机器人层 — 进度记录

**日期：2026-08-24**
**分支：`feat/shared-task-config`**
**范围：让乒乓球成为可用的机器人测试基准**

本文件记录这一批实际完成了什么、验证到什么程度、以及明确没有做的部分。设计说明见
[`ROBOT_LAYER.md`](ROBOT_LAYER.md)，任务拆解见 [`TODO.md`](TODO.md)。

---

## 一、起点：为什么需要这一批

2026-08-24 盘点时，乒乓球已有完整的**评测管线**（固定 Shot Bank、语义接触判定、Gymnasium 环境、
Runner/CLI、JSON+Markdown 报告、Wilson 区间与分桶），但**不是机器人基准**：动作是 mocap 拍面 pose，
MuJoCo 直接把拍面瞬移过去。没有关节就无从定义 `safety_violations`，没有执行器就无从积分
`energy_joule`，没有动力学就无从训练可迁移的策略。

一个机器人基准需要提供的六条能力基线，以及当时的缺口：

| # | 能力基线 | 2026-08-24 起点 | 本批次后 |
|---|---|---|---|
| 1 | 有动力学的机器人本体 | 无（mocap 瞬移） | **已完成**（Franka Panda，关节空间） |
| 2 | 明确的观测契约 | 特权球状态，无传感器层 | **已完成**（2026-08-30）：state 与 vision 两条轨道 |
| 3 | 统计充分的固定集 + 种子协议 | dev 12 / test 24，无 train split | **已完成**：Panda 默认 v1，train/dev/test = 1200/300/600 |
| 4 | 完整原始指标 | 缺 safety / energy / contact_error / robustness_gap / latency | **已完成**：八项指标齐全，bucket 另带 Wilson 区间 |
| 5 | 参考分数线 | 只有 noop + scripted mocap | **已完成**四条机体基线（含 vision，均非提交成绩） |
| 6 | 可复现提交链路 | 报告扎实，缺 `submission/` 打包 | **已完成**：`make submission` |

---

## 二、已完成的工作

### 2.1 后端无关机器人契约（TODO #6、#7、#8）

新文件 `src/multisport_sim/benchmark/robot.py`，不导入任何模拟器。

- `RobotAdapter` Protocol：`robot_id`、**有序**的 `joint_names`、`dof`、`effector_name`、
  `control_modes`、`safety_limits`、`reset/observe/apply/describe`。任务代码只按下标和语义角色访问。
- `ControlMode` 枚举 + `JointCommand` / `EffectorPoseCommand`。未实现的模式抛
  `UnsupportedControlMode`，而不是静默降级成别的模式。
- `RobotObservation`，含 `mechanical_power_w()`（能耗指标的被积函数，对制动方向也计费）。
- `JointLimits`、`WorkspaceBox`、`SafetyLimits`、`SafetyViolation`、`SafetyMonitor`。
  `SafetyLimits.violations()` 覆盖五类：`joint_position` / `joint_velocity` / `joint_torque` /
  `collision` / `workspace`，一次观测里**所有**违规都返回而不是第一条。

**验证**：`tests/test_robot.py`，24 项，全部通过，不需要任何资产。

### 2.2 资产解析与许可证清单（TODO #10）

新文件 `src/multisport_sim/benchmark/assets.py`。**不 vendoring**：查找顺序为
`MULTISPORT_MENAGERIE_PATH` → `MUJOCO_MENAGERIE_PATH` → `~/mujoco_menagerie`，缺资产时
`require_asset()` 抛出带获取命令的 `AssetUnavailableError`，测试自动 skip。

选定 **Franka Emika Panda**（`panda_nohand.xml`，Apache-2.0）。理由是硬条件而非偏好：MuJoCo
Menagerie 和 Isaac Lab **同时**自带它，这是让"同一机器人跑两个后端"这条发布门槛低成本成立的唯一
选择；7 DoF 在固定拍面位置和法向后还剩一个冗余自由度，六轴臂做不到。

四条本地修改全部记录在 `AssetSource.modifications` 并进入每份报告。

**验证**：`tests/test_robot_assets.py`，12 项，全部通过（其中 2 项需资产）。

### 2.3 Panda 适配器与场景装配

新文件 `src/multisport_sim/benchmark/robots/{panda,kinematics}.py`、
`src/multisport_sim/benchmark/backends/mujoco_robot.py`。

- 用 `mujoco.MjSpec.attach` 把手臂挂进现有乒乓球场景，前缀 `rb_`，上游文件不改。已验证场景的
  `degree` 与手臂的 `radian` 编译器差异被正确处理（关节 4 的非对称限位 `[-3.0718, -0.0698]` 完整迁移）。
- 球拍作为 **运动链的一部分** 挂在手臂 `attachment` body 下，带 0.17 kg 真实质量。击球是手臂动力学
  产生的真实接触，不是 mocap 瞬移。
- 拍面 site 绕 x 轴转 −90°，使 **site 的 z 轴等于拍面法向**，瞄准球拍因此是单轴约束，绕法向的旋转留给
  冗余度使用。
- 阻尼最小二乘 IK（`IKSolver`），支持全姿态约束和**仅轴约束**两种模式。

**验证**：`tests/test_panda_adapter.py`，16 项，全部通过。

### 2.4 球拍胶皮接触标定

给机器人模型的球-拍面单独定义接触对：`solref=(0.023, 0.042)`，`solimp=(0.96, 0.99, 0.001, 0.5, 2.0)`，
`condim=6`。落球测试实测恢复系数 **e ≈ 0.82–0.86**，落在实测反胶区间内。

这个数是承重的：没有专门接触对时拍面继承场景默认阻尼比 0.7，回球恢复系数只有 **0.17**——手臂等于在挥
一块海绵，任何可达拍速都产生不了合法回球。

该接触对**只加在机器人模型上**，v0 的 mocap 夹具保持冻结时的参数，避免静默移动已有的 v0 分数。

### 2.5 可达性标定（本批次的核心发现）

新脚本 `scripts/calibrate_reachability.py`，输出 `reports/table-tennis-panda-reachability.json`。
所有机器人侧常量都来自它。

冻结结果：

| 量 | 值 |
|---|---|
| 击球平面 | `x = -1.55` m，36/36 条来球全部穿过 |
| 到达时刻 | 0.532 – 1.133 s |
| 横向跨度 | `y ∈ [-0.386, +0.319]` m |
| 高度跨度 | `z ∈ [0.923, 1.310]` m |
| 底座 | `(-1.95, 0, 0.75)`，击球平面覆盖率 97.5% |
| **最坏情况关节行程时间** | **0.525 s** |
| **最短拦截窗口** | **0.532 s** |
| 拍面前向最大速度 | 1.34 – 3.50 m/s |
| 合法回球所需球速（吊高） | 3.36 – 6.33 m/s |

三条值得记录的发现：

1. **原动作空间宽了 4 倍。** `EffectorWorkspace` 原本声明 `y ∈ [-1.5, 1.5]`（3 m 跨度），任何定基座臂
   都够不着；实测只需要 `y ∈ [-0.39, +0.32]`。声明过宽的动作空间不会让任务更通用，只会让它说谎。
2. **Panda 能拦到每一球，余量只有 7 ms。** 拦截是行程时间问题。
3. **但大部分球回不过去。** 回球是速度问题：需要 3.36–6.33 m/s 的出球速度，而拍面前向最大速度只有
   1.34–3.50 m/s。两者以不同方式失败，因此分开报告。

`PANDA_READY_QPOS` 是最坏情况关节行程时间的最优解，不是随便一个 home 位姿。这里有个容易掉进去的坑：
冗余臂有很多位形能到达同一个拍面位姿，如果拿"求解器从该 pose 起步永远不会返回的解集"来打分，这个 pose
看起来会好上大约一倍。必须对求解器的**实际行为**打分。按正确口径优化，从 1.141 s 降到 0.525 s。

### 2.6 新任务与环境

`table-tennis-return-panda-v1` / `MultiSportRobot/TableTennisReturn-Panda-v1`。

与 `table-tennis-return-v0` 是**并列的版本化任务，不是新旧关系**：共享同一个 Judge、同一份固定 Shot
Bank、同一组 reward 权重，唯一差别是谁在击球。v0 保留不动，继续作为规则引擎的回归夹具。

- 动作：7 维关节位置设定值（rad），发布的就是手臂真实关节范围而非归一化到 `[-1,1]`。
- 观测：33 维 = 球位置/线速度/角速度(9) + 关节位置/速度(14) + 拍面位置/四元数/线速度(10)。
- 安全：任一类违规使 `terminated=True` 且 `failure_reason="safety"`，**逐物理步**检查。

**验证**：`tests/test_robot_env.py`，16 项，全部通过，含 Gymnasium `check_env`。

### 2.7 参考基线（三条，均非可提交成绩）

| 控制器 | `controller_id` |
|---|---|
| `hold` | `hold-pose-v1` |
| `random` | `random-joint-v1` |
| `intercept` | `scripted-panda-intercept-v1` |

`JointRateLimiter` 放在**控制器**里而不是适配器里：适配器如果对每条命令静默限速，安全包络就变得不可
证伪，也分不出一个策略是否尊重硬件。

脚本拦截器使用特权球状态（和它替代的 mocap 夹具一样，所以分数不是机器人策略成绩），但不碰正在运行的
仿真——模型和求解器都是它自己的。

调参经过 72 组网格搜索，最优组合为 `blade_tilt_deg=24`、`swing_lead_s=0.16`、`follow_through_m=0.45`、
`rate_margin=0.95`、`backswing_m=0.10`，已设为默认值。

### 2.8 指标与报告

- `run_shots` 对带机器人的后端逐物理步检查安全违规并积分执行器机械功。
- 报告新增 `robot_metrics`：`safety_violations`、`energy_joule`、`total_safety_violations`、
  `mean_energy_joule`。补齐 `BENCHMARK_SPEC` §7 的两项缺失指标。
- 对 mocap 夹具这两项是**空数组，不是零**：它没有安全包络也没有执行器，报 0 等于宣称通过了从未执行的检查。
- 报告的 `task` 字段现在是产生该结果的任务 id，bank 名字移到 `shot_bank_task`。
- CLI 新增 `--robot {none,panda}`，`--controller` 增加 `intercept` / `random` / `hold`。

---

## 三、实测数据的验证程度

**已实跑并确认：**

| 项目 | 结果 |
|---|---|
| `tests/test_robot.py` | 24 项通过 |
| `tests/test_robot_assets.py` | 12 项通过 |
| `tests/test_panda_adapter.py` | 16 项通过 |
| `tests/test_robot_env.py` | 16 项通过（含 `check_env`） |
| `tests/test_benchmark_runner.py` | 5 项通过（runner 改动后回归） |
| `tests/test_docs.py` | 4 项通过（文档链接） |
| 可达性标定脚本 | 全量跑通，报告已写入 `reports/` |
| CLI `--robot panda --level L2 --split dev` | 端到端跑通，hit 100% / return 100%（2 球） |
| 脚本基线在 dev 全 12 球 | hit 9/12，valid_return 4/12 |

**2026-08-30 补跑：**

| 项目 | 结果 |
|---|---|
| 完整基线表（3 条基线 × dev+test × L0–L5，36 次运行） | 已生成 [`reports/table-tennis-panda-baselines.md`](../reports/table-tennis-panda-baselines.md)，由 `make baselines` 复现 |
| 全量 `pytest` | 224 passed / 1 skipped |
| `make lint`（含 `scripts/`） | 0 问题（原 51 条） |
| 可达性标定脚本重跑 | 复现冻结值：最短拦截窗口 0.532 s、最坏行程 0.525 s |

基线表读出的曲线与标定结论一致：`intercept` 在 L1 命中率 100%（dev/test 均通过 90% 门槛），L2 起
valid_return_rate 塌到 50%（test），L3 之后 target_rate 为 0。拦截是行程时间问题、回球是拍面速度
问题，两者以不同方式失败，这在报告里是分开的两列。

---

## 四、明确没有做的工作

### 4.1 本批次内未完成

以下六项在 2026-08-30 补齐，只剩最后一项：

| 项目 | 说明 |
|---|---|
| ~~完整基线表~~ | 已生成，见上 |
| ~~完整测试套件回归~~ | 224 passed / 1 skipped；顺带修掉 `test_task_registry` 的过时假设 |
| ~~`ruff` 检查~~ | `make lint` 清零；`TRY004` 在 `pyproject.toml` 显式忽略并说明理由 |
| ~~README 更新~~ | 中英文均新增机器人任务小节，含自训练策略的评测入口 |
| ~~Makefile~~ | 新增 `benchmark-robot` / `baselines` / `calibrate`，`lint` 覆盖 `scripts/` |
| **提交** | 所有改动仍在工作区，未 commit |

### 4.2 原计划就不在本批次

| 项目 | TODO 编号 | 影响 |
|---|---|---|
| ~~传感器层~~ | #9 | 2026-08-30 完成：`sensors.py` + `backends/mujoco_sensors.py` + `vision.py`，Vision track 已开放，见 [`VISION_TRACK.md`](VISION_TRACK.md) |
| **双足机器人资产** | #10b | 足球任务需要；机械臂路径已跑通可复用 |
| **Isaac 侧适配器** | M1 #4 | Isaac Lab 自带 Panda 资产，但本适配器只有 MuJoCo 实现；Isaac 环境本身仍缺 GPU 实跑验证 |
| ~~力矩控制模式~~ | #7 | 已完成：Panda affine servo 可逆切换为 unit-gain torque actuator，按数据手册限幅 |

### 4.3 更大的、仍然敞开的缺口

这些在起点盘点里就列出，本批次没有触及：

| 缺口 | 现状 | 差距 |
|---|---|---|
| ~~固定集规模~~ | 2026-08-30 新增 `table_tennis/return-v1`（train 1200 / dev 300 / test 600，每级 100）与 `tennis/return-v0`；v0 保持冻结 | 每 **bucket** 仍只有 14–68 条，bucket 级结论仍不可发表 |
| ~~L5 的随机化~~ | 2026-08-30 `perturbations.py` 实现三项并在 v1 manifest 声明 | v0 仍是零扰动（有意保持），扰动与分布变化的贡献尚未解耦 |
| ~~剩余指标~~ | 2026-08-30 补齐 `contact_error`、`robustness_gap`、`inference_latency_ms`、`mean_episode_time_s` | `BENCHMARK_SPEC` §7 八项已全部落地 |
| **学习基线** | 无 | 训练脚本、权重、视频属于 M5 |
| ~~提交链路~~ | `scripts/package_submission.py`（`make submission`）产出 manifest/config/metrics/policy/videos/environment | leaderboard 审计流程仍未定 |
| **物理保真** | 已补拍面胶皮接触 | 飞行段轨迹和击球冲量仍未对真实数据标定；`reports/` 只有回弹保真 |

---

## 五、下一步建议顺序

1. ~~跑通完整基线表~~、~~全量回归与 lint~~、~~README 与 Makefile~~ —— 2026-08-30 完成。
2. 提交本批次。
3. ~~传感器层（#9）~~ —— 2026-08-30 完成，Vision track 已开放。
4. ~~扩固定集规模与 L5 随机化~~ —— 2026-08-30 完成，见 [`SHOT_BANKS.md`](SHOT_BANKS.md)。
5. ~~剩余指标~~、~~`submission/` 打包链路~~ —— 2026-08-30 完成，见 [`SUBMISSION.md`](SUBMISSION.md)。
6. M3 的四项新运动（先把网球端到端做完当模板）。

---

## 六、一句话结论

**乒乓球现在是一个真实的机器人测试基准了**：动作进关节、执行器有限位、安全违规会终止 episode 并计入分母、
能耗被积分、资产带许可证溯源、每个常量都有标定来源。它测出的第一个结论也是有价值的——一台定基座 Franka
Panda 能拦到这份固定集里的每一球（余量 7 ms），但只能把其中少数转成合法回球，因为它的拍面速度上限
（1.34–3.50 m/s）低于回球所需（3.36–6.33 m/s）。**这是这个机体的真实能力上限，不是缺陷。**

**2026-08-30 补充：它现在也是一个视觉基准了。** 同一个任务多了一条 vision track——策略拿到的是两路
320×240 图像和本体感知，观测对象上根本没有 `ball` 字段。参考感知管线在 dev 全集上把球定位到 0.81 cm
中位误差、8.1% 丢帧，用完全相同的挥拍控制律在 L1 打满命中率，到 L2 掉到 50%。**这个差值就是感知的
代价，而且是测出来的。**

它还不是可提交的公开基准：缺统计充分的固定集、缺观测噪声与延迟、缺学习基线、缺审计流程。
