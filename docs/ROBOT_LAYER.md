# 机器人与传感器层

本文件描述 M2 的机器人层：后端无关的适配器契约、控制模式、安全限制，以及第一个真实机器人实现
`table-tennis-return-panda-v1`。

设计稿见 [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) 第 6 节，任务规则见
[`TABLE_TENNIS_SHOT_SKILL.md`](TABLE_TENNIS_SHOT_SKILL.md)。本批次（2026-08-24）实际完成到哪一步、
哪些验证跑过、哪些明确没做，见 [`M2_STATUS.md`](M2_STATUS.md)。

## 为什么需要这一层

`table-tennis-return-v0` 的动作是 mocap 拍面 pose，MuJoCo 直接把拍面瞬移过去。这条链路验证了发球、
接触事件、规则 Judge 和报告 schema，但它测不出任何机器人性质：没有关节就无从定义 `safety_violations`，
没有执行器就无从积分 `energy_joule`，没有动力学就无从训练可迁移的策略。

`table-tennis-return-panda-v1` 是同一个任务的真实机体版本。**两者是并列的版本化任务，不是新旧关系**：
它们共享同一个 Judge、同一份固定 Shot Bank、同一组 reward 权重，唯一的差别是谁在击球。v0 保留不动，
继续作为规则引擎的回归夹具。

## 后端无关契约

`multisport_sim.benchmark.robot` 不导入任何模拟器。任务代码只通过这些类型接触机器人。

### `RobotAdapter`

| 成员 | 含义 |
|---|---|
| `robot_id` | 写进每份报告的稳定标识 |
| `joint_names` | **有序**的受控关节；任务按下标访问，永远不按名字查找 |
| `dof` | 关节数 |
| `effector_name` | 末端的语义角色（`racket` / `foot` / …），不是几何体名 |
| `control_modes` | 本适配器实现的模式集合；其余必须报错 |
| `safety_limits` | 关节、速度、力矩、碰撞、工作空间包络 |
| `reset()` / `observe()` / `apply(command)` | 生命周期 |
| `describe()` | 报告用的身份、资产溯源与限制快照 |

任务不得按关节名查找——同一个任务必须能换一条手臂跑而不改代码，这是这层存在的全部理由。

### 控制模式

`ControlMode` 的字符串值是稳定的，会进报告和提交 manifest：一份结果只能和**同一模式**下的结果比较。

| 模式 | 命令类型 | 单位 |
|---|---|---|
| `joint_position` | `JointCommand` | rad |
| `joint_velocity` | `JointCommand` | rad/s |
| `joint_torque` | `JointCommand` | N·m |
| `effector_pose` | `EffectorPoseCommand` | m + `wxyz` 单位四元数 |

适配器收到未实现的模式必须抛 `UnsupportedControlMode`，而不是退化成别的模式静默执行——后者会让一份
结果说不清它到底测了什么。

接受 `effector_pose` 的适配器**拥有**它的 IK 或阻抗控制器，并且必须在 `describe()` 里声明：求解器成为
了这份提交所测量内容的一部分。

Panda 的 `joint_torque` 不是把 N·m 数字写进位置设定值。Menagerie 资产使用 affine `general` actuator；
adapter 进入力矩模式时将其显式改为 unit gain、zero bias，并把 ctrl range 改成数据手册的 87/12 N·m。
切回 position、velocity 或 effector pose 时恢复构造时捕获的 gain、bias 和 ctrl range；`reset()` 也强制
恢复位置伺服。力矩模式不隐式做重力补偿，补偿属于控制器本身，因而报告中的 action 仍有明确物理含义。

### 安全限制

`SafetyLimits.violations(observation)` 覆盖 `BENCHMARK_SPEC` §7 要求的四类，返回 `SafetyViolation`
列表：

| `kind` | 触发条件 |
|---|---|
| `joint_position` | 关节超出资产声明的范围 |
| `joint_velocity` | 关节速度超过**数据手册**限速（MJCF 里没有这个数，由 benchmark 提供） |
| `joint_torque` | 实际施加力矩超过限值 |
| `collision` | 机器人碰到 `forbidden_contacts` 里的语义标签（默认 `table` / `net` / `floor`） |
| `workspace` | 末端离开声明的作业单元 |

一次观测里**所有**违规都会返回，不是第一条：同时打破三条限制的策略不该被记成打破一条。

容差（`position_tolerance_rad` 等）是声明的、版本化的数字，不是私下的宽容值。它们存在是因为模拟器要
用几步才解算完一个约束：关节正贴着限位、或接触瞬间力矩超调千分之几，都是正常执行器行为，不该记成安全失败。

违规接到 `failure_reason="safety"` 终止路径。**安全失败计入分母，和 miss 一样。**

## 资产：不 vendoring

`multisport_sim.benchmark.assets` 声明每个第三方资产的上游 URL、版本、许可证和本地修改，但**不把资产
复制进仓库**。复制等于静默 fork：副本收不到上游修复，许可证文本会漂移，而且一份结果再也说不清它用的是
哪个版本的模型。

查找顺序（先命中者胜）：

1. `MULTISPORT_MENAGERIE_PATH` 环境变量；
2. `MUJOCO_MENAGERIE_PATH` 环境变量（不少 MuJoCo 项目已经设了）；
3. `~/mujoco_menagerie`。

获取方式：

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
git -C ~/mujoco_menagerie sparse-checkout add franka_emika_panda
# 若装在别处
export MULTISPORT_MENAGERIE_PATH=/path/to/mujoco_menagerie
```

资产缺失时，`require_asset()` 抛出带上述命令的 `AssetUnavailableError`，相关测试自动 skip 而不是 fail。

### 许可证清单

| 资产 | 用途 | 上游 | 许可证 |
|---|---|---|---|
| `franka_emika_panda` (`panda_nohand.xml`) | 乒乓球任务的 7 DoF 机械臂 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) | Apache-2.0 |

本地修改（不改上游文件，全部记录在 `AssetSource.modifications` 并进入每份报告）：

- 通过 `mujoco.MjSpec` 加载并挂进乒乓球场景，关节/执行器/几何体统一加 `rb_` 前缀；
- 在手臂自己的 `attachment` body 下挂一个 benchmark 球拍；
- 底座下加一根固定支柱几何体；
- 保留父场景的 `impratio=1` 而非模型的 `impratio=10`（一个编译模型只能有一个值）。

`license_manifest()` 返回可直接进报告的清单。

## Franka Panda 乒乓球实现

### 为什么选 Panda

MuJoCo Menagerie 与 Isaac Lab **都自带**它，且许可证清晰——这是让"同一个机器人在两个后端跑同一个任务"
这条发布门槛低成本成立的唯一选择。7 个关节在固定了拍面位置和法向之后还剩一个冗余自由度，六轴臂做不到。

它的代价也如实记录：数据手册把近端关节限速在 2.175 rad/s，这是本任务里真正的约束（见下文标定）。

### 场景装配

球拍是**运动链的一部分**，不是 mocap body：击球是手臂动力学产生的真实物理接触。拍面是沿自身 y 轴很薄的
椭球，所以击球面法向就是那个轴；拍面 site 绕 x 轴转了 −90°，让 **site 的 z 轴等于拍面法向**——瞄准球拍
因此是求解器的单轴约束，绕法向的旋转留给冗余度。

球拍带 0.17 kg 真实质量（成品拍加胶皮的量级），手臂要真的把它加速起来。

### 球拍胶皮接触对

机器人模型为球-拍面单独定义了标定接触对：

```
solref = (0.023, 0.042)    solimp = (0.96, 0.99, 0.001, 0.5, 2.0)    condim = 6
```

落球测试（`tests/test_panda_adapter.py`）测得恢复系数 **e ≈ 0.82–0.86**，落在实测反胶的区间内。

这个数是承重的。没有专门接触对时，拍面继承场景默认阻尼比 0.7，回球恢复系数只有 **0.17**——手臂等于在挥
一块海绵，任何可达的拍速都产生不了合法回球。

该接触对**只加在机器人模型上**。v0 的 mocap 夹具保持它被冻结时的接触参数，因为改动会静默移动每一个 v0
分数——两个任务在这里就不同，这是它们分开版本化的又一个理由。

## 可达性标定

`scripts/calibrate_reachability.py` 是上述每一个机器人侧常量的来源。把手臂挂上任务，不等于任务对它可解；
这个脚本用数字而不是断言来回答。

```bash
python scripts/calibrate_reachability.py --report reports/table-tennis-panda-reachability.json
```

四步：

1. **击球平面。** 无机器人动作发射全部 36 条固定来球，记录每球穿过候选平面的时刻与位置。得到的是任务**实际
   要求**的区域，而不是动作空间恰好声明的区域。
2. **底座位置。** 在关节范围内采样正运动学，按对该区域的覆盖率给候选底座打分。
3. **Ready pose。** 决定拦截成败的是行程时间而非可达范围，所以按"到击球平面任意点的最坏情况关节行程时间"
   打分——并且是对**求解器从该 pose 起步时的实际行为**打分。
4. **击球速度。** 在每个拦截位形上，计算数据手册限速内手臂能产生的最大拍面速度，与合法回球所需的球速对比。

第 4 步是可能判定"任务-机器人配对不成立"的一步，无论结论好看与否都照报。

### 冻结结果

以下数字来自 `reports/table-tennis-panda-reachability.json`：

| 量 | 值 |
|---|---|
| 击球平面 | `x = -1.55` m，36/36 条来球全部穿过 |
| 到达时刻 | 0.532 – 1.133 s |
| 横向跨度 | `y ∈ [-0.386, +0.319]` m |
| 高度跨度 | `z ∈ [0.923, 1.310]` m |
| 底座 | `(-1.95, 0, 0.75)`，击球平面覆盖率 97.5% |
| 最坏情况关节行程时间 | **0.525 s** |
| 最短拦截窗口 | **0.532 s** |
| 拍面前向最大速度 | 1.34 – 3.50 m/s |
| 合法回球所需球速（吊高） | 3.36 – 6.33 m/s |

**结论：Panda 能拦到每一球，余量只有 7 ms。** 拦截是行程时间问题，回球是速度问题，两者以不同方式失败，
因此分开报告。

原来 `EffectorWorkspace` 声明的横向范围是 ±1.5 m（3 m 跨度），任何定基座臂都够不着；实测只需要
`y ∈ [-0.39, +0.32]`。声明宽 4 倍的动作空间不会让任务更通用，只会让它说谎。

### Ready pose

`PANDA_READY_QPOS` 不是随便一个 home 位姿，而是最坏情况关节行程时间的最优解。这里有一个容易掉进去的坑：
冗余臂有很多位形能到达同一个拍面位姿，如果拿"求解器从该 pose 起步永远不会返回的解集"来打分，这个 pose
看起来会好上大约一倍。必须对求解器的实际行为打分。按这个口径，从 1.141 s 优化到 **0.525 s**。

### 安全包络

工作空间是**作业单元**，不是可达性估计：

```
x ∈ [-2.95, -1.05]    y ∈ [-1.00, 1.00]    z ∈ [0.30, 2.00]
```

拍面可以去手臂在球台后方物理能到的任何地方，但不能越到台面上方、也不能下到桌腿之间。收得再紧就会把正常
的手臂运动记成安全失败，那样一份提交从自己的分数里学不到任何东西。撞地和撞台由碰撞违规单独捕获。

## 环境

```python
import gymnasium as gym
import multisport_sim.benchmark  # 注册环境

env = gym.make("MultiSportRobot/TableTennisReturn-Panda-v1", split="dev")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- **动作**：7 维关节位置设定值（rad），保持一个 control step，按手臂自身限位夹紧。
  动作空间发布的就是手臂的真实关节范围，而不是归一化到 `[-1, 1]`——策略因此无法请求硬件做不到的运动，
  代价是 `check_env` 会给一条关于归一化的建议性警告。`BENCHMARK_SPEC` §5 要求声明单位和物理含义，这里
  以真实单位为准。
- **观测**：33 维 = 球位置/线速度/角速度(9) + 关节位置/速度(14) + 拍面位置/四元数/线速度(10)。
  没有任何几何体名或模拟器句柄——换一条手臂也能提供同样的量。
- **安全**：任一类违规使 `terminated=True` 且 `failure_reason="safety"`，
  `info["safety_violation_details"]` 给出具体数字。安全检查在**每个物理步**执行，不是每个 control step：
  在 decimation 中间被打破的限制照样是被打破了。

## 参考基线

三条基线随任务发布，**没有一条是可提交成绩**；它们的作用是给分数划出可读的区间。

| 控制器 | `controller_id` | 说明 |
|---|---|---|
| `hold` | `hold-pose-v1` | 全程停在 ready pose。确认 miss / timeout 失败路径在真实机体上确实会触发 |
| `random` | `random-joint-v1` | 在声明的动作空间里均匀采样，每 `hold_steps` 步重采一次 |
| `intercept` | `scripted-panda-intercept-v1` | 预测击球点、解 IK、挥拍 |

### 速率限制

两条会动的基线都用 `JointRateLimiter` 限制设定值推进速度，并且都声明了这一点。它放在**控制器**里而不是
适配器里是有意的：适配器如果对每条命令都静默限速，安全包络就变得不可证伪，也就分不出一个策略是否尊重硬件。
提交方可以不用它——然后被随之而来的安全失败计分。

### 脚本拦截器怎么工作

它使用特权球状态，和它替代的 mocap 夹具一样，所以**它的分数不是机器人策略成绩**。但它不碰正在运行的仿真：
模型和求解器都是它自己的，因此一个带相机和手臂模型的真实策略能复现它产生的每一个数字。

1. **预测**：解出球到达击球平面的弹道时间和该处高度，每个 control step 重算，让阻力和 Magnus 力的预测
   误差被持续纠正而不是一次性提交。
2. **准备**：拦截前停在击球点的后上方。关节空间插值走的是曲线不是直线，直接对着低球的拦截点插值，那条曲线
   会下沉进球台边缘——碰撞限制会正确地判为安全失败。真实球员出于同样的原因也在后上方引拍。
3. **挥击**：在 `swing_lead_s` 内把拍面向前推过球。在拦截点停下来等球，回球会掉进自己半台；只有拍速能把
   一次命中转成一次回球。

IK 用**上一次的规划**做种子，不用手臂的实测关节位置。7 个关节对 5 个约束，求解器可以在零空间里游走；用滞后
的测量值做种子会让它每一步挑一条不同的冗余分支，设定值于是在追一个不断移动的目标。

## 运行

```bash
# 机器人任务
multisport-benchmark --robot panda --level L2 --split dev --controller intercept

# 随机基线（会触发安全包络）
multisport-benchmark --robot panda --level L2 --split dev --controller random

# v0 mocap 夹具（不受影响）
multisport-benchmark --level L1 --split dev --controller scripted
```

报告新增 `robot_metrics` 字段：

```json
{
  "safety_violations": [0, 0],
  "energy_joule": [58.85, 75.44],
  "total_safety_violations": 0,
  "mean_energy_joule": 67.14
}
```

对 mocap 夹具这两项是**空数组，不是零**：它没有安全包络也没有执行器，报 0 等于宣称通过了从未执行的检查。

## 尚未完成

- **传感器层（TODO #9）**：相机、IMU、接触、关节力矩、frame transform 的后端无关 `SensorSuite` 未实现，
  因此 `BENCHMARK_SPEC` 的 Vision track 仍未开放。当前观测是特权状态。
- **Isaac 侧**：Isaac Lab 的 Panda 资产存在，但本适配器只有 MuJoCo 实现，且 Isaac 环境本身仍缺 GPU 实跑
  验证（见 [`ISAAC_SIM.md`](ISAAC_SIM.md)）。
- **双足机器人**：足球任务需要，未接入。
- **学习基线**：训练脚本与权重属于 M5。
- **固定集规模**：Shot Bank 仍是 12+24 的工程验证集，不满足 `BENCHMARK_SPEC` §8 的统计要求。
