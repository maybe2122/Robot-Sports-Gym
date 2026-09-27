# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) 的结构，并计划从首次公开发布开始采用语义化版本。

## [Unreleased]

### Added

- **羽毛球发球 `badminton-serve-v0`**（M3 第三项运动，第一个发射类任务）：`rules/launch.py` 的
  `LaunchJudge` 是发射类任务（发球/射门/投篮）的共用生命周期；`BadmintonServeJudge` 按 BWF 第 9 条判
  1.15 m 击球高度（新失败原因 `fault`）、对角单打发球区与擦网好球。固定集 `badminton/serve-v0`
  由新的 `scripts/generate_launch_banks.py` 生成（每级 train 200 / dev 50 / test 100）；
  `MultiSportRobot/BadmintonServe-v0` 通过 `check_env`；CLI `--sport badminton`；
  `scripts/run_fixture_baselines.py --sport badminton` 产出 noop+scripted 基线。见 `docs/BADMINTON.md`。
- **Isaac Lab 环境首次实跑**与 `scripts/backend_parity.py`（`make parity`）：同一批球在 MuJoCo 与
  Isaac Lab 上逐步对比，报告 `reports/table-tennis-backend-parity.md`。

### Fixed

- Isaac Lab 环境的气动力每个物理步被施加两次（阻力翻倍，0.3 s 轨迹差 32 cm）；环境加载固定集时忽略
  `task.bank_resource`。
- 羽毛球气动力改在压心处按当前状态计算：原实现没有俯仰阻尼、且读的是滞后一步的状态，翻滚的羽毛球会数值发散。
- mocap 夹具路径从未施加 manifest 声明的 L5 扰动；现在与机体路径一致。网球 `scripted` 的 L5 因此从
  32% 变为 0%（L1–L4 不变）。
- `robustness_gap` 的分布内分子不再包含 L1（L1 击球即结束，回球率恒为 0）；已发布报告按逐级明细重算。
- IK 求解在够不到的目标上跑满迭代预算：加停滞退出，拦截基线控制延迟均值 0.94→0.35 ms。
- `StrikeZone` 声明值更新为 `return-v1` 实测穿越范围。

### Added

- **自由站立人形任务 `table-tennis-return-g1-standing-v2`**：保留 G1 骨盆自由关节，站在 0.1 m 平台上，
  四个踝关节由理想骨盆 IMU 反馈局部维持平衡，腰部 + 右臂 10 个关节由策略控制；无机身外力、无 mocap 约束，
  跌倒显式判负。环境 `TableTennisReturn-G1-Standing-v2` 与 `-Vision-v2`，CLI `--robot g1-standing`。
- **G1 视觉轨道**：`TableTennisReturn-G1-Vision-v1`，支持双目 RGB 与单相机对齐 RGB-D；新增纯深度感知
  （`depth_vision.py`，背景深度 + 小球几何前景，RGB 对策略屏蔽）。
- **严格 JSON 运行配置** `configs/*.json`（`--config`），字段对应 CLI 参数、命令行覆盖配置、未知字段报错；
  `--viewer` 实时窗口与 `--video` 录像（`visualization.py`）。见 [`docs/HUMANOID_PLAY.md`](docs/HUMANOID_PLAY.md)。

- **第二个机体 `table-tennis-return-g1-v1`**：固定基座 Unitree G1（BSD-3-Clause，同样不 vendoring），
  腰部 3 + 右臂 7 共 10 个受控关节，其余 19 个关节带真实质量与碰撞几何、保持在资产的 stand 位姿。
  同一个 Judge、同一份固定集、同一批 L0–L5 门槛、同一个报告 schema。环境
  `MultiSportRobot/TableTennisReturn-G1-v1` 通过 `check_env`，动作 10 维、**观测 39 维**——
  两个数字都不写在任何地方，都是 observation layout 算出来的宽度。CLI `--robot g1`、
  `scripts/run_baselines.py --robot g1`、`make baselines-g1`。见 [`docs/G1.md`](docs/G1.md)。
- 站位、ready pose、肩关节求解边界全部由 `calibrate_reachability.py --robot g1` 在**任务实际打分的
  固定集**上测出：1600 组（站位 × 击球平面 × 肩上界）扫描，胜出 soles `(-2.00, -0.05, 0.10)`、
  击球平面 `x = -1.55`（与 Panda 同一平面，是搜出来的不是设定的）。
  **最坏关节行程 0.067 s 对最短拦截窗口 0.558 s——8 倍余量，而 Panda 是 7 ms。
  Panda 卡在速度上，G1 卡在够不够得着。**
- 后端、环境、基线控制器、基线扫描脚本、标定脚本全部参数化到"机体"这一层
  （`MujocoEmbodiedTableTennisBackend`、`EmbodiedTableTennisReturnTaskConfig`、`SwingRobot`、
  `RobotProfile`）。接第二台机器人没有新增一行控制律，也没有复制一个环境。
- - **弹道预测器** `multisport_sim.benchmark.ball_flight`：`BallFlightModel` 用与后端相同的力律
  （二次阻力 + 有界 Magnus）积分自由飞行，`TableBounce` 把台面接触作为可积分的冲量律接进来
  （恢复系数与切向/自旋耦合系数由 train split 上的 595 次实测反弹拟合，切向与自旋 R²=0.999）。
  自旋是可选输入：state track 有，vision track 没有，少掉的那部分误差如实体现而不是假设一个自旋。
  积分核用标量而非三元 numpy 数组：数组版一次预测 23 ms，超出 200 Hz 的 5 ms 控制周期；标量版 0.6 ms。
- **观测契约** `multisport_sim.benchmark.observation`：`ObservationLayout` 把观测发布成有名字、
  有尺寸、有边界的字段序列而不是一个匿名宽度。任务定字段语义，机体定字段尺寸，扁平向量和
  Gymnasium space 由 layout 派生。`layout.view(("ball.position", ...))` 让策略按名字取字段，
  同一份不含 `robot.joint_*` 的策略换机体不用改一行；要一个该机体没有的字段当场报错而不是补零。
  layout 进每份报告的 `rule_geometry.observation_layout`，分数因此可审计。见
  [`docs/POLICY_INTERFACE.md`](docs/POLICY_INTERFACE.md)。
- `ShotBank.available_splits()`：拟合任何东西之前先问固定集有没有 train split。
- 标定脚本新增 `--bank` / `--search`，报告新增 `reachability.per_shot`（逐球可达与来得及的计数）。
- - 后端无关机器人层 `multisport_sim.benchmark.robot`：`RobotAdapter` 协议、`ControlMode` 与命令类型、
  `JointLimits`、`WorkspaceBox`、`SafetyLimits`（位置/速度/力矩/碰撞/工作空间五类违规）与 `SafetyMonitor`。
- 第三方资产解析与许可证清单 `multisport_sim.benchmark.assets`：Franka Panda（Apache-2.0）不 vendoring，
  通过 `MULTISPORT_MENAGERIE_PATH` / `MUJOCO_MENAGERIE_PATH` / `~/mujoco_menagerie` 定位，缺资产时测试 skip。
- 新任务 `table-tennis-return-panda-v1` 与环境 `MultiSportRobot/TableTennisReturn-Panda-v1`：动作是 Panda
  的 7 维关节位置设定值，观测 33 维，通过 Gymnasium `check_env`。与 `table-tennis-return-v0` 共享 Judge、
  规则和 reward 权重；Panda 默认使用统计充分的 `table_tennis/return-v1`，v0 夹具仍冻结用于回归。
- Panda 适配器与场景装配：球拍作为运动链的一部分挂在手臂法兰上，带标定的胶皮接触对（恢复系数 0.82–0.86）。
- Panda `joint_torque` 控制：Menagerie affine servo 可逆切换为 unit-gain、zero-bias torque actuator，
  命令按 87/12 N·m 数据手册限制裁剪；切回其他模式或 reset 时完整恢复位置伺服参数。
- 可达性标定脚本 `scripts/calibrate_reachability.py` 与报告 `reports/table-tennis-panda-reachability.json`。
- 三条参考基线：`hold-pose-v1`、`random-joint-v1`、`scripted-panda-intercept-v1`。均非可提交成绩。
- **第二项运动 `tennis-return-v0`**：复用同一个 `NetReturnJudge`、同一套 L0–L5 门槛与报告 schema，
  只换几何与量纲（23.77 × 8.23 m 单打场地、地面即落点、91.4 cm 球网、3 s episode、17–34 m/s 来球）。
  含 `MultiSportRobot/TennisReturn-v0`（通过 `check_env`）、按运动参数化的 benchmark 器材夹具与
  线床接触标定（e≈0.79；不标定的话继承默认阻尼只有 0.20）、noop/scripted 基线与
  [`reports/tennis-baselines.md`](reports/tennis-baselines.md)。见 [`docs/TENNIS.md`](docs/TENNIS.md)。
- **统计充分的固定集** `scripts/generate_shot_bank.py`（`make shot-bank`）：候选球在真实场景里发射，
  只有 Judge 判 `incoming_valid` 才保留；`short`/`deep`/`edge` 由**实测第一落点**决定；L4/L5 按
  pass_bucket 分层并要求测得的标签命中该 bucket，覆盖是构造出来的。产出
  `table_tennis/return-v1` 与 `tennis/return-v0`：**train 1200 / dev 300 / test 600（每级 100）**，
  三个 split 的种子和球本身都互不重叠。`return-v0` 一个字节未改，历史分数全部仍然有效。
- Shot Bank manifest 可声明 `shot_origin` 与 `initial_motion`，严格支持 robot-side 静止/向前发球与既有
  opponent-to-robot 来球；旧 manifest 缺省语义保持不变，为 BadmintonServe 等任务解除方向硬编码。
- **L5 扰动** `multisport_sim.benchmark.perturbations`：观测噪声、观测/动作延迟、域随机化
  （球质量、空气密度、相机安装位置），全部由 manifest 声明、由 episode 种子决定、可精确重放，
  且**只扰动策略看到的东西，不扰动 Judge 看到的东西**。v0 的
  `"not_implemented_in_v0"` 被解释为零扰动，因此 v0 依然完全不受影响。见
  [`docs/SHOT_BANKS.md`](docs/SHOT_BANKS.md)。
- CLI `--bank`（选择固定集）、`--split train`、`--sport tennis`；`eval_policy.py` 与
  `package_submission.py` 同样支持 `--bank`。
- Makefile 目标 `tennis`、`tennis-baselines`、`shot-bank`。
- **后端无关传感器层** `multisport_sim.benchmark.sensors`：`SensorSuite` 协议，相机 / IMU / 接触 /
  关节力矩 / frame transform 五类 spec 与读数类型，`SensorSchedule` 在一处统一执行采样率（120 Hz 相机
  在 200 Hz 控制循环下返回上一帧和上一帧的时间戳），`look_at_quaternion` 让相机按"看向哪里"声明。
- MuJoCo 参考实现 `benchmark/backends/mujoco_sensors.py`：离屏渲染（可选深度）、真加速度 IMU、语义
  类别接触力、关节力矩、任意 body/site 之间的 frame transform；所有名字在构造时解析，拼错不会拖到
  episode 中间才报错。
- **Vision track** `multisport_sim.benchmark.vision`：声明式立体相机套件
  （320×240 / 120 Hz，基线 4.2 m）、`VisionTrackBackend`（返回的观测对象上没有 `ball` 字段，规则由
  构造保证）、参考感知管线 `StereoBallTracker`（dev 全集 540 帧实测位置误差中位数 0.81 cm、p90
  1.02 cm、丢帧 8.1%）、基线控制器 `vision-panda-intercept-v1`。见
  [`docs/VISION_TRACK.md`](docs/VISION_TRACK.md)。
- CLI `--track {state,vision}`，`--controller vision`；`scripts/eval_policy.py --track vision`。
- Gymnasium 环境 `MultiSportRobot/TableTennisReturn-Panda-Vision-v1`：字典观测（两路 uint8 图像 +
  本体感知 + `frame_age_s`），通过 `check_env`。训练与评测共用 `vision.vision_observation_dict()`。
- 完整基线表 [`reports/table-tennis-panda-baselines.md`](reports/table-tennis-panda-baselines.md)（四条基线 ×
  dev/test × L0–L5，含 vision track），由 `scripts/run_baselines.py`（`make baselines`）复现。
- `scripts/eval_policy.py`：把任意 `obs(33) -> action(7)` 策略接进与基线相同的打分链路，逐难度输出报告与
  跨难度汇总表。CLI 本身只跑内置基线，这是接自训练策略的入口。
- `envs.panda_observation_vector()`：观测向量的唯一定义，Gymnasium 环境与离线打分路径共用，避免训练与
  评测读到不同的排列。
- `registry.tasks_for_sport()`：列出一个运动下的全部任务。一个运动有多个机体后，`task_for_sport()` 会明确
  报错而不是替调用方猜一个。
- **补齐 `BENCHMARK_SPEC` §7 的剩余原始指标**：`contact_error`（首次球—拍接触点到拍面中心的**面内**
  距离，外加接触瞬间拍速；没击球的回合报 `null` 而不是 0）、`robustness_gap`（L1–L3 的
  `valid_return_rate` 减 L4–L5，单 level 运行报 `null`）、`inference_latency_ms`（只对 `controller.act()`
  计时，mean/p50/p95/max，并注明是本机数字）、`mean_episode_time_s`。
- **结果包** `scripts/package_submission.py`（`make submission`）：manifest（含 commit 与**脏树标记**、
  硬件、轨道、复现命令）、config、metrics（每回合原始结果）、policy（sha256 内容哈希，没带权重会明说）、
  videos（按 shot_id 排序取前 N 个成功**和**失败）、environment.txt。见
  [`docs/SUBMISSION.md`](docs/SUBMISSION.md)。
- `multisport_sim.benchmark.policy_eval`：把策略评测从脚本提到包里，`scripts/eval_policy.py` 变成薄壳，
  提交打包与评测因此走同一条路径。
- `EpisodeResult.from_dict()`：报告可以被工具读回并重新聚合，且读回时仍然校验全部不变量。
- 可选依赖 `submission`（Pillow）：只有录像导出需要它，缺它时结果包照常生成并在 manifest 里注明。
- Makefile 目标 `benchmark-robot`、`baselines`、`calibrate`、`submission`；`make lint` 覆盖 `scripts/`。
- `scripts/check_baseline_reports.py` 与 `make check-reports`：CI 校验提交的基线表矩阵、固定集 digest、
  每级样本量、primary Wilson 区间和 L1–L5 breakdown；CI 另 smoke 运行乒乓与网球 benchmark CLI。
- `TaskEntry.vision_env_entry_point` 与 `config.vision_env_id`：一个任务可以发布两条轨道的环境，而不是
  被拆成两个任务——同一个 Judge、同一份固定集、同一批阈值，结果属于同一张表。
- 报告新增 `robot_metrics` 字段（`safety_violations`、`energy_joule`），补齐 `BENCHMARK_SPEC` §7 的两项指标。
- CLI 新增 `--robot {none,panda}`，`--controller` 增加 `intercept` / `random` / `hold`。
- 文档 [`docs/ROBOT_LAYER.md`](docs/ROBOT_LAYER.md)。
- 开源协作、治理、安全、引用和 benchmark 规范文档。
- GitHub issue、Pull Request 模板和 MuJoCo CI。
- README 增加 MuJoCo 与 Isaac Sim 五项单项场景的真实渲染对照图。
- 实验性 `table-tennis-return-v0` Shot Skill：固定 Shot Bank、MuJoCo 真接触后端、规则 Judge、分桶指标、报告 CLI 与脚本球拍测试夹具。
- 后端共享任务配置 `TableTennisReturnTaskConfig`：坐标约定、`TaskFrame` 平移、球台几何、control rate、动作/观测边界与 reward 权重集中定义；MuJoCo 环境、Runner 和 CLI 全部由它派生，报告新增 `task_config` 字段。
- 实验性 Isaac Lab `ManagerBasedRLEnv` 向量化环境，复用同一 Shot Bank、Judge 与 `EpisodeResult` schema；仓库 CI 无 Isaac 运行时，相关测试在缺少 `isaaclab` 时自动 skip，尚未实跑验证。
- 多运动任务框架：sport-agnostic 的 `ShotTaskConfig` 基类、`ShotJudge` 协议与 `RectangularSurface`、可复用的 `NetReturnJudge` 网类回球引擎、按 sport 参数化的 MuJoCo `MujocoSportProfile`，以及 `TaskEntry` 任务注册表（Gymnasium 环境注册改由注册表驱动）。
- 规则尺寸壁球场景、壁球参数与事件驱动的单回合判分；Isaac PhysX 完整得分演示会真实观察前墙和落地事件，并可复现生成带校验的 GIF 与 JSON 时间线；另提供不依赖 Isaac/Pillow 的独立资产校验 CLI。

### Changed

- `TableTennisReturnPandaTaskConfig.OBSERVATION_DIM` / `ACTION_DIM` 由 layout 与 `joint_action.dof`
  派生，不再是写死的 33 与 7。Panda 上的数值和扁平边界逐位不变，已发布的分数含义不受影响。
- `envs.panda_observation_vector()` 保留原名，实现改为 `robot_observation_vector(state, config)`
  的薄壳，后者按任务的 layout 打包，因此对关节数不同的机体同样正确。
- `run_shots` 与 `RunConfig` 不再绑定乒乓球：任务配置放宽到 `ShotTaskConfig`，Judge 由注册表按
  `task_id` 解析。加一项运动因此是加一条注册表条目，而不是在运行器里加一个分支。
- Gymnasium mocap 环境按类属性参数化（`config_type` / `default_config` / `judge_type`），网球环境
  是它的子类而不是它的副本。
- benchmark 器材夹具由 `scene.BENCHMARK_EFFECTORS` 表驱动，新增一项运动只需加一行。
- `run_shots` 对带机器人的后端逐物理步检查安全违规并积分执行器机械功；对 mocap 夹具这两项返回空数组而非零。
- 报告的 `task` 字段现在是产生该结果的任务 id，而不是共享 Shot Bank 的名字；bank 名字移到 `shot_bank_task`。
- campus 场景各单项的地面偏移改为 `specs.CAMPUS_OFFSETS` 单一定义，MuJoCo 与 Isaac 场景构造共用。
- 项目展示名称由 MultiSport Physics Sim 更名为 Robot Sports Gym，突出面向多种机器人形态的球类运动训练与能力评测宗旨；现有 distribution、Python 包、CLI 和版本化 benchmark ID 保持兼容，当前版本仍处于 physics foundation 阶段。

### Fixed

- **`applied_torque` 报的是执行器输出而不是关节实际受到的力矩。** G1 的资产把力矩上限声明在
  *关节* 上（`actuatorfrcrange`），MuJoCo 在 `qfrc_actuator` 上执行钳位而 `actuator_force` 不钳位。
  于是安全监视器看到手腕 250 N·m 对着 5 N·m 的限值，**每个 episode 都因为一次物理上从未发生的
  安全违规而失败**。两个适配器都改读 `qfrc_actuator`；Panda 上两者逐位相同（已验证），
  所以这个 bug 在只有一台机器人时既无害又不可见。
- **`VisionInterceptController.controller_id` 被遮蔽。** 父类现在在 `__init__` 里按机体设置实例属性，
  子类的同名类属性因此失效，视觉轨道的结果会被贴上 `scripted-panda-intercept-v1`。
  改为在 `super().__init__()` 之后按机体名生成，并加了钉住它的测试。
- - **脚本拦截基线在 `return-v1` 上跨不过台面反弹。** 控制器用纯弹道（等速 + 抛物线）预测击球点，
  而挥拍要在球到达前约 160 ms 承诺。L1 的球在承诺前 338 ms 就已经弹过（0/60 落在窗口内），所以
  L1 没事；L2–L5 有 55–85% 的球在承诺**之后**才落台，预测因此差 33–53 cm（中位），球拍停在离球
  8–11 cm 处——球拍半径只有 7.5 cm。改用 `BallFlightModel` + `TableBounce` 后，同一位置的预测误差
  降到 0.3–1.2 cm（p90 约 3 cm）。test split 上 `intercept` 的命中率 L1 63%→**100%**（首次通过
  90% 门槛）、L2 55%→94%、L3 53%→99%；回球率 L2 2%→16%、L4 12%→28%。`vision` 同样受益
  （L1 46%→89%）。挥拍参数经 train split 上 36 组网格复查，最优组与现有默认值只差 1/80，属噪声，
  因此**默认值不变**——收益全部来自预测本身。
- **`max_horizon_s` 短于固定集。** 默认 1.2 s，而 `return-v1` 最慢的球要 1.577 s 才到击球平面，
  这些球在手臂开始规划前一直停着。改为 1.8 s。
- **可达性标定跑错了固定集。** `calibrate_reachability.py` 调用 `ShotBank.from_resource(split=...)`
  用的是默认 task，也就是冻结的 `return-v0`（36 球），而任务已经在 `return-v1` 上打分。所有机器人侧
  常量因此拟合在另一个分布上：实际需要的击球区是 y∈[-0.71,0.70]、z∈[0.71,1.41]，而声明的
  `StrikeZone` 还是 v0 的 y∈[-0.45,0.40]、z∈[0.88,1.36]。现在 `--bank` 默认取任务自己的
  `bank_resource`，`--splits` 默认取 train（有 train 才用；绝不默认落到 dev/test 上拟合）。
- **可达性判据本身会误导。** 网格判据（矩形外接框的 49 个角点 + 全局最短到达时间）说冻结底座
  11/49 不可达、最坏行程 2.20 s 对 0.543 s 窗口；逐球判据说它够到 1197 球里的 1196 球、其中 1182 球
  来得及。搜了 78 组底座 × ready pose，没有一组比冻结值更好，**底座与 ready pose 因此不变**。
  两个数都进报告，`verdict.notes` 说明该读哪个。
- **提交侧脚本的默认固定集与任务不一致。** `scripts/eval_policy.py` 与
  `scripts/package_submission.py` 的 `--bank` 默认写死 `table_tennis/return-v0`，而
  `policy_eval` 库和 `benchmark_cli` 都默认取任务自己的 `bank_resource`。提交者按默认跑出来的
  是冻结夹具上每级 2–4 球的分数，和发布的基线表不可比。两处默认改为任务的 `bank_resource`。
- - 删除 `robot_controllers.py` 中被后定义完整遮蔽的两份控制器副本（约 160 行死代码，ruff F811）。
  运行时行为不变，前一份的 `act()` 是残缺的。
- `tests/test_task_registry.py` 不再假设一个运动只对应一个任务；乒乓球现在有两个机体任务。
- `make lint` 从 51 条问题降到 0（导入顺序、`typing` 弃用导入、集合内隐式字符串拼接、嵌套 if、TypeVar
  变体命名）。`TRY004` 在 `pyproject.toml` 中显式忽略并说明理由：本包统一用 `ValueError` 表示输入非法，
  测试也钉住了这一契约。

## [0.2.0] - 2026-08-08

### Added

- 网球、乒乓球、足球、羽毛球和篮球的程序化场景。
- MuJoCo 与 Isaac Sim/PhysX 双后端。
- 球体接触、空气阻力、Magnus 力和羽毛球稳定力矩。
- 规则落球测试和 `multisport-fidelity-v1` 量化报告。
- USD 导出与无头运行。

### Changed

- 补齐乒乓球桌网格及篮球篮板、篮圈和篮网细节。
- 校准 Isaac PhysX 乒乓球与羽毛球恢复系数。
