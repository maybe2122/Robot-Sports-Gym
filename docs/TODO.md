# 待办清单

本清单是 [`ROADMAP.md`](ROADMAP.md) 里程碑的可执行拆解：路线图给发布门槛，这里给具体交付项、依赖关系和验收标准。里程碑勾选以路线图为准，本文件跟踪进行中的工作。

状态含义：`done` 已合入并有测试覆盖；`blocked` 代码就位但缺运行环境验证；`todo` 未开始。

**最后核对：2026-09-28，v0.3.0 发布前，分支 `feat/shared-task-config`。**

---

## 一、现在做到哪了

一句话：**五项标准单回合任务都能跑、都有固定集与基线（含学习基线），乒乓球还有两台机体和视觉轨道；
缺的是机体化（除乒乓球外都是 mocap 夹具）、双足资产、真实世界标定和排行榜本身。**

| 维度 | 现状 |
|---|---|
| 已注册任务 | 8 个：乒乓球 4 个（mocap 夹具 v0、Panda v1、G1 v1、站立 G1 v2）、`tennis-return-v0`、`badminton-serve-v0`、`football-kick-v0`、`basketball-shoot-v0` |
| Gymnasium 环境 | 11 个（含 3 个视觉环境），全部通过 `check_env` |
| 后端 | MuJoCo 全通；Isaac Lab 乒乓球夹具环境在 CPU 与 GPU PhysX 上实跑，与 MuJoCo 逐条对比（飞行段 3.5 mm、判定一致 99%） |
| 固定集 | 7 份：`table_tennis/return-v0`（冻结小集）、`return-v1`、`tennis/return-v0`、`badminton/serve-v0`、`football/kick-v0`、`basketball/shoot-v0`，每级 train 200 / dev 50 / test 100；全部 `experimental` |
| 基线 | 每项任务都有下限（hold/noop/random）与参考控制器；**五项任务都有 5 种子的 PPO 原语学习基线**（`docs/LEARNED_BASELINES.md`）；`reports/cross-task-summary.md` 一表汇总 |
| 策略接口 | L3 目标经 `info["target"]` / `TargetObservation` / `reset(target=...)` 交给策略 |
| 发布件 | 带版本号的 JSON Schema、结果包 + `audit_submission.py`、`THIRD_PARTY_LICENSES.md`、英文 `API.md`、`uv.lock`（3.10 与 3.13 从零复现后全部测试通过）、wheel/sdist 可构建 |
| 测试 | 530 项：529 通过、1 skip（`test_isaac_lab_env` 在无 Isaac 的环境里跳过）；Python 3.10 与 3.13 均验证 |

按里程碑：**M1 完成**（Isaac 实跑 + 共享任务配置）；**M2 差双足资产与 Isaac 侧机体 adapter**；**M3 五项都有任务与
三类基线，但只有乒乓球是机体任务**；M4 未开始；M5 有了第一批学习基线与 5 种子协议；M6 的文档/schema/许可证/
审计/锁文件已具备，DOI 与外部复现未做。

### M1 — Benchmark API

| # | 任务 | 状态 | 验收标准 |
|---|---|---|---|
| 1 | 后端共享任务配置 `TableTennisReturnTaskConfig` | done | 坐标约定、球台几何、control rate、动作/观测边界、reward 权重集中定义；MuJoCo 环境、Runner、CLI 全部由它派生 |
| 2 | world↔task 坐标帧变换 `TaskFrame` | done | 位置/速度/接触样本换算有往返测试；轴约定不一致时报错而非静默算错 |
| 3 | MuJoCo 路径消费共享配置 | done | `envs.py` 无硬编码 Box 边界；报告带 `task_config` 快照 |
| 4 | Isaac Lab `ManagerBasedRLEnv` 向量化环境 | done（CPU + GPU PhysX） | 2026-09-27 在 Isaac Sim 5.0 / Isaac Lab 0.46.2 上实跑 300 并行环境；`scripts/backend_parity.py` 与 MuJoCo 同批球逐条对比：飞行段中位差 3.5 mm，Judge 一致率 99%，接触段反弹高度差约一成（M4 待标定）。实跑修复了气动力双重施加与固定集参数漏传。GPU PhysX 同样实跑（与 CPU 结果几乎相同）。见 [`ISAAC_SIM.md`](ISAAC_SIM.md) |

### M2 — Robot and sensor layer

五项任务的机器人层。除资产接入外都不依赖具体机器人型号。

| # | 任务 | 状态 | 依赖 | 验收标准 |
|---|---|---|---|---|
| 6 | 后端无关 robot adapter 协议 | done | — | `robot.py`：`RobotAdapter` Protocol、`RobotObservation`、`JointLimits`；任务代码只按下标和语义角色访问 |
| 7 | 位置/速度/力矩与末端控制模式 | done | 6 | `ControlMode` 与 `JointCommand`/`EffectorPoseCommand`；Panda 实现 position/velocity/torque/effector_pose，力矩按数据手册 87/12 N·m 限幅，模式切换完整恢复伺服参数 |
| 8 | 速度/力矩/碰撞/工作空间安全限制 | done | 6 | `SafetyLimits.violations()` 覆盖五类（位置/速度/力矩/碰撞/工作空间）；接入 `failure_reason="safety"`，逐物理步检查，计入分母 |
| 9 | 相机/IMU/接触/关节力矩/frame transform 传感器层 | done | 6 | `sensors.py` 后端无关 `SensorSuite` 与五类传感器；`backends/mujoco_sensors.py` 参考实现；Isaac 侧只保留同名协议。Vision track 见 [`VISION_TRACK.md`](VISION_TRACK.md) |
| 9b | Vision track 端到端 | done | 9 | `vision.py`：声明式立体相机套件、`VisionTrackBackend`、参考感知管线（dev 全集位置误差中位数 0.81 cm、丢帧 8.1%）、`vision-panda-intercept-v1` 基线、`--track vision` CLI 与 `TableTennisReturn-Panda-Vision-v1` |
| 9c | 完整原始指标与结果包 | done | 9 | `contact_error`、`robustness_gap`、`inference_latency_ms`、`mean_episode_time_s` 全部进报告；`scripts/package_submission.py` 产出 `BENCHMARK_SPEC` §9 的完整结果包。见 [`SUBMISSION.md`](SUBMISSION.md) |
| 9d | 统计充分的固定集与 L5 扰动 | done | — | `scripts/generate_shot_bank.py`（`make shot-bank`）：仿真验证、实测打标签、分层保证 bucket 覆盖；每级 100 条 test、含 train split、种子不重叠；`perturbations.py` 实现 L5 的观测噪声/延迟/域随机化，v0 保持零扰动。见 [`SHOT_BANKS.md`](SHOT_BANKS.md) |
| 10 | 许可证清晰的机械臂资产接入 | done | 6, 7 | `assets.py`：Franka Panda（Apache-2.0）不 vendoring，走 `MULTISPORT_MENAGERIE_PATH` / `MUJOCO_MENAGERIE_PATH` / `~/mujoco_menagerie`，缺资产时测试 skip；许可证与本地修改清单进每份报告 |
| 10b | 双足机器人资产 | todo | 6, 7 | 足球任务需要；机械臂路径已跑通，可复用同一套 adapter 与资产解析 |
| 10c | 乒乓球机体任务端到端 | done | 6–8, 10 | `table-tennis-return-panda-v1`：关节空间环境通过 `check_env`、可达性标定报告、球拍胶皮接触标定、四条基线、报告新增 `robot_metrics` |

本批次（2026-08-24 起，2026-08-30 补齐基线表、全量回归、lint、README 与 Makefile）的完成范围、
验证程度和明确未做项见 [`M2_STATUS.md`](M2_STATUS.md)。机器人层设计与标定结果见
[`ROBOT_LAYER.md`](ROBOT_LAYER.md)。

### M3 — Five canonical tasks

| # | 任务 | 状态 | 依赖 | 验收标准 |
|---|---|---|---|---|
| 11 | 多运动任务框架泛化 | done | — | `ShotTaskConfig`、`ShotJudge`、`RectangularSurface`、`NetReturnJudge`、`MujocoSportProfile`、`TaskEntry`；Shot Bank 支持声明 opponent→robot 来球或 robot-side 静止/向前发球，旧 manifest 兼容 |
| 12 | TennisReturn | done | 11 | `tennis-return-v0`：复用 `NetReturnJudge` 与全部 L0–L5 门槛、网球 benchmark 线床夹具（e≈0.79 标定）、固定 Shot Bank、`TennisReturn-v0` 通过 `check_env`、noop+scripted 基线。见 [`TENNIS.md`](TENNIS.md) |
| 13 | FootballKickToTarget | done（mocap 夹具） | 11 | `football-kick-v0`：任务坐标原点在球门线、IFAB 进球判定（整球越线）、球门口 (y,z) 放置平面、`football/kick-v0` 固定集（滚动球从纯滚动开始）、足球 benchmark 场景用 PGS 求解器（修复 Newton 的方向相关伪影）、`FootballKick-v0` 通过 `check_env`、noop+scripted 基线。机体版仍依赖 #10b。见 [`LAUNCH_TASKS.md`](LAUNCH_TASKS.md) |
| 14 | BadmintonServe | done | 11 | `badminton-serve-v0`：发射类任务框架 `LaunchJudge` + BWF 发球规则 Judge（1.15 m 击球高度、对角单打发球区、擦网好球）、0.5 ms 步长与插值 mocap 夹具、压心来流的羽毛球气动力、`badminton/serve-v0` 固定集（每级 test 100）、`BadmintonServe-v0` 通过 `check_env`、noop+scripted 基线。见 [`BADMINTON.md`](BADMINTON.md) |
| 15 | BasketballShoot | done（mocap 夹具） | 11 | `basketball-shoot-v0`：原点在篮圈正下方、FIBA 入筐判定（从上方穿过圈内，从下方穿过判 `fault`）、篮圈平面 (x,y) 放置平面、`basketball/shoot-v0` 固定集、推板夹具与出手延迟模型、`BasketballShoot-v0` 通过 `check_env`、noop+scripted 基线。见 [`LAUNCH_TASKS.md`](LAUNCH_TASKS.md) |
| 16 | 五项任务 baseline 与统一报告 | done | 12–15 | 每项任务都有 floor 与 scripted/control 基线；`scripts/cross_task_report.py`（`make cross-task`）汇总全部任务 × 全部级别，`make baselines-all` 一条命令重跑全部输入。见 [`reports/cross-task-summary.md`](../reports/cross-task-summary.md) |

---

## 二、接下来计划做什么

| 顺序 | 动作 | 为什么 | 完成标志 |
|---:|---|---|---|
| 1 | **双足资产 + 足球机体任务**（#10b / #13 机体版） | 足球是唯一"应该用腿"的任务，现在只有 mocap 鞋面 | G1 站立任务的机体层复用到踢球；许可证清单进报告 |
| 2 | **网球/羽毛球/篮球的机体任务** | 除乒乓球外全是夹具，分数不描述机器人。**选型约束已量化（2026-09-28）**：羽毛球合法发球最少需要约 11 m/s 拍速（实测射程表，最近合法落点水平约 5.4 m），而 Panda 拍面前向速度上限只有 4.70 m/s（数据手册关节速度），最多送出约 2.8 m——**Panda 发不出合法发球**，在它上面做这个任务只会得 0%。网球来球 17–34 m/s，同理不适合 Panda。候选是 G1（报告上限 19.9 m/s，但来自未经核实的占位关节速度，先要核实） | 先核实 G1 手臂关节速度限制；若成立，在 G1 上跑通羽毛球发球 L0–L2 |
| 3 | **Isaac 侧机体 adapter 与其他运动的 Isaac 环境** | 双后端目前只覆盖乒乓球夹具 | Panda 在 Isaac 上跑同一批球，出一致性报告 |
| 4 | **M4 保真标定** | 反弹后轨迹两后端差约 5 cm、接触参数都来自夹具设定 | 至少一项运动有实测轨迹/冲量数据与拟合 |
| 5 | **闭环学习基线** | 现在的学习基线是"一次决策的原语参数"，L3 定点基本没学出来 | 至少一项任务上闭环策略超过原语策略 |
| 6 | **排行榜落地**（M6） | 审计工具有了，表没有；固定集都还是 experimental | 冻结一份 `leaderboard_eligible` 固定集 + 隐藏测试集 |
| 7 | **DOI 与外部复现** | M6 发布门槛 | GitHub Release 接入 Zenodo；至少一位外部用户按 README 复现一张基线表 |

## 三、当前已知问题

按"会不会让别人读错结论"排序。前四条是会的。

| # | 问题 | 具体表现 | 影响 | 处理方向 |
|---:|---|---|---|---|
| 1 | ~~整批工作未提交~~（已解决） | 工作区 60+ 个改动/新增文件，`feat/shared-task-config` 上最后一次提交还停在框架泛化之前 | 结果包 manifest 会打脏树标记；任何人拿到 commit 都复现不出这些报告 | 分批提交，见计划 1 |
| 2 | ~~机体基线统计量不足~~（已修复） | Panda 默认固定集已改为 `table_tennis/return-v1`，完整报告已重跑 | dev 每格 n=50 / test 每格 n=100，v1 digest 与 Wilson 区间已记录 |
| 3 | ~~难度分级不单调~~（已澄清） | 网球 scripted 的聚合 gap 曾为负：一半因为 L1 被计入分布内分子（已改，见 #15），一半因为 mocap 路径从未施加 L5 扰动（已修，L5 现在 0%，gap +31%） | 报告新增逐级 breakdown 与区间；规范明确两级点估计不承诺单调 |
| 4 | ~~Isaac Lab 从未实跑~~（已解决） | 2026-09-27 在 CPU 与 GPU PhysX 上实跑并与 MuJoCo 逐条对比，修复了两个缺陷 | 见 `ISAAC_SIM.md` | 机体 adapter 仍缺（问题 5） |
| 5 | **Isaac 侧没有 robot / sensor 实现** | `robot.py`、`sensors.py` 是后端无关协议，但只有 MuJoCo 的实现（`backends/mujoco_robot.py`、`mujoco_sensors.py`） | "同一个 Panda 跑两个后端"目前是设计意图，不是已验证事实 | 与问题 4 一起做 |
| 6 | ~~力矩控制模式未实现~~（已修复） | Panda adapter 将 affine 位置 actuator 显式切换为 unit-gain、zero-bias torque actuator，并可逆恢复 | 命令以 N·m 解释并按 87/12 N·m 限幅；reset 回到位置模式；模式往返有物理测试 |
| 7 | **bucket 级样本量仍有限（已缓解）** | v1 每级 100 条，但每 bucket 只有 14–68 条 | L4/L5 主指标噪声仍较大 | `assessment` 已逐 bucket 报样本量与 Wilson 95% 区间；发表级结果仍应扩固定集 |
| 8 | **学习基线只有原语参数级别**（部分解决） | 全部基线都是脚本控制器，且 `intercept` / `scripted` 读特权球状态 | 现在所有分数刻画的是"夹具/机体能力上限"，不是"策略能做到什么" | M5 |
| 9 | **物理保真只覆盖回弹** | `reports/` 只有回弹保真；飞行段轨迹、击球冲量未对真实测量标定。拍面 e≈0.82–0.86 只是落在文献区间内，不是拟合本项目实测数据 | sim-to-real gap 无法量化 | M4 |
| 10 | **双足资产空缺** | 未选型（G1 站立任务已有，但没有踢球的腿部控制） | 足球只有 mocap 鞋面夹具 | 计划 1 |
| 11 | **Panda 在这份固定集上的能力上限很硬**（已按 v1 重测） | 逐球标定：1197 条 train 球够到 1196 条、其中 1182 条来得及（L0–L3 全中，L4 193/197，L5 189/200）。拍面前向最大速度 1.16–4.70 m/s，合法回球（吊高）需要 3.44–6.48 m/s | 拦截已不是瓶颈——`intercept` 在 L1 命中 100%；回球才是，test 上 L2 只有 16%，L3 以上 `target_rate` 仍为 0 | **这是真实能力上限不是 bug**，但要在 spec 里写清楚。注意别读 `verdict.can_reach_every_shot_in_time`：那个判据打的是外接矩形的角点，没有球去过那里 |
| 12 | **CI 覆盖面仍窄（已缓解）** | CPU job 已 lint `src tests scripts`，校验基线报告的 bank digest/n/区间/breakdown，并 smoke 跑乒乓与网球 CLI；仍没有 GPU/Isaac job 或 nightly 物理重跑 | 报告与固定集口径漂移已会失败；物理回归漂移仍需 nightly | M5 的 CI 项 |
| 13 | **leaderboard 审计流程未定** | 结果包格式已定，怎么审没定 | M6 发布门槛 | M6 |
| 18 | **`incoming_valid_rate` 不是机体无关的** | L0 的主指标本应只描述固定集里的球合不合法，但一个在球完成第一落之前就因安全违规中止的 episode 记录不到它。G1 的 `random` 基线在 L0 拿 0%，Panda 的同一条拿 90%——差别只是违规发生的时刻 | L0 目前测的是控制器而不是固定集 | 要么让 L0 的分子不受中止影响，要么在 spec 里写明 L0 对中止敏感。改语义会移动已发布的 Panda 分数，所以先记不改 |
| 19 | **G1 上 `random` 基线没有区分能力** | 600 个回合 100% 以安全违规告终：kp=500 的人形伺服被随机指令驱动会立刻冲出声明的速度包络 | 随机基线本该给出一个可读的地板，在这个机体上给不出 | 要么给它一个机体相关的采样率/幅度，要么在报告里明说它在这个机体上只是安全包络的触发器 |
| 20 | **G1 的 `hold` 基线命中率不是 0** | ready pose 把球拍停在来球路径上，test L3 上 `hold` 命中 53% | 读 `intercept` 的命中率要减掉这个地板，否则会高估控制的贡献 | 报告已并列 `hold` 一行；spec 里应写明命中率的地板是机体相关的 |
| 15 | **`valid_return_rate` 在这个机体上随难度递增**（短期修法已做） | test `intercept`：L4 回球率高于 L2/L3。实测来球速度随难度单调上升，而出球速度 ≈ 1.85·v_拍 + 0.85·v_来球、拍速有硬上限——**来球越快越容易打回去** | 旧口径把 L1（击球即结束、回球率恒为 0）计入分布内分子，放大了负 gap | 2026-09-28：gap 分子改为只取 L2–L3，L1 仍在逐级明细中；已发布报告按 `level_breakdown` 离线重算（Panda test `intercept` −9.5%→−4.0%）。剩余的负值是真实现象；长期仍需让难度轴不只沿来球速度加码 |
| 16 | ~~`StrikeZone` 声明值仍是 v0 量纲~~（已修复） | 默认值改为 v1 train 实测穿越范围向外取整：y∈[-0.72,0.71]、z∈[0.70,1.42] | 测试改为在 dev split 真实发球取穿越点，钉住"声明区域包含实测穿越"与"Panda IK 覆盖真实穿越点"；不再对外接矩形角点求 IK |
| 17 | ~~拦截基线的最坏单步延迟仍偏高~~（已修复） | 根因不是预热：约 15% 的 IK 求解目标够不到（残差 6.5 cm），每次跑满 300 次迭代（~9 ms），而收敛的求解只要 1 次 | `IKSolver` 加停滞退出（单次关节步长 < 1e-7 rad 即返回，结果与跑满 300 次相差 < 1 µrad）；dev L2 实测均值 0.94→0.35 ms、p95 8.8→2.0 ms。Panda/G1 全表重跑：84 行主指标全部不变，4 行次要字段差 1 个回合 |
| 14 | **本机跑测试要绕开 ROS**（开发环境，非代码缺陷） | 系统 `PYTHONPATH` 带进 `/opt/ros/humble` 的 py3.10 site-packages，`launch_testing` 插件在 3.13 venv 里 import 失败 | 直接 `pytest` 会崩在收集阶段 | 用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`，或在 shell 里清掉 `PYTHONPATH` |

---

## 不在当前批次

连续对打、多机器人比赛、柔性球网、球体有限元变形和机器人搏击见路线图「暂缓项」。
学习基线（RL 训练配置与权重）属于 M5，不在 M3 批次内。

每项新运动的固定工作量：MuJoCo 场景 benchmark 器材夹具 → 固定 Shot Bank 数据与 manifest → 规则 Judge
→ 任务配置与注册表条目 → baseline 控制器 → 测试与文档。**网球已按这个顺序端到端做完（#12），可作为其余
三项的模板**：`BenchmarkEffector` 表加一项、`MujocoSportProfile` 加一项、`generate_shot_bank.py` 的
`BANKS` 加一个 `BankPlan`、`SCRIPTED_DEFAULTS` 加一组参数、`TaskEntry` 加一条。足球与篮球还需要新的
Judge（射门/投篮的接触序列不同），羽毛球可以复用 `NetReturnJudge`。

| 21 | ~~L3 目标从未告知策略~~（已修复） | 观测、`info`、`reset` 参数里都没有目标，只有读 `ShotSpec` 的参考夹具能完成 L3 | 学习策略在 L3 上只能瞎猜 | 2026-09-28：`info["target"]`、`wrappers.TargetObservation`、`PolicyController` 向接受 `target` 的 `reset` 传目标。见 [`POLICY_INTERFACE.md`](POLICY_INTERFACE.md) |
