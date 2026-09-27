# 待办清单

本清单是 [`ROADMAP.md`](ROADMAP.md) 里程碑的可执行拆解：路线图给发布门槛，这里给具体交付项、依赖关系和验收标准。里程碑勾选以路线图为准，本文件跟踪进行中的工作。

状态含义：`done` 已合入并有测试覆盖；`blocked` 代码就位但缺运行环境验证；`todo` 未开始。

**最后核对：2026-09-02，分支 `feat/shared-task-config`。**

---

## 一、现在做到哪了

一句话：**乒乓球是一个能跑的机体基准（state + vision 两条轨道），网球是一个能跑的夹具基准；
其余三项运动、双足机体、Isaac 实跑、学习基线都还没有。**

| 维度 | 现状 |
|---|---|
| 已注册任务 | 4 个：`table-tennis-return-v0`（mocap 夹具）、`table-tennis-return-panda-v1`（Franka Panda，7 关节）、`table-tennis-return-g1-v1`（固定基座 Unitree G1，10 关节）、`tennis-return-v0`（mocap 夹具） |
| Gymnasium 环境 | 5 个，全部通过 `check_env`：`TableTennisReturn-v0`、`TableTennisReturn-Panda-v1`(obs 33)、`TableTennisReturn-Panda-Vision-v1`、`TableTennisReturn-G1-v1`(obs 39)、`TennisReturn-v0` |
| 观测轨道 | state 与 vision 两条；vision 的观测对象上没有 `ball` 字段，由构造保证 |
| 后端 | MuJoCo 全通；Isaac Lab 代码就位但**未实跑** |
| 固定集 | `table_tennis/return-v0`（冻结，dev 12 / test 24）、`table_tennis/return-v1` 与 `tennis/return-v0`（各 train 1200 / dev 300 / test 600，每级 100） |
| 指标 | `BENCHMARK_SPEC` §7 八项全部落地（含 safety、energy、contact_error、robustness_gap、latency） |
| 基线 | Panda 4 条（hold/random/intercept/vision）、G1 3 条（hold/random/intercept）、网球 2 条。**没有一条是可提交成绩**。Panda `intercept` L1 命中 100%（过 90% 门槛），G1 85% |
| 提交链路 | `scripts/package_submission.py`（`make submission`）产出完整结果包 |
| 测试 | 379 项收集，**378 通过 / 1 skip**（skip 是 `test_isaac_lab_env`，缺 Isaac Lab 运行时） |
| Lint | `ruff check src tests scripts` 零问题 |

按里程碑：**M1 差 Isaac 实跑；M2 差双足资产与 Isaac 侧实现；M3 五项运动做完两项。**

### M1 — Benchmark API

| # | 任务 | 状态 | 验收标准 |
|---|---|---|---|
| 1 | 后端共享任务配置 `TableTennisReturnTaskConfig` | done | 坐标约定、球台几何、control rate、动作/观测边界、reward 权重集中定义；MuJoCo 环境、Runner、CLI 全部由它派生 |
| 2 | world↔task 坐标帧变换 `TaskFrame` | done | 位置/速度/接触样本换算有往返测试；轴约定不一致时报错而非静默算错 |
| 3 | MuJoCo 路径消费共享配置 | done | `envs.py` 无硬编码 Box 边界；报告带 `task_config` 快照 |
| 4 | Isaac Lab `ManagerBasedRLEnv` 向量化环境 | done（CPU PhysX） | 2026-09-27 在 Isaac Sim 5.0 / Isaac Lab 0.46.2 上实跑 300 并行环境；`scripts/backend_parity.py` 与 MuJoCo 同批球逐条对比：飞行段中位差 3.5 mm，Judge 一致率 99%，接触段反弹高度差约一成（M4 待标定）。实跑修复了气动力双重施加与固定集参数漏传。GPU PhysX 未跑。见 [`ISAAC_SIM.md`](ISAAC_SIM.md) |

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
| 13 | FootballKickToTarget | todo | 11, 10b | 新 Judge（触球、目标区/球门命中、出界）、踢球器夹具、Shot Bank、环境注册、baseline |
| 14 | BadmintonServe | todo | 11 | 新 Judge（发球击球点合法性、过网、对角发球区）、球拍夹具、Shot Bank、环境注册、baseline |
| 15 | BasketballShoot | todo | 11 | 新 Judge（出手、篮圈/篮板接触序列、空心与打板进球）、投篮器夹具、Shot Bank、环境注册、baseline |
| 16 | 五项任务 baseline 与统一报告 | todo | 12–15 | 每项提供 random 与 scripted baseline，统一进入 CLI 与 JSON/Markdown 报告；跨任务指标汇总 |

---

## 二、接下来计划做什么

按依赖和收益排序。前三项是"把已经做完的东西变成可信的东西"，之后才是加运动。

| 顺序 | 动作 | 为什么现在做 | 完成标志 |
|---:|---|---|---|
| 1 | **提交本批次** | 工作区有 60+ 个未提交文件，整个 M2/M3 都还没进历史；现在产出的任何结果包都带脏树标记，不可复现（问题 1） | `git status` 干净，`package_submission.py` 的 manifest 不再标脏 |
| 2 | ~~机体基线表切到 `table_tennis/return-v1` 并重跑~~（done） | Panda 默认 bank 与脚本已切换，完整基线表已重跑 | dev 每格 n=50、test 每格 n=100；报告记录 v1 digest、逐级结果和区间 |
| 3 | ~~复查 L4/L5 分级构造~~（done） | L4 与 L5 是独立分层的 challenge distribution，只有 L5 加扰动，点估计不保证单调 | `robustness_gap.level_breakdown` 分开报告 L4/L5 的 n、点估计与 Wilson 区间；解释写入 `SUBMISSION.md` |
| 4 | **M3 剩余三项运动**（#13/#14/#15） | 网球已经把模板跑通，边际成本最低的扩展 | 每项：环境过 `check_env`、固定集有 manifest、两条基线进统一报告 |
| 5 | **双足机器人资产**（#10b） | 是 #13 足球的硬前置 | 与 Panda 同一套 `RobotAdapter` 与 `assets.py` 解析路径，许可证清单进报告 |
| 6 | **Isaac Lab GPU 实跑 + Isaac 侧 adapter/sensor 实现**（M1 #4） | "同一机器人跑两个后端"是发布门槛，目前只是设计不是事实（问题 4、5） | `test_isaac_lab_env` 不再 skip；双后端相同初始条件的轨迹差异报告 |
| 7 | **统一跨任务报告**（#16） | 五项运动齐了才有意义 | 一条命令产出全部任务 × 全部难度的汇总表 |
| 8 | **M4 保真标定 / M5 学习基线** | 没有学习基线，所有分数都还是脚本控制器读特权状态跑出来的（问题 8、9） | 见路线图 |

小项（随手可做，不排队）：

- ~~同步 `ROADMAP.md` 的传感器层与 TennisReturn 勾选~~（done）。
- ~~让 CI 与 Makefile 一样 lint `src tests scripts`~~（done）。

---

## 三、当前已知问题

按"会不会让别人读错结论"排序。前四条是会的。

| # | 问题 | 具体表现 | 影响 | 处理方向 |
|---:|---|---|---|---|
| 1 | **整批工作未提交** | 工作区 60+ 个改动/新增文件，`feat/shared-task-config` 上最后一次提交还停在框架泛化之前 | 结果包 manifest 会打脏树标记；任何人拿到 commit 都复现不出这些报告 | 分批提交，见计划 1 |
| 2 | ~~机体基线统计量不足~~（已修复） | Panda 默认固定集已改为 `table_tennis/return-v1`，完整报告已重跑 | dev 每格 n=50 / test 每格 n=100，v1 digest 与 Wilson 区间已记录 |
| 3 | ~~难度分级不单调~~（已澄清） | 网球 scripted 的聚合 gap 为负，根因是 L4/L5 独立分层而非嵌套样本，只有 L5 应用扰动 | 报告新增逐级 breakdown 与区间；规范明确两级点估计不承诺单调 |
| 4 | **Isaac Lab 从未实跑** | `test_isaac_lab_env` 是全套 379 项里唯一的 skip（缺运行时） | 双后端是发布门槛。目前 Isaac 路径的正确性完全没有证据 | 需要一台装 Isaac Sim 的 GPU 机器，先做最小实例化 |
| 5 | **Isaac 侧没有 robot / sensor 实现** | `robot.py`、`sensors.py` 是后端无关协议，但只有 MuJoCo 的实现（`backends/mujoco_robot.py`、`mujoco_sensors.py`） | "同一个 Panda 跑两个后端"目前是设计意图，不是已验证事实 | 与问题 4 一起做 |
| 6 | ~~力矩控制模式未实现~~（已修复） | Panda adapter 将 affine 位置 actuator 显式切换为 unit-gain、zero-bias torque actuator，并可逆恢复 | 命令以 N·m 解释并按 87/12 N·m 限幅；reset 回到位置模式；模式往返有物理测试 |
| 7 | **bucket 级样本量仍有限（已缓解）** | v1 每级 100 条，但每 bucket 只有 14–68 条 | L4/L5 主指标噪声仍较大 | `assessment` 已逐 bucket 报样本量与 Wilson 95% 区间；发表级结果仍应扩固定集 |
| 8 | **没有学习基线** | 全部基线都是脚本控制器，且 `intercept` / `scripted` 读特权球状态 | 现在所有分数刻画的是"夹具/机体能力上限"，不是"策略能做到什么" | M5 |
| 9 | **物理保真只覆盖回弹** | `reports/` 只有回弹保真；飞行段轨迹、击球冲量未对真实测量标定。拍面 e≈0.82–0.86 只是落在文献区间内，不是拟合本项目实测数据 | sim-to-real gap 无法量化 | M4 |
| 10 | **双足资产空缺** | 未选型 | 直接阻塞 #13 FootballKickToTarget | 计划 5 |
| 11 | **Panda 在这份固定集上的能力上限很硬**（已按 v1 重测） | 逐球标定：1197 条 train 球够到 1196 条、其中 1182 条来得及（L0–L3 全中，L4 193/197，L5 189/200）。拍面前向最大速度 1.16–4.70 m/s，合法回球（吊高）需要 3.44–6.48 m/s | 拦截已不是瓶颈——`intercept` 在 L1 命中 100%；回球才是，test 上 L2 只有 16%，L3 以上 `target_rate` 仍为 0 | **这是真实能力上限不是 bug**，但要在 spec 里写清楚。注意别读 `verdict.can_reach_every_shot_in_time`：那个判据打的是外接矩形的角点，没有球去过那里 |
| 12 | **CI 覆盖面仍窄（已缓解）** | CPU job 已 lint `src tests scripts`，校验基线报告的 bank digest/n/区间/breakdown，并 smoke 跑乒乓与网球 CLI；仍没有 GPU/Isaac job 或 nightly 物理重跑 | 报告与固定集口径漂移已会失败；物理回归漂移仍需 nightly | M5 的 CI 项 |
| 13 | **leaderboard 审计流程未定** | 结果包格式已定，怎么审没定 | M6 发布门槛 | M6 |
| 18 | **`incoming_valid_rate` 不是机体无关的** | L0 的主指标本应只描述固定集里的球合不合法，但一个在球完成第一落之前就因安全违规中止的 episode 记录不到它。G1 的 `random` 基线在 L0 拿 0%，Panda 的同一条拿 90%——差别只是违规发生的时刻 | L0 目前测的是控制器而不是固定集 | 要么让 L0 的分子不受中止影响，要么在 spec 里写明 L0 对中止敏感。改语义会移动已发布的 Panda 分数，所以先记不改 |
| 19 | **G1 上 `random` 基线没有区分能力** | 600 个回合 100% 以安全违规告终：kp=500 的人形伺服被随机指令驱动会立刻冲出声明的速度包络 | 随机基线本该给出一个可读的地板，在这个机体上给不出 | 要么给它一个机体相关的采样率/幅度，要么在报告里明说它在这个机体上只是安全包络的触发器 |
| 20 | **G1 的 `hold` 基线命中率不是 0** | ready pose 把球拍停在来球路径上，test L3 上 `hold` 命中 53% | 读 `intercept` 的命中率要减掉这个地板，否则会高估控制的贡献 | 报告已并列 `hold` 一行；spec 里应写明命中率的地板是机体相关的 |
| 15 | **`valid_return_rate` 在这个机体上随难度递增，`robustness_gap` 因此读不出鲁棒性** | test `intercept`：L1 命中 100%/回球 **0%**，L4 命中 59%/回球 **28%**，gap −9%。实测来球速度随难度单调上升（击球平面处 |vx| 中位数 L1 2.64 → L5 4.17 m/s），而出球速度 ≈ 1.85·v_拍 + 0.85·v_来球，拍速又是硬上限——**来球越快越容易打回去**。去掉 L1 后 gap 仍是 −4%（test），所以不只是分子口径问题 | 一个负的 gap 看起来像"扰动让策略变好了"；实际是难度轴和回球难度对速度受限的机体是反的 | 短期：gap 分子只取主指标为回球率的级别，并在 spec 里写明这一非单调性的机制。长期：难度分级不该只沿来球速度加码 |
| 16 | ~~`StrikeZone` 声明值仍是 v0 量纲~~（已修复） | 默认值改为 v1 train 实测穿越范围向外取整：y∈[-0.72,0.71]、z∈[0.70,1.42] | 测试改为在 dev split 真实发球取穿越点，钉住"声明区域包含实测穿越"与"Panda IK 覆盖真实穿越点"；不再对外接矩形角点求 IK |
| 17 | **拦截基线的最坏单步延迟仍偏高** | `inference_latency_ms` 均值 0.52 ms、p95 0.72 ms（预算 5 ms），但 max 达 11.7 ms | 单个控制步超时；对脚本基线无害，但作为"参考实现"的延迟画像不干净 | 大概率是首次调用的 numpy/IK 预热，需确认后要么预热要么在报告里注明 |
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
