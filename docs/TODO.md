# 待办清单

本清单是 [`ROADMAP.md`](ROADMAP.md) 里程碑的可执行拆解：路线图给发布门槛，这里给具体交付项、依赖关系和验收标准。里程碑勾选以路线图为准，本文件跟踪进行中的工作。

状态含义：`done` 已合入并有测试覆盖；`blocked` 代码就位但缺运行环境验证；`todo` 未开始。

## M1 — Benchmark API

| # | 任务 | 状态 | 验收标准 |
|---|---|---|---|
| 1 | 后端共享任务配置 `TableTennisReturnTaskConfig` | done | 坐标约定、球台几何、control rate、动作/观测边界、reward 权重集中定义；MuJoCo 环境、Runner、CLI 全部由它派生 |
| 2 | world↔task 坐标帧变换 `TaskFrame` | done | 位置/速度/接触样本换算有往返测试；轴约定不一致时报错而非静默算错 |
| 3 | MuJoCo 路径消费共享配置 | done | `envs.py` 无硬编码 Box 边界；报告带 `task_config` 快照 |
| 4 | Isaac Lab `ManagerBasedRLEnv` 向量化环境 | blocked | 需在装有 Isaac Sim + Isaac Lab 的 GPU 工作站实跑：先做 `make_env_cfg(num_envs=4, device="cpu")` 最小实例化，暴露 scene/sensor 配置错误；再验证逐物理步 Judge 回调与 filtered contact 语义映射。详见 [`ISAAC_SIM.md`](ISAAC_SIM.md) |

## M2 — Robot and sensor layer

五项任务的机器人层。除资产接入外都不依赖具体机器人型号，可先行完成。

| # | 任务 | 状态 | 依赖 | 验收标准 |
|---|---|---|---|---|
| 6 | 后端无关 robot adapter 协议 | todo | — | `RobotAdapter` Protocol（robot_id、有序 joint_names、dof、末端、reset/observe/apply）与 `RobotObservation`；任务代码只按下标和语义角色访问，不绑定关节名 |
| 7 | 位置/速度/力矩与末端控制模式 | todo | 6 | `ControlMode` 枚举与对应命令类型；adapter 声明支持的模式集合，不支持的模式明确报错 |
| 8 | 速度/力矩/碰撞/工作空间安全限制 | todo | 6 | `SafetyLimits.violations()` 覆盖四类违规；违规接到目前无人触发的 `failure_reason="safety"` 判定路径 |
| 9 | 相机/IMU/接触/关节力矩/frame transform 传感器层 | todo | 6 | 后端无关 `SensorSuite`；MuJoCo 侧参考实现有测试，Isaac 侧保留同名接口 |
| 10 | 许可证清晰的机械臂与双足资产接入 | todo | 6, 7 | 逐项许可证清单文档；不 vendoring 资产的加载路径（本地 checkout / 环境变量），缺资产时测试自动 skip |

## M3 — Five canonical tasks

| # | 任务 | 状态 | 依赖 | 验收标准 |
|---|---|---|---|---|
| 11 | 多运动任务框架泛化 | done | — | `ShotTaskConfig` 基类、`ShotJudge` 协议与 `RectangularSurface`、可复用 `NetReturnJudge`、按 sport 参数化的 `MujocoSportProfile`、`TaskEntry` 注册表 |
| 12 | TennisReturn | todo | 11 | 复用 `NetReturnJudge`；需网球 benchmark 球拍夹具、固定 Shot Bank + manifest、环境注册、baseline |
| 13 | FootballKickToTarget | todo | 11 | 新 Judge（触球、目标区/球门命中、出界）、踢球器夹具、Shot Bank、环境注册、baseline |
| 14 | BadmintonServe | todo | 11 | 新 Judge（发球击球点合法性、过网、对角发球区）、球拍夹具、Shot Bank、环境注册、baseline |
| 15 | BasketballShoot | todo | 11 | 新 Judge（出手、篮圈/篮板接触序列、空心与打板进球）、投篮器夹具、Shot Bank、环境注册、baseline |
| 16 | 五项任务 baseline 与统一报告 | todo | 12–15 | 每项提供 random 与 scripted baseline，统一进入 CLI 与 JSON/Markdown 报告；跨任务指标汇总 |

每项新运动的固定工作量：MuJoCo 场景 benchmark 器材夹具 → 固定 Shot Bank 数据与 manifest → 规则 Judge → 任务配置与注册表条目 → baseline 控制器 → 测试与文档。建议先把网球端到端做完作为模板，再复制到其余三项。

学习基线（RL 训练配置与权重）属于 M5，不在 M3 批次内。

## 不在当前批次

连续对打、多机器人比赛、柔性球网、球体有限元变形和机器人搏击见路线图「暂缓项」。
