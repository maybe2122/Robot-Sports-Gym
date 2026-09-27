# Robot Sports Gym 文档

本项目面向多种机器人形态，以统一任务、物理规范和指标训练并评测机器人在多球类运动中的感知、规划与控制能力。

| 文档 | 内容 |
|---|---|
| [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) | 机器人球类任务、轨道、接口、指标和提交协议 |
| [`TABLE_TENNIS_SHOT_SKILL.md`](TABLE_TENNIS_SHOT_SKILL.md) | 实验性乒乓球单球回球测试、CLI、指标和适配接口 |
| [`ROBOT_LAYER.md`](ROBOT_LAYER.md) | 机器人适配器契约、控制模式、安全限制、资产许可证与 Franka Panda 乒乓球实现 |
| [`G1.md`](G1.md) | 第二个机体 Unitree G1：范围决策、站位与 ready pose 的实测来源、只有接第二台机器人才暴露的三个 bug、实测基线 |
| [`POLICY_INTERFACE.md`](POLICY_INTERFACE.md) | 策略接口：不同机器人的观测宽度不同时，字段怎么命名、怎么按名字取、跨机体怎么比 |
| [`TENNIS.md`](TENNIS.md) | 网球回球任务：与乒乓球共享的部分、必须不同的部分、线床标定与实测基线 |
| [`BADMINTON.md`](BADMINTON.md) | 羽毛球发球：发射类任务框架、BWF 发球规则、三处物理修正（插值夹具、0.5 ms 步长、压心来流）与基线 |
| [`LAUNCH_TASKS.md`](LAUNCH_TASKS.md) | 足球射门与篮球投篮：目标处的任务坐标系、IFAB/FIBA 判定、滚动球与 PGS 修正、冲量律夹具与基线 |
| [`LEARNED_BASELINES.md`](LEARNED_BASELINES.md) | M5 第一批学习基线：挥拍原语上的 PPO、5 种子协议、与随机原语的对比 |
| [`API.md`](API.md) | 英文 API 参考：任务、环境、固定集、Judge、runner、schema、脚本 |
| [`LEADERBOARD.md`](LEADERBOARD.md) | 英文：结果包提交要求与 `audit_submission.py` 审计流程 |
| [`SHOT_BANKS.md`](SHOT_BANKS.md) | 固定集规模与生成方式、train/dev/test 种子、L5 的观测噪声/延迟/域随机化 |
| [`VISION_TRACK.md`](VISION_TRACK.md) | 传感器层、声明的相机套件、参考感知管线的实测精度，以及视觉策略的接入方式 |
| [`SUBMISSION.md`](SUBMISSION.md) | 完整原始指标（contact_error / robustness_gap / inference_latency_ms）与可复现结果包 |
| [`M2_STATUS.md`](M2_STATUS.md) | M2 机器人层进度记录（2026-08-24）：已完成、验证程度、明确未做项 |
| [`ROADMAP.md`](ROADMAP.md) | 从物理场景仓库到公开 benchmark 的里程碑和发布门槛 |
| [`TODO.md`](TODO.md) | 里程碑的可执行拆解：交付项、依赖关系、验收标准和当前状态 |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | 版本、随机种子、环境记录和结果复现要求 |
| [`PHYSICS.md`](PHYSICS.md) | 物理模型、量化评分和标准来源 |
| [`ISAAC_SIM.md`](ISAAC_SIM.md) | Isaac Sim/PhysX 安装、运行和评测 |

当前发布状态是 **Alpha（v0.3.0）**：五项标准单回合任务均可在 MuJoCo 上运行（乒乓球有 Franka Panda 与 Unitree G1
机体和视觉轨道，其余四项为 mocap 夹具），每级 test 100 条的固定集、冻结的 L0–L5 判定、参考基线与学习基线、
跨任务汇总、Isaac Lab 实跑与跨后端一致性报告、带版本号的 schema 与可审计的结果包都已落地。所有固定集仍是
`experimental`，排行榜未开放。任何论文或报告都应注明使用的 commit、后端、Shot Bank 哈希和任务协议版本；
v0 小型固定集不能用于排行榜声明。
