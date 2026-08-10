# Robot Sports Gym 文档

本项目面向多种机器人形态，以统一任务、物理规范和指标训练并评测机器人在多球类运动中的感知、规划与控制能力。

| 文档 | 内容 |
|---|---|
| [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) | 机器人球类任务、轨道、接口、指标和提交协议 |
| [`TABLE_TENNIS_SHOT_SKILL.md`](TABLE_TENNIS_SHOT_SKILL.md) | 实验性乒乓球单球回球测试、CLI、指标和适配接口 |
| [`ROADMAP.md`](ROADMAP.md) | 从物理场景仓库到公开 benchmark 的里程碑和发布门槛 |
| [`TODO.md`](TODO.md) | 里程碑的可执行拆解：交付项、依赖关系、验收标准和当前状态 |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | 版本、随机种子、环境记录和结果复现要求 |
| [`PHYSICS.md`](PHYSICS.md) | 物理模型、量化评分和标准来源 |
| [`ISAAC_SIM.md`](ISAAC_SIM.md) | Isaac Sim/PhysX 安装、运行和评测 |

当前发布状态是 **Alpha**：场景和球体物理可运行，`table-tennis-return-v0` 已提供实验性 MuJoCo Shot Skill 链路和 Gymnasium 测试夹具，但真实机器人和 Isaac Lab RL 环境尚未实现。任何论文或榜单都应注明使用的 commit、后端、Shot Bank 哈希和任务协议版本；v0 小型固定集不能用于排行榜声明。
