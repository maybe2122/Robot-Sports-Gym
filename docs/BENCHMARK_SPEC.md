# MultiSport Robot Benchmark Specification

状态：**Draft v0.1**。本文件定义拟议协议；当前仓库还不是可提交策略成绩的完整机器人 benchmark。

## 1. 目标与非目标

目标是用同一套运动对象、任务语义和评测指标，研究机器人对高速、旋转、接触敏感球体的感知、规划和控制。MuJoCo 用于快速控制研究，Isaac Lab/PhysX 用于并行训练、传感器仿真和视觉任务。

本 benchmark 不承诺两个引擎逐步状态完全相同，也不把仿真成绩视为真实机器人安全或 sim-to-real 成功证明。

## 2. 版本化对象

以下内容必须分别版本化，禁止只使用模糊的“latest”：

- `physics_spec`：尺寸、质量、接触和气动参数；
- `task_spec`：初始化、动作、观测、成功、终止和评分；
- `asset_spec`：机器人、传感器和运动器材资产；
- `randomization_spec`：训练、验证和测试分布；
- `report_schema`：机器可读结果格式。

环境 ID 使用 `MultiSportRobot/<Task>-<Robot>-vN`。`vN` 变化表示会影响可比性的任务协议变化。

## 3. 建议的首批任务

| Task | 任务 | 主成功条件 | 状态 |
|---|---|---|---|
| `TennisReturn` | 机械臂回击来球 | 球越网并落入目标区 | Planned |
| `TableTennisReturn` | 高速乒乓回球 | 合法触台且目标误差最小 | Planned |
| `FootballKickToTarget` | 双足或单腿机器人定点射门 | 球进入指定球门区域 | Planned |
| `BadmintonServe` | 机械臂发高远球 | 越网并落入发球区 | Planned |
| `BasketballShoot` | 机械臂定点投篮 | 球自上而下穿过篮圈 | Planned |

第二阶段再加入连续颠球、运球、移动接球、双机器人对打和多智能体比赛。首个公开版本应先把五个单回合任务做稳定，避免用大量未验证任务稀释基准质量。

## 4. 评测轨道

### State track

策略可接收机器人本体状态及无噪声或规定噪声的球位姿/速度。该轨道主要比较控制和规划。

### Vision track

策略只能使用机器人可获得的关节传感器、相机、IMU 和规定的接触传感器。仿真真值只能用于评测，不得进入策略输入。

### Robustness track

使用未在训练分布中出现的质量、摩擦、恢复系数、风、传感器噪声和执行器延迟组合，报告分布内与分布外性能差值。

### Sim-to-real track

使用公开的真实机器人日志或统一采集协议。仿真成绩和真实成绩分别报告，禁止只公布最佳真实试验。

## 5. 环境接口

单机器人任务应实现 Gymnasium `Env` 语义：

```text
reset(seed, options) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

双机器人同时行动任务采用 PettingZoo Parallel API。Isaac 侧优先实现 `ManagerBasedRLEnv`，将 scene、action、observation、event/randomization、reward、termination、curriculum 和 recorder 分离。

接口依据：[Gymnasium Env API](https://gymnasium.farama.org/api/env/)、[Gymnasium custom environment guide](https://gymnasium.farama.org/main/introduction/create_custom_env/)、[PettingZoo Parallel API](https://pettingzoo.farama.org/main/api/parallel/) 和 [Isaac Lab manager-based environment guide](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html)。

每个环境必须声明：

- `physics_dt`、`control_dt`、动作保持步数和最大 episode 时长；
- 动作空间的物理含义、单位、缩放和限制；
- 观测字段、形状、单位、坐标系、历史长度和噪声；
- `terminated` 与 `truncated` 的独立条件；
- 可由 `reset(seed=...)` 控制的全部随机源；
- `info` 中不影响策略但用于评分的原始事件和指标。

## 6. 机器人适配层

任务逻辑不得硬编码某个机器人关节名称。Robot adapter 至少暴露：

- 关节位置、速度、力矩限制；
- 末端执行器/脚/球拍/球拍面的 frame；
- 默认姿态、复位和自碰撞配置；
- 控制模式：位置、速度、力矩或末端目标；
- 传感器能力与仅评测可见的 privileged state。

首个版本建议选择一个开源许可证清晰的 6–7 DoF 机械臂作为网球、乒乓球、羽毛球和篮球基线，再为足球增加一个双足机器人。所有机器人资产必须提供上游链接、版本、许可证和修改记录。

## 7. 指标

每个任务报告原始指标，不能只给合成分数：

- `success_rate`：主指标，固定测试集上的成功比例及置信区间；
- `target_error_m`：首次合法落点、球门或篮圈目标误差；
- `contact_error`：击球时机、拍面位置/法向或脚—球接触误差；
- `safety_violations`：关节、力矩、速度、碰撞和工作空间违规次数；
- `energy_joule`：执行器机械功积分；
- `episode_time_s`：完成时间；
- `robustness_gap`：分布内与隐藏扰动集成功率差；
- `inference_latency_ms`：指定硬件上的策略推理延迟。

训练样本数、环境步数和墙钟时间是效率指标，不与任务成功率混成一个不可解释的数字。

## 8. 评测协议

- 训练种子、开发种子和最终测试种子必须不重叠。
- 每个测试条件至少运行 100 episodes；随机任务至少使用 5 个策略训练种子。
- 报告均值、标准差和 95% bootstrap 置信区间。
- 最终评测关闭探索噪声，除非任务协议明确要求随机策略。
- 不允许根据最终测试结果继续调参。
- 失败、超时、数值发散和安全终止都必须计入分母。
- 比较结果必须使用相同任务、资产、物理、随机化和报告 schema 版本。

建议的公开 split 编号将在任务实现时冻结；冻结前不得把临时种子称为 leaderboard test set。

## 9. 提交产物

一个可比较的结果包至少包含：

```text
submission/
├── manifest.json       # commit、task/physics/asset 版本、后端和硬件
├── config/             # 完整训练与评测配置
├── metrics.json        # 每个 episode 的原始指标及汇总
├── policy/             # 权重或可复现获取方式
├── videos/             # 固定成功/失败样本，不得只挑成功案例
└── environment.txt     # Python、驱动、模拟器和依赖版本
```

所有结果必须能由一条无交互命令复现。预训练权重和数据应使用内容哈希；大文件不直接提交到 Git 历史。

## 10. 当前发布门槛

在满足以下条件前，README 应继续标记为“benchmark planned”：

1. 至少一个 Gymnasium 环境和一个 Isaac Lab 向量化环境通过 API 检查；
2. 至少一个开源机器人适配器在两个后端完成同一任务；
3. 固定任务协议、测试 seeds 和结果 schema；
4. 提供 random、scripted、传统控制或 RL 参考基线；
5. 提供训练脚本、评测脚本、权重、视频和可复现报告；
6. 物理 fidelity 不只包含回弹，还覆盖任务涉及的飞行与器材接触。
