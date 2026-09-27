# Robot Sports Gym Roadmap

路线图按发布门槛排序，不代表时间承诺。进行中里程碑的逐项交付拆解、依赖关系和验收标准见 [`TODO.md`](TODO.md)。

## M0 — Physics foundation（当前）

- [x] 六种球类和比赛尺寸场景
- [x] MuJoCo + Isaac Sim/PhysX
- [x] 球体接触、空气阻力、Magnus 力和羽毛球稳定力矩
- [x] 回弹真实度量化报告
- [x] 基础测试、许可证和公开协作文档

## M1 — Benchmark API（P0）

已完成：版本化 Gymnasium 环境、固定 Shot Bank、后端无关事件/Judge、逐 episode schema、共享任务配置，以及在真实
Isaac Sim/Isaac Lab 运行时（CPU 与 GPU PhysX）上实跑、与 MuJoCo 逐条对比过的向量化环境。

- [x] 引入 Gymnasium，注册版本化环境 ID（实验性 MuJoCo `MultiSportRobot/TableTennisReturn-v0`）
- [x] 明确定义 observation/action/reward/termination/truncation（见 Shot Skill 文档）
- [x] 实现 seed、episode recorder 和机器可读结果 schema（固定 Shot Bank 和 `EpisodeResult`）
- [x] 通过 Gymnasium `check_env`（MuJoCo 乒乓球测试夹具）
- [x] 建立 MuJoCo/Isaac 共享任务配置和坐标约定（`TableTennisReturnTaskConfig` 与 `TaskFrame`）
- [x] Isaac Lab `ManagerBasedRLEnv` 向量化实现（CPU 与 GPU PhysX 各实跑 300 并行环境，与 MuJoCo 同批球逐条对比，见 [`ISAAC_SIM.md`](ISAAC_SIM.md)）

## M2 — Robot and sensor layer（P0）

机械臂侧已完成并在乒乓球上端到端跑通（`table-tennis-return-panda-v1`，见
[`ROBOT_LAYER.md`](ROBOT_LAYER.md)）；MuJoCo 传感器参考实现已完成，Isaac 侧实现和双足资产仍缺，
因此本里程碑尚未完成。

- [x] 选择许可证清晰的机械臂资产（Franka Panda，Apache-2.0，不 vendoring）
- [ ] 双足机器人资产（足球任务需要）
- [x] 实现 robot adapter，任务代码不绑定具体关节名
- [x] 加入位置/速度/力矩和末端控制模式（Panda 四种模式均实现）
- [x] 加入相机、IMU、接触、关节力矩和 frame transform（MuJoCo 参考实现）
- [x] 定义速度、力矩、碰撞和工作空间安全限制，接入 `failure_reason="safety"`

## M3 — Five canonical tasks（P0）

五项任务都有版本化环境、固定集、Judge 与三类基线（下限、参考控制器、学习基线），汇总见
[`reports/cross-task-summary.md`](../reports/cross-task-summary.md)。`TableTennisReturn` 有 Panda/G1 机体任务，
但机体在 Isaac 侧还没有 adapter，"同一机器人跑两个后端"尚未成立，因此保持未勾选；其余四项目前是 mocap 夹具任务。

- [x] TennisReturn
- [ ] TableTennisReturn
- [x] FootballKickToTarget（mocap 夹具；机体版需双足资产，见 [`LAUNCH_TASKS.md`](LAUNCH_TASKS.md)）
- [x] BadmintonServe（mocap 夹具，见 [`BADMINTON.md`](BADMINTON.md)）
- [x] BasketballShoot（mocap 夹具，见 [`LAUNCH_TASKS.md`](LAUNCH_TASKS.md)）
- [x] 每项任务提供 scripted/control baseline、random baseline 和至少一个学习基线（学习基线为挥拍原语上的 PPO，见 [`LEARNED_BASELINES.md`](LEARNED_BASELINES.md)）

## M4 — Fidelity and sim-to-real（P1）

- [ ] 高速相机轨迹数据：位置、速度、旋转和落点
- [ ] 球拍/脚/篮圈/篮板测力或冲量数据
- [ ] 滚动减速、表面摩擦和器材接触评测
- [ ] 双后端相同初始条件轨迹差异报告
- [ ] 参数辨识、置信区间和 held-out validation 数据
- [ ] 真实机器人日志与 sim-to-real gap 报告

## M5 — Reproducible baselines（P1）

- [ ] 固定 train/dev/test 协议与隐藏扰动集（train/dev/test 已固定；隐藏扰动集未做）
- [x] 5 个训练种子、置信区间和原始 episode 指标（五项任务的原语学习基线）
- [x] Stable-Baselines3/RSL-RL/skrl 等参考训练配置（SB3 PPO：`scripts/train_launch_policies.py`）
- [ ] 权重、数据、视频、硬件与墙钟时间记录（权重/训练曲线/硬件/墙钟已提交；学习基线的视频未做）
- [ ] CPU smoke CI、GPU/Isaac self-hosted CI 和 nightly fidelity 回归（CPU CI 覆盖五项任务冒烟；GPU/Isaac CI 与 nightly 未做）

## M6 — Public benchmark release（发布门槛）

- [x] 英文主文档与完整 API 文档（`README.en.md`、[`API.md`](API.md)）
- [x] 版本化任务和结果 schema（`multisport_sim.benchmark.schemas`）
- [x] 资产/数据/权重的逐项许可证清单（[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)）
- [x] 可审计 leaderboard 提交流程（[`LEADERBOARD.md`](LEADERBOARD.md) + `scripts/audit_submission.py`；排行榜本身未开放）
- [x] PyPI/容器或可复现环境锁文件（`uv.lock`，从零复现后 3.10/3.13 全部测试通过；wheel 可构建，未上传 PyPI）
- [ ] 归档 DOI、正式 citation 和 release notes（citation 与 release notes 已有；DOI 需在 GitHub Release 接入 Zenodo 后生成）
- [ ] 至少一个外部使用者成功复现实验

## 暂缓项

连续对打、多机器人比赛、柔性球网、球体有限元变形和机器人搏击不进入首个 benchmark release。它们应在核心单回合任务稳定后作为独立版本扩展。
