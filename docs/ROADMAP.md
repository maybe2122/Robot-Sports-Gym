# Roadmap to an Open Robot Ball-Sports Benchmark

路线图按发布门槛排序，不代表时间承诺。

## M0 — Physics foundation（当前）

- [x] 五种球类和比赛尺寸场景
- [x] MuJoCo + Isaac Sim/PhysX
- [x] 球体接触、空气阻力、Magnus 力和羽毛球稳定力矩
- [x] 回弹真实度量化报告
- [x] 基础测试、许可证和公开协作文档

## M1 — Benchmark API（P0）

- [ ] 引入 Gymnasium，注册版本化环境 ID
- [ ] 明确定义 observation/action/reward/termination/truncation
- [ ] 实现 seed、episode recorder 和机器可读结果 schema
- [ ] 通过 Gymnasium `check_env`
- [ ] Isaac Lab `ManagerBasedRLEnv` 向量化实现
- [ ] 建立 MuJoCo/Isaac 共享任务配置和坐标约定

## M2 — Robot and sensor layer（P0）

- [ ] 选择许可证清晰的机械臂与双足机器人资产
- [ ] 实现 robot adapter，任务代码不绑定具体关节名
- [ ] 加入位置/速度/力矩和末端控制模式
- [ ] 加入相机、IMU、接触、关节力矩和 frame transform
- [ ] 定义速度、力矩、碰撞和工作空间安全限制

## M3 — Five canonical tasks（P0）

- [ ] TennisReturn
- [ ] TableTennisReturn
- [ ] FootballKickToTarget
- [ ] BadmintonServe
- [ ] BasketballShoot
- [ ] 每项任务提供 scripted/control baseline、random baseline 和至少一个学习基线

## M4 — Fidelity and sim-to-real（P1）

- [ ] 高速相机轨迹数据：位置、速度、旋转和落点
- [ ] 球拍/脚/篮圈/篮板测力或冲量数据
- [ ] 滚动减速、表面摩擦和器材接触评测
- [ ] 双后端相同初始条件轨迹差异报告
- [ ] 参数辨识、置信区间和 held-out validation 数据
- [ ] 真实机器人日志与 sim-to-real gap 报告

## M5 — Reproducible baselines（P1）

- [ ] 固定 train/dev/test 协议与隐藏扰动集
- [ ] 5 个训练种子、置信区间和原始 episode 指标
- [ ] Stable-Baselines3/RSL-RL/skrl 等参考训练配置
- [ ] 权重、数据、视频、硬件与墙钟时间记录
- [ ] CPU smoke CI、GPU/Isaac self-hosted CI 和 nightly fidelity 回归

## M6 — Public benchmark release（发布门槛）

- [ ] 英文主文档与完整 API 文档
- [ ] 版本化任务和结果 schema
- [ ] 资产/数据/权重的逐项许可证清单
- [ ] 可审计 leaderboard 提交流程
- [ ] PyPI/容器或可复现环境锁文件
- [ ] 归档 DOI、正式 citation 和 release notes
- [ ] 至少一个外部使用者成功复现实验

## 暂缓项

连续对打、多机器人比赛、柔性球网、球体有限元变形和机器人搏击不进入首个 benchmark release。它们应在核心单回合任务稳定后作为独立版本扩展。

