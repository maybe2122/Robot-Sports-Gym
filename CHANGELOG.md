# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) 的结构，并计划从首次公开发布开始采用语义化版本。

## [Unreleased]

### Added

- 开源协作、治理、安全、引用和 benchmark 规范文档。
- GitHub issue、Pull Request 模板和 MuJoCo CI。
- README 增加 MuJoCo 与 Isaac Sim 五项单项场景的真实渲染对照图。
- 实验性 `table-tennis-return-v0` Shot Skill：固定 Shot Bank、MuJoCo 真接触后端、规则 Judge、分桶指标、报告 CLI 与脚本球拍测试夹具。
- 后端共享任务配置 `TableTennisReturnTaskConfig`：坐标约定、`TaskFrame` 平移、球台几何、control rate、动作/观测边界与 reward 权重集中定义；MuJoCo 环境、Runner 和 CLI 全部由它派生，报告新增 `task_config` 字段。
- 实验性 Isaac Lab `ManagerBasedRLEnv` 向量化环境，复用同一 Shot Bank、Judge 与 `EpisodeResult` schema；仓库 CI 无 Isaac 运行时，相关测试在缺少 `isaaclab` 时自动 skip，尚未实跑验证。

### Changed

- campus 场景各单项的地面偏移改为 `specs.CAMPUS_OFFSETS` 单一定义，MuJoCo 与 Isaac 场景构造共用。
- 项目展示名称由 MultiSport Physics Sim 更名为 Robot Sports Gym，突出面向多种机器人形态的球类运动训练与能力评测宗旨；现有 distribution、Python 包、CLI 和版本化 benchmark ID 保持兼容，当前版本仍处于 physics foundation 阶段。

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
