# Robot Sports Gym（机器人球类运动训练与评测平台）

[English](README.en.md) | 简体中文

Robot Sports Gym（RSG）是面向多种机器人形态的球类运动训练与评测平台，目标是在统一任务、物理规范和指标下，通过网球、乒乓球、足球、羽毛球、篮球和壁球训练并评测机器人在感知、规划、控制、鲁棒性与 sim-to-real 方面的能力。当前仓库提供无外部美术资产依赖的 **MuJoCo + Isaac Sim/PhysX 双后端** 物理基础层，并按国际比赛尺寸程序化构建场地、球体和运动器材。

> **项目状态：Alpha。** 场景、球体物理和回弹量化已可运行，并提供实验性的乒乓球 Shot Skill 测试系统及 MuJoCo Gymnasium 测试夹具；真实机器人适配、Isaac Lab RL 环境、其他标准任务和参考策略仍在路线图中。当前版本不应宣传为已完成的机器人球类 benchmark。

## 文档导航

- [机器人 benchmark 协议草案](docs/BENCHMARK_SPEC.md)
- [乒乓球 Shot Skill 测试系统](docs/TABLE_TENNIS_SHOT_SKILL.md)
- [开源 benchmark 路线图](docs/ROADMAP.md)
- [复现规范](docs/REPRODUCIBILITY.md)
- [物理模型与真实度评测](docs/PHYSICS.md)
- [Isaac Sim 使用说明](docs/ISAAC_SIM.md)
- [贡献指南](CONTRIBUTING.md) · [治理](GOVERNANCE.md) · [安全策略](SECURITY.md) · [变更记录](CHANGELOG.md)

## 已实现内容

| 场景 | 场地与器材 | 运动物理 |
|---|---|---|
| 网球 | 23.77 × 10.97 m 场地、单双打线、网柱/球网、2 支球拍、网球 | 57.7 g 球体、硬地摩擦、ITF 落球回弹、二次空气阻力、旋转 Magnus 力 |
| 乒乓球 | 2.74 × 1.525 × 0.76 m 球台、15.25 cm 球网、2 支球拍、乒乓球 | 2.7 g / 40 mm 球、球桌专用接触副、约 23 cm 标准回弹、空气阻力与旋转 |
| 足球 | 105 × 68 m 球场、边线/中圈/禁区、2 个 7.32 × 2.44 m 球门、足球 | 430 g 5 号球、草地摩擦/滚阻、弹性、空气阻力与弧线球 Magnus 力 |
| 羽毛球 | 13.40 × 6.10 m 单双打场线、1.55 m 球网、2 支球拍、16 羽球 | 5.0 g 羽毛球、方向相关投影面积、二次阻力、压心偏置自动稳定力矩、软木低回弹 |
| 篮球 | 28 × 15 m 球场、中圈/罚球区/三分线、3.05 m 双篮架、篮球 | 600 g 7 号球、木地板摩擦/滚阻、FIBA 落球回弹、空气阻力与旋转 |
| 壁球 | 9.75 × 6.40 m 单打封闭场地、前后及侧墙、发球区、壁球 | 地板/围墙碰撞、空气阻力与旋转 |

`campus` 模式会同时加载以上全部运动；每项运动也有独立场景，便于近距离观察和训练环境扩展。两个后端共享尺寸、质量、气动参数和场景布局，不是互不相关的两个示例。

## 双后端场景预览

以下图片由当前仓库代码和各单项场景的默认相机直接渲染，不是概念图。点击图片可在 GitHub 中查看原始分辨率。

| 运动 | MuJoCo | Isaac Sim / PhysX |
|:---:|:---:|:---:|
| 网球 | [![MuJoCo 网球场景](docs/images/rendered/mujoco/tennis.png)](docs/images/rendered/mujoco/tennis.png) | [![Isaac Sim 网球场景](docs/images/rendered/isaac/tennis.png)](docs/images/rendered/isaac/tennis.png) |
| 乒乓球 | [![MuJoCo 乒乓球场景](docs/images/rendered/mujoco/table_tennis.png)](docs/images/rendered/mujoco/table_tennis.png) | [![Isaac Sim 乒乓球场景](docs/images/rendered/isaac/table_tennis.png)](docs/images/rendered/isaac/table_tennis.png) |
| 足球 | [![MuJoCo 足球场景](docs/images/rendered/mujoco/football.png)](docs/images/rendered/mujoco/football.png) | [![Isaac Sim 足球场景](docs/images/rendered/isaac/football.png)](docs/images/rendered/isaac/football.png) |
| 羽毛球 | [![MuJoCo 羽毛球场景](docs/images/rendered/mujoco/badminton.png)](docs/images/rendered/mujoco/badminton.png) | [![Isaac Sim 羽毛球场景](docs/images/rendered/isaac/badminton.png)](docs/images/rendered/isaac/badminton.png) |
| 篮球 | [![MuJoCo 篮球场景](docs/images/rendered/mujoco/basketball.png)](docs/images/rendered/mujoco/basketball.png) | [![Isaac Sim 篮球场景](docs/images/rendered/isaac/basketball.png)](docs/images/rendered/isaac/basketball.png) |
| 壁球 | [![MuJoCo 壁球场景](docs/images/rendered/mujoco/squash.png)](docs/images/rendered/mujoco/squash.png) | [![Isaac Sim 壁球场景](docs/images/rendered/isaac/squash.png)](docs/images/rendered/isaac/squash.png) |

## MuJoCo 快速开始

需要 Python 3.10+。Linux 桌面运行原生 MuJoCo 查看器还需要可用的 OpenGL/GLFW。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"

# 全部场地总览
multisport-sim --scene campus

# 单项场景
multisport-sim --scene tennis
multisport-sim --scene table_tennis
multisport-sim --scene football
multisport-sim --scene badminton
multisport-sim --scene basketball
multisport-sim --scene squash
```

查看器中按 `Space` 再次发球，按 `R` 完整复位。鼠标操作沿用 MuJoCo 查看器：左键旋转、中键平移、滚轮缩放。

无显示器的服务器可以运行：

```bash
multisport-sim --scene campus --headless --duration 5
multisport-sim --scene badminton --headless --duration 3 --wind 2 0 0
multisport-sim --scene table_tennis --headless --duration 2 --no-launch
```

## Isaac Sim / PhysX 快速开始

Isaac 后端已在 Isaac Sim 5.0.0 + Isaac Lab 0.46.2 上验证。先进入已安装 Isaac Sim/Isaac Lab 的 Python 环境，再安装本仓库：

```bash
# 示例：使用当前工作区已有的 Isaac 环境
/home/maybe/code/rl/env_isaaclab/bin/python -m pip install -e . --no-deps

# GUI：加载全部场地，自动发球
multisport-isaac --scene campus

# 单项场景
multisport-isaac --scene badminton --wind 2 0 0
multisport-isaac --scene basketball
multisport-isaac --scene squash

# 完整得分演示：A 发球、B 回击、两次落地、B 得分（0:0 → 0:1）
multisport-isaac --scene squash --squash-demo --duration 9 \
  --demo-report reports/isaac/squash-demo.json

# 用同一条真实 PhysX 回合重建 JSON 报告和下方 GIF
make demo-squash ISAAC_PYTHON=/home/maybe/code/rl/env_isaaclab/bin/python

# 不启动 Isaac，独立复核已有 JSON 与 GIF 是否构成完整得分证据
make verify-squash-demo PYTHON=.venv/bin/python

# 无头运行和 USD 导出
multisport-isaac --headless --device cpu --scene campus --duration 2
multisport-isaac --headless --device cpu --scene tennis --duration 0.1 \
  --export-usd generated/tennis.usda

# PhysX 规则落球测试
multisport-isaac --headless --device cpu --scene table_tennis --drop-test
multisport-isaac --headless --device cpu --scene basketball --drop-test
```

演示中的前墙接触和两次落地由 PhysX 球路产生，脚本只施加发球与回击两次球拍冲量；JSON
报告会记录完整事件时间线，并且只有实际观察到第二次落地后才输出 `complete: true`。GIF
生成器还会核对固定事件顺序、最终 `0:1` 比分、帧数和逐帧变化，失败时拒绝覆盖演示资产。
`make demo-squash` 会把可提交的证据 JSON 与 GIF 一起写入 `docs/images/shot-skill/`。
独立校验器不依赖 Isaac 或 Pillow，可在普通 Python/CI 中复核事件时间线及 GIF 容器元数据。

![Isaac Sim 壁球完整得分演示](docs/images/shot-skill/squash-serve-score-demo.gif)

没有执行 editable install 时，也可直接运行：

```bash
PYTHONPATH=src /home/maybe/code/rl/env_isaaclab/bin/python \
  -m multisport_sim.isaac_cli --headless --device cpu --scene campus --duration 1
```

Isaac 后端使用 240 Hz PhysX TGS、CCD、每种球独立的 PhysX 材质、质量和恢复系数。空气阻力与 Magnus 力通过 Isaac Lab `RigidObject` 外力接口每个物理步施加；生成的 USD 可直接在 Isaac Sim 中再次打开。更多说明见 [docs/ISAAC_SIM.md](docs/ISAAC_SIM.md)。

## 后端对应关系

| 能力 | MuJoCo | Isaac Sim |
|---|:---:|:---:|
| 六个单项场景 + campus | ✓ | ✓ |
| 全部场地、器材和球 | ✓ | ✓ |
| 接触恢复、摩擦、滚阻 | MuJoCo soft contact | PhysX materials + TGS/CCD |
| 二次阻力、Magnus 力 | passive callback | RigidObject external wrench |
| 羽毛球方向阻力/稳定力矩 | ✓ | ✓ |
| 无头运行 | ✓ | ✓ |
| 可导出 USD | — | ✓ |

## Python API

```python
from multisport_sim import build_model
from multisport_sim.simulation import Simulation

model = build_model("campus")

simulation = Simulation("badminton", wind=(1.5, 0.0, 0.0))
simulation.launch()
for _ in range(1_000):
    simulation.step()
```

所有尺寸、质量、半径与气动系数集中在 `src/multisport_sim/specs.py`；MuJoCo 场地由 `scene.py` 生成，Isaac 场地由 `isaac_scene.py` 描述并由 `isaac_backend.py` 转成 USD/PhysX。详细公式、回弹校准和标准来源见 [docs/PHYSICS.md](docs/PHYSICS.md)。

## 验证

```bash
make test
make lint
make verify
make benchmark
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
```

普通测试会编译全部 MuJoCo 场景、验证全部 Isaac 场景描述、核对球体质量/尺寸与器材名称、验证阻力方向和速度平方律，并实际仿真 MuJoCo 的 ITF/ITTF/FIBA 落球回弹。`test-isaac` 会真正启动 Isaac Sim、初始化全部六个 PhysX 球体并运行 campus。

## 乒乓球 Shot Skill

实验性的 `table-tennis-return-v0` 已实现固定 Shot Bank、MuJoCo 自动发球、真实球拍接触、合法回球/目标落点 Judge、分桶指标，以及 JSON/Markdown 报告。内置脚本 mocap 球拍只用于验证测试链路，不属于可提交的机器人策略。

```bash
# 完整链路基线
multisport-benchmark --level L1 --split dev --controller scripted \
  --report reports/table-tennis-l1.json --markdown reports/table-tennis-l1.md

# 无动作失败基线
multisport-benchmark --level L1 --split dev --controller noop
```

坐标约定、L0–L5 门槛、报告字段、固定集完整性和机器人 adapter 要求见 [乒乓球 Shot Skill 文档](docs/TABLE_TENNIS_SHOT_SKILL.md)。

## 真实度量化评测

仓库提供统一的 `multisport-fidelity-v1` 报告格式。它不是只检查参数，而是实际执行落球仿真，测量第一次回弹最高点，并输出绝对误差、相对误差、容差占用率、等效恢复系数、PASS/FAIL、单项分数和总分。ITF、ITTF、FIBA 指标与项目工程指标分别统计，避免混淆官方标准和经验校准范围。

```bash
# MuJoCo：一次评测六种球，生成 JSON 和 Markdown
make evaluate PYTHON=/path/to/python

# Isaac Sim：每种球在独立 Kit 进程中评测，再聚合为同格式报告
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=/path/to/python

# 也可只评测指定项目
multisport-eval --sports tennis table_tennis basketball
multisport-isaac --headless --device cpu --scene basketball \
  --evaluate --report reports/isaac/basketball.json
```

当前提交附带的基线报告：MuJoCo 为 [`reports/mujoco-fidelity.md`](reports/mujoco-fidelity.md)，Isaac Sim 为 [`reports/isaac-fidelity.md`](reports/isaac-fidelity.md)。评分公式、参考区间和解释见 [`docs/PHYSICS.md`](docs/PHYSICS.md)。高分表示这些已测指标接近参考值，不代表尚未测量的球拍碰撞、柔性网或完整比赛行为已经得到验证。

## 设计边界

这是刚体动力学和接触/气动力仿真，不是有限元球体变形模型。普通展示场景中的球拍固定在场边；Shot Skill 模式会额外加载独立 mocap 拍面作为测试夹具，但它不具备真实机器人的关节、执行器、动力学或安全约束。项目仍没有人体运动员或完整比赛规则。

拟议的首批机器人任务、state/vision/robustness 轨道、指标、结果包和发布门槛见 [benchmark 协议](docs/BENCHMARK_SPEC.md)。实验性 Shot Skill 可用于开发和回归测试，但在这些门槛满足前不能作为完整机器人 benchmark 或排行榜发布。

## 开源协作与引用

提交代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。资产、数据和权重必须具有可再分发的许可证与明确来源。研究使用可引用 [CITATION.cff](CITATION.cff)；获得归档 DOI 后会更新正式引用信息。

## License

MIT，见 [LICENSE](LICENSE)。
