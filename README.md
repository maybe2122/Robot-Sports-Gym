# MultiSport Physics Sim（多球类物理仿真）

一个独立、无外部美术资产依赖的 **MuJoCo + Isaac Sim/PhysX 双后端** 3D 运动场仓库。项目按国际比赛尺寸程序化建立网球、乒乓球、足球、羽毛球和篮球场景，并为每项运动提供球、球拍/球台/球门/篮架以及差异化物理。

## 已实现内容

| 场景 | 场地与器材 | 运动物理 |
|---|---|---|
| 网球 | 23.77 × 10.97 m 场地、单双打线、网柱/球网、2 支球拍、网球 | 57.7 g 球体、硬地摩擦、ITF 落球回弹、二次空气阻力、旋转 Magnus 力 |
| 乒乓球 | 2.74 × 1.525 × 0.76 m 球台、15.25 cm 球网、2 支球拍、乒乓球 | 2.7 g / 40 mm 球、球桌专用接触副、约 23 cm 标准回弹、空气阻力与旋转 |
| 足球 | 105 × 68 m 球场、边线/中圈/禁区、2 个 7.32 × 2.44 m 球门、足球 | 430 g 5 号球、草地摩擦/滚阻、弹性、空气阻力与弧线球 Magnus 力 |
| 羽毛球 | 13.40 × 6.10 m 单双打场线、1.55 m 球网、2 支球拍、16 羽球 | 5.0 g 羽毛球、方向相关投影面积、二次阻力、压心偏置自动稳定力矩、软木低回弹 |
| 篮球 | 28 × 15 m 球场、中圈/罚球区/三分线、3.05 m 双篮架、篮球 | 600 g 7 号球、木地板摩擦/滚阻、FIBA 落球回弹、空气阻力与旋转 |

`campus` 模式会同时加载以上全部运动；每项运动也有独立场景，便于近距离观察和训练环境扩展。两个后端共享尺寸、质量、气动参数和场景布局，不是互不相关的两个示例。

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

# 无头运行和 USD 导出
multisport-isaac --headless --device cpu --scene campus --duration 2
multisport-isaac --headless --device cpu --scene tennis --duration 0.1 \
  --export-usd generated/tennis.usda

# PhysX 规则落球测试
multisport-isaac --headless --device cpu --scene table_tennis --drop-test
multisport-isaac --headless --device cpu --scene basketball --drop-test
```

没有执行 editable install 时，也可直接运行：

```bash
PYTHONPATH=src /home/maybe/code/rl/env_isaaclab/bin/python \
  -m multisport_sim.isaac_cli --headless --device cpu --scene campus --duration 1
```

Isaac 后端使用 240 Hz PhysX TGS、CCD、每种球独立的 PhysX 材质、质量和恢复系数。空气阻力与 Magnus 力通过 Isaac Lab `RigidObject` 外力接口每个物理步施加；生成的 USD 可直接在 Isaac Sim 中再次打开。更多说明见 [docs/ISAAC_SIM.md](docs/ISAAC_SIM.md)。

## 后端对应关系

| 能力 | MuJoCo | Isaac Sim |
|---|:---:|:---:|
| 五个单项场景 + campus | ✓ | ✓ |
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
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
```

普通测试会编译全部 MuJoCo 场景、验证全部 Isaac 场景描述、核对球体质量/尺寸与器材名称、验证阻力方向和速度平方律，并实际仿真 MuJoCo 的 ITF/ITTF/FIBA 落球回弹。`test-isaac` 会真正启动 Isaac Sim、初始化全部五个 PhysX 球体并运行 campus。

## 设计边界

这是刚体动力学和接触/气动力仿真，不是有限元球体变形模型。球拍目前固定在场边作为带碰撞的器材，场景没有人体运动员或自动比赛规则；后续可在稳定的 `Simulation` API 上接控制器、机器人、强化学习环境或轨迹回放。

## License

MIT，见 [LICENSE](LICENSE)。
