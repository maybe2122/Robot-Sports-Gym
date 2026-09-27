# Isaac Sim / PhysX 后端

## 版本和启动顺序

已验证组合：

- Isaac Sim 5.0.0
- Isaac Lab 0.46.2
- Python 3.11
- CPU PhysX 与 RTX 5090 图形设备

`isaac_cli.py` 只在启动前导入 `AppLauncher`。`isaac_backend.py` 中的 `omni`、Isaac Lab simulation 和 PhysX 接口会在 SimulationApp 完成启动后才加载，符合 Isaac 扩展生命周期要求。

## 场景构造

`isaac_scene.py` 是不依赖 Omniverse 的场景描述层，生成带名称、位置、姿态、尺寸、颜色、碰撞和材质参数的 primitive 列表。它有普通 pytest 覆盖，因此不启动 Isaac 也能检查六类球、场地和器材是否齐全。

运行时 `isaac_backend.py` 将描述转换成：

- `CuboidCfg`：场地、场线、球台、球网、篮板；
- `CylinderCfg` / `CapsuleCfg`：球网柱、球门、篮圈、球拍框和拍线；
- `RigidObjectCfg + SphereCfg`：带显式质量、碰撞和 PhysX 材质的六类球；
- 羽毛球的 16 根羽毛、足球色块和篮球缝线作为动态球体的纯视觉子 prim。

campus 中的所有六类球都具备 `UsdPhysics.RigidBodyAPI`、质量 schema 和子碰撞网格。

## 物理循环

PhysX 负责重力、接触、恢复、滑动/扭转摩擦和刚体旋转。每个 1/240 秒步长：

1. 从 PhysX tensor view 读取球的世界线速度、角速度和姿态；
2. 计算相对风、二次阻力与 Magnus 力；
3. 对羽毛球计算随姿态变化的投影面积和压心稳定力矩；
4. 用世界坐标 external wrench 写回各 `RigidObject`；
5. 执行 TGS/CCD 物理步并更新 tensor view。

## 命令

```bash
# GUI
multisport-isaac --scene campus

# 无头 CPU 验收
multisport-isaac --headless --device cpu --scene campus --duration 0.1

# 导出可重载场景
multisport-isaac --headless --device cpu --scene campus --duration 0.1 \
  --export-usd generated/multisport_campus.usda

# 标准落球测试
multisport-isaac --headless --device cpu --scene tennis --drop-test

# 壁球完整得分回合，同时输出得分和事件时间线
multisport-isaac --scene squash --squash-demo --duration 9 \
  --demo-report reports/isaac/squash-demo.json

# 无头重建经过校验的演示 GIF 和 JSON
make demo-squash ISAAC_PYTHON=/path/to/isaac/python

# 普通 Python 环境中独立校验已有演示资产
make verify-squash-demo PYTHON=/path/to/python

# 输出与 MuJoCo 相同 schema 的量化报告
multisport-isaac --headless --device cpu --scene tennis \
  --evaluate --report reports/isaac/tennis.json
```

`--duration 0` 在 GUI 中表示运行到用户关闭窗口；无头模式没有给时长时自动运行 5 秒。

壁球演示从 `0:0` 开始，顺序为 A 发球、前墙、首次落地、B 回击、前墙、首次落地、
第二次落地，最后判 B 得分并把场内计分板更新为 `0:1`。前墙接触和落地由 PhysX 速度反向检测，
脚本只负责两次球拍冲量；若给定时长内没有观察到完整事件链，命令会报错而不会生成伪造分数。
`scripts/capture_squash_demo.py` 通过 Isaac Camera 捕获同一回合，添加比分/事件字幕和纯视觉球体标记，
并在覆盖 GIF 前验证事件顺序、最终比分和所有相邻帧均发生变化；视觉标记没有碰撞，不参与物理判定。
`multisport_sim.squash_demo` 还能在不加载 Isaac/Pillow 的情况下独立检查 JSON 契约和 GIF
容器的尺寸、帧数、播放时长与体积，适合放入普通 CPU CI。录制目标会将这两个可提交的证据文件
同时写入 `docs/images/shot-skill/`，避免校验依赖被忽略的本地报告目录。

Isaac Sim 每个 Kit 进程只评测一个单项场景，以避免重复创建 `SimulationContext`。`make evaluate-isaac` 会依次启动六个进程，并将结果聚合为 `reports/isaac-fidelity.json` 和 Markdown 报告。

## Isaac Lab 向量化 Shot Skill 环境（experimental，CPU 与 GPU PhysX 均已实跑）

`src/multisport_sim/benchmark/backends/isaac_lab.py` 提供 `table-tennis-return-v0` 的
`ManagerBasedRLEnv` 向量化实现，与 MuJoCo 环境共享同一份
[`TableTennisReturnTaskConfig`](../src/multisport_sim/benchmark/task_config.py)、同一个固定
Shot Bank、同一个 `TableTennisReturnJudge` 和同一套 `EpisodeResult` schema，因此两个后端只可能在
物理求解上不同，不会在任务定义、判定规则或 reward 上分叉。

```python
# 必须先由 AppLauncher 启动 SimulationApp
from multisport_sim.benchmark.backends.isaac_lab import (
    IsaacTableTennisReturnEnv,
    make_env_cfg,
)

env = IsaacTableTennisReturnEnv(make_env_cfg(num_envs=1024, device="cuda:0"))
```

设计要点：

- **任务帧即环境原点。** 每个 env 的球台建在自身原点上，观测在返回前减去 `scene.env_origins`，因此
  `TaskFrame` 在 Isaac 一侧同样是恒等变换，Judge 不需要任何后端偏移。
- **逐物理步判定。** 乒乓球的一次台面接触远短于一个 control step，所以 Judge 由 physics callback
  驱动（物理步 1 ms，与 MuJoCo 模型步长一致），而不是每个 control step 采样一次接触。
- **语义接触来自 filtered contact sensor。** 球上的 `ContactSensor` 过滤 table/net/floor/blade
  四个刚体；PhysX 的 filtered pair 不提供接触点，Judge 会退回使用球心位置（既有接口本就允许
  `position=None`）。为了能被过滤，球台、球网和地面是 kinematic 刚体而非静态碰撞体。
- **空气动力学与 MuJoCo 同式。** 二次阻力与上限 0.35 的 Magnus 升力按 `physics.py` 的同一公式向量化。
- **场景只包含规则可见的物体**，不含装饰几何；测试用拍面是长方体而非 MuJoCo 的椭球，拍面边缘的接触时刻
  因此不具备逐步可比性。

### 实跑验证（2026-09-27）

在 Isaac Sim 5.0.0 + Isaac Lab 0.46.2、CPU PhysX 上实例化 300 个并行环境，`return-v1` dev 全集每个环境
一条球、拍面停放，逐控制步采样球状态，与 MuJoCo 3.13 的同一批球逐条比较
（[`scripts/backend_parity.py`](../scripts/backend_parity.py)，`make parity`）。完整结果见
[`reports/table-tennis-backend-parity.md`](../reports/table-tennis-backend-parity.md)：

| 量 | 中位数 | p95 |
|---|---:|---:|
| 首次落台前轨迹最大偏差 | 3.5 mm | 4.7 mm |
| 首次落台时间差 | 0 ms | 5 ms（= 一个采样周期） |
| 首次落台位置差 | 1.8 mm | 19 mm |
| 落台后 0.1 s 内最大偏差 | 78 mm | 219 mm |
| 反弹最高点高度差 | 50 mm | 102 mm |

Judge 判定一致率：`incoming_valid` 98.7%、`failure_reason` 99.3%。4 个不一致全部带 `deep` 标签——落点
贴着己方台端线，毫米级差异就会让界内变界外。

读法：**飞行段两个后端一致**（毫米级，残差来自积分器与 5 ms 采样）；**接触段不一致**，反弹高度差约
一成，是 MuJoCo 软接触与 PhysX 恢复系数/摩擦模型的差别。这是 M4 要标定的量，不是 bug；在它被标定
之前，两个后端上的回球类分数不可直接互比。

实跑中发现并修复的缺陷：

- **气动力被施加两次。** 物理回调里调用了 `ball.write_data_to_sim()`，而 `ManagerBasedRLEnv` 在每个
  物理步前本就会写入所有资产的外力，PhysX 会把同一步内的多次施加累加——阻力实际翻倍，0.3 s 时轨迹差
  32 cm、L0 判定一致率只有 62.5%。回调现在只缓存外力。
- **固定集参数漏传。** 环境按 `split` 加载固定集时没有传 `task=bank_resource`，任何非默认固定集的任务
  配置都会静默回退到 `return-v0`。
- 新增 `env.shot_queue`（按顺序派发指定球）与 `env.current_shots`，跨后端比较需要知道每个环境在打哪一条球。

**GPU PhysX 同样验证过**（`--device cuda:0`，同 300 条球，见
[`reports/table-tennis-backend-parity-gpu.md`](../reports/table-tennis-backend-parity-gpu.md)）：飞行段中位差
3.4 mm、首次落台位置差中位 1.5 mm、判定一致率 98.7% / 99.3%，与 CPU PhysX 几乎相同。PhysX 在 GPU 上不支持
CCD（启动时会提示忽略该设置），在这批球上没有出现穿透；更快的球（网球等）仍需单独确认。第一次尝试 GPU
时本机显存被其他训练任务占满，PhysX 无法分配内存——这是环境问题，不是代码问题。

### 机体：Franka Panda 的运动学一致性（2026-09-28）

Isaac 侧加载 Franka 官方 USD（`franka.usd`，由 `MULTISPORT_FRANKA_USD` 指定，不随仓库分发），与 MuJoCo 侧
Menagerie 的 Panda 在 50 组随机关节角下比较手部坐标系（`scripts/robot_kinematic_parity.py`，报告
[`reports/panda-kinematic-parity.json`](../reports/panda-kinematic-parity.json)）：

- 手部位置差最大 0.46 µm，7 个关节限位最大差 5.7e-8 rad——**两边是同一条运动学链**；
- 手部姿态差是一个在所有位形下都恒定的绕手部 z 轴 180° 旋转（各位形间偏差 5.9e-7）：两份资产对手部坐标系的
  约定不同，Isaac 侧装拍面时要把偏移量先转这 180°。
- Isaac Lab 默认初始位形不在 joint4 的限位内（0 ∉ [−3.07, −0.07]），必须显式给 ready pose 作为初始状态。

仍未完成：Isaac 侧的 Panda 回球环境本身（装拍、关节位置动作、同一 Judge、与 MuJoCo 的轨迹对比）；G1 与其他运动
的 Isaac 环境。

如果 Isaac 环境里 Isaac Lab 的可编辑安装指向的源码目录已被移动，可以把源码目录临时放到 `PYTHONPATH` 上：

```bash
make parity ISAAC_PYTHON=/path/to/isaac/python \
  ISAAC_PYTHONPATH=/path/to/IsaacLab/source/isaaclab
```

## 无头快速退出

部分同时装有 AMD 核显与 NVIDIA 独显的 Linux 工作站会在仿真完成后卡在 Kit 的 `Framework::unload_all_plugins`。无头 CLI 在输出和 USD 完全刷新后使用快速进程退出，避免 CI 永久挂起；GUI 模式仍执行标准 `SimulationApp.close()`。这不会跳过仿真、USD 写入或验证结果，只跳过进程结束前的插件卸载。
