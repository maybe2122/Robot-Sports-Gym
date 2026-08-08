# Isaac Sim / PhysX 后端

## 版本和启动顺序

已验证组合：

- Isaac Sim 5.0.0
- Isaac Lab 0.46.2
- Python 3.11
- CPU PhysX 与 RTX 5090 图形设备

`isaac_cli.py` 只在启动前导入 `AppLauncher`。`isaac_backend.py` 中的 `omni`、Isaac Lab simulation 和 PhysX 接口会在 SimulationApp 完成启动后才加载，符合 Isaac 扩展生命周期要求。

## 场景构造

`isaac_scene.py` 是不依赖 Omniverse 的场景描述层，生成带名称、位置、姿态、尺寸、颜色、碰撞和材质参数的 primitive 列表。它有普通 pytest 覆盖，因此不启动 Isaac 也能检查五类球、场地和器材是否齐全。

运行时 `isaac_backend.py` 将描述转换成：

- `CuboidCfg`：场地、场线、球台、球网、篮板；
- `CylinderCfg` / `CapsuleCfg`：球网柱、球门、篮圈、球拍框和拍线；
- `RigidObjectCfg + SphereCfg`：带显式质量、碰撞和 PhysX 材质的五类球；
- 羽毛球的 16 根羽毛、足球色块和篮球缝线作为动态球体的纯视觉子 prim。

campus 导出的 ASCII USD 约包含 2955 个 prim。所有五类球都具备 `UsdPhysics.RigidBodyAPI`、质量 schema 和子碰撞网格。

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

# 输出与 MuJoCo 相同 schema 的量化报告
multisport-isaac --headless --device cpu --scene tennis \
  --evaluate --report reports/isaac/tennis.json
```

`--duration 0` 在 GUI 中表示运行到用户关闭窗口；无头模式没有给时长时自动运行 5 秒。

Isaac Sim 每个 Kit 进程只评测一个单项场景，以避免重复创建 `SimulationContext`。`make evaluate-isaac` 会依次启动五个进程，并将结果聚合为 `reports/isaac-fidelity.json` 和 Markdown 报告。

## 无头快速退出

部分同时装有 AMD 核显与 NVIDIA 独显的 Linux 工作站会在仿真完成后卡在 Kit 的 `Framework::unload_all_plugins`。无头 CLI 在输出和 USD 完全刷新后使用快速进程退出，避免 CI 永久挂起；GUI 模式仍执行标准 `SimulationApp.close()`。这不会跳过仿真、USD 写入或验证结果，只跳过进程结束前的插件卸载。
