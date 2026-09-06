# 传感器层与 Vision Track

**日期：2026-08-30**
**范围：`table-tennis-return-panda-v1` 的第二条观测轨道**

State track 把球的精确位置、速度和旋转直接从模拟器交给策略。真实球台上没有任何设备能提供这个，
所以 state track 的分数只衡量控制，不衡量感知。这份文档定义另一条轨道：**策略只能看见相机。**

设计与标定见 [`ROBOT_LAYER.md`](ROBOT_LAYER.md)，任务拆解见 [`TODO.md`](TODO.md)。

---

## 一、两条轨道的区别只有一项

| | state track | vision track |
|---|---|---|
| 球的状态 | 特权真值（位置/线速度/角速度） | **没有**——观测对象上不存在 `ball` 属性 |
| 本体感知 | 关节位置/速度、拍面位姿 | 同左 |
| 传感器 | 无 | 立体相机对 + 拍面 IMU / 接触 / 关节力矩 / frame transform |
| 机器人、动作、Judge、Shot Bank、reward、阈值 | 完全相同 | 完全相同 |

规则由构造保证而不是靠文档约定：`VisionTrackBackend` 包住机体后端，它的 `observe()` 返回
`VisionObservation`，这个 dataclass 根本没有 `ball` 字段。策略拿不到特权状态，不是因为不该读，
是因为读不到。

两条轨道的分数因此可以直接相减，差值就是感知的代价。

## 二、声明的传感器套件

`multisport_sim.benchmark.vision.TABLE_TENNIS_VISION_SENSORS`：

| 传感器 | 类型 | 速率 | 安装 | 参数 |
|---|---|---|---|---|
| `ball_camera_left` | camera | 120 Hz | 世界固定 | 320×240，fovy 70°，位于 `(-0.70, -2.10, 1.55)` |
| `ball_camera_right` | camera | 120 Hz | 世界固定 | 320×240，fovy 70°，位于 `(-0.70, +2.10, 1.55)` |
| `blade_pose` | frame_transform | 200 Hz | 世界 | 拍面 site 在世界系的位姿 |
| `blade_imu` | imu | 200 Hz | 拍面 body | 真加速度（含重力）+ 角速度 + 姿态 |
| `blade_contact` | contact | 200 Hz | 拍面 body | 语义类别 + 法向力 |
| `arm_torque` | joint_torque | 200 Hz | 拍面 body | 7 个关节的实际输出力矩 |

两条设计决定值得写下来：

**1. 基线 4.2 m，而不是一对近距离相机。** 在这个距离上一个 40 mm 球只占约 9 个像素，用视半径估计
深度的噪声远大于球本身；两条接近正交的射线相交则不受此影响。深度来自几何，不来自表观大小。

**2. 相机位姿由 spec 决定，不是从场景文件里抄的。** `build_panda_table_tennis_model(cameras=...)`
按 `CameraSpec` 往模型里加相机，所以报告里写的位姿和视场角就是实际渲染用的位姿和视场角。

除相机外的四个传感器是真实机械臂本来就有的本体感知。扣掉它们测的是另一个更难的问题，不是这个
benchmark 声称要测的问题。

## 三、速率由一处统一执行

`SensorSchedule` 决定哪些传感器到期。120 Hz 的相机在 200 Hz 控制循环下，中间那一步拿到的是
**上一帧**，并且带着**上一帧的时间戳**。忽略这个时间戳的策略犯的是硬件本来就会让它犯的错。

Gymnasium 观测里因此有一项 `frame_age_s`：最新一帧到当前控制步之间的间隔。

## 四、参考感知管线

`StereoBallTracker`：颜色分割 → 最大连通块 → 双射线三角化 → 最近若干次检测的最小二乘速度拟合。

颜色分割用三个判据，每个都不可省：

| 判据 | 排除什么 |
|---|---|
| `R > 110` | 棕色地板 |
| `R − B > 55` | 蓝色台面 |
| `R − G > 40`（绿蓝差） | **红色球拍和场地标记** |

第三条最容易被忽略：球拍在前两条上和球无法区分，去掉它以后测量误差从**亚厘米直接变成 2.2 m**，
因为红色球拍的像素数远多于球，会把质心整个拉过去。

三角化保留残差（两射线最近点间距），残差大于 15 cm 的检测直接丢弃：两条射线没有接近相交，说明
两个相机看的不是同一个东西，平均两个不同的东西比不报告更糟。

**实测（dev split 全部 12 球，540 个立体帧）：**

| 量 | 值 |
|---|---|
| 位置误差中位数 | **0.81 cm** |
| 位置误差 p90 | 1.02 cm |
| 位置误差最大 | 1.42 cm |
| 无检测帧比例 | **8.1%** |

这是一个下界而不是上界，任何提交都可以替换它。

## 五、实测分数：感知的代价

三条 state track 基线加一条 vision 基线，见
[`reports/table-tennis-panda-baselines.md`](../reports/table-tennis-panda-baselines.md)。
`vision` 用的是和 `intercept` **完全相同**的挥拍控制律，唯一区别是球的状态来自三角化而不是真值。

test split 主指标：

| Level | 主指标 | `intercept`（state） | `vision` | 门槛 |
|---|---|---|---|---|
| L0 | incoming_valid_rate | 100% PASS | 100% PASS | 100% |
| L1 | hit_rate | 100% PASS | **100% PASS** | 90% |
| L2 | valid_return_rate | 50% | 0%（命中 50%） | 80% |
| L3 | target_rate | 0%（回球 25%） | 0%（回球 25%） | 70% |
| L4 | worst_bucket | 33% | 0%（命中 75%、回球 25%） | 60% |
| L5 | worst_bucket | 0%（命中 75%） | 0%（命中 50%） | 50% |

读法：**拦截这件事视觉几乎不损失**（L1 两条轨道都满分），**回球开始损失**（L2 命中率从 100% 掉到
50%）。原因是可测的——8.1% 的丢帧和 0.26 m/s 的速度估计误差在需要拍面速度的击球时刻代价最大，而
这个机体本来就只有 7 ms 的拦截余量。

## 六、把你自己的视觉策略接进来

策略是任意 `obs -> action` 的可调用对象。vision track 上 `obs` 是 `VisionObservation`：
`time_s`、`robot`（本体感知）、`sensors`（传感器读数）。没有 `ball`。

```bash
python scripts/eval_policy.py \
  --policy my_pkg.eval:load_policy --policy-id my-vision-policy \
  --track vision --split test --levels all --out reports/my-vision-policy
```

```python
def load_policy():
    model = load_my_weights()

    def policy(obs):
        left = obs.sensors.camera("ball_camera_left").rgb    # (240, 320, 3) uint8
        right = obs.sensors.camera("ball_camera_right").rgb
        qpos = obs.robot.joint_positions                      # 7 个关节角
        return model(left, right, qpos)                       # 7 个关节位置设定值（rad）
    return policy
```

训练用 Gymnasium 环境 `MultiSportRobot/TableTennisReturn-Panda-Vision-v1`，观测是字典：

| key | 形状 | 说明 |
|---|---|---|
| `ball_camera_left` / `ball_camera_right` | `(240, 320, 3)` uint8 | 两路图像 |
| `joint_positions` / `joint_velocities` | `(7,)` float32 | 本体感知 |
| `blade_pose` | `(7,)` float32 | 拍面位置 + 四元数 |
| `frame_age_s` | `(1,)` float32 | 最新一帧的陈旧程度 |

训练和评测读的是**同一个** `vision.vision_observation_dict()`，所以策略被评测的观测就是它训练的观测。

内置参考基线：

```bash
multisport-benchmark --robot panda --track vision --level L1 --split dev
```

## 七、明确没有做的

| 项目 | 说明 |
|---|---|
| 观测噪声与延迟 | 基础渲染无噪声；`return-v1` 的 L5 按 manifest 加像素/状态噪声及 2 步观测、1 步动作延迟 |
| 域随机化 | L5 已有球质量、空气密度和相机外参扰动；光照、材质、纹理仍未随机化 |
| Isaac 侧实现 | `SensorSuite` 协议是后端无关的，Isaac 的 tiled renderer 实现尚未编写（无 GPU 环境验证） |
| 深度相机的使用 | `CameraSpec(depth=True)` 已实现并有测试，但声明的套件里没有开启 |
| 固定集规模 | Panda 默认 v1：dev 每级 50、test 每级 100；bucket 仍只有 14–68 条，报告逐 bucket 给 Wilson 区间 |
