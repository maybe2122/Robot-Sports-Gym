# 策略接口：不同机器人的输入怎么给

一句话：**任务定字段，机体定尺寸，扁平向量是导出的、不是写死的。**

这份文档回答一个具体问题：`table-tennis-return-panda-v1` 的观测是 33 维，但 33 里有 14 维是
Franka Panda 的七个关节。换一台六轴臂就是 31 维，换一台双足就完全不是这个形状。如果把 33
当成"这个任务的观测维度"发布出去，第二台机器人接进来的那天，已经有人按 33 训练过策略了。

设计与实现见 [`observation.py`](../src/multisport_sim/benchmark/observation.py)，
测试见 [`tests/test_observation.py`](../tests/test_observation.py)。

---

## 一、三层拆分

观测里的每一个数只属于三类来源之一，混在一个匿名向量里是它们看起来一样的唯一原因。

| 层 | 谁决定 | 换机器人会变吗 | 例子 |
|---|---|---|---|
| 任务层 | 任务 | **不变** | `ball.position`、`ball.linear_velocity`、`ball.angular_velocity` |
| 机体层 | 机器人 adapter | **尺寸变，语义不变** | `robot.joint_position`（Panda 7、UR5 6）、`effector.position` |
| 传感层 | 轨道（track） | 按轨道变 | vision track 的两路 320×240 图像 |

`effector.*` 归在机体层但宽度固定：末端位姿永远是 3+4+3，无论手臂有几个关节。这一点后面很关键。

## 二、契约是一张 layout，不是一个维度

```python
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1 as task

layout = task.observation_layout()
layout.size                      # 33
layout.names                     # ('ball.position', ..., 'effector.linear_velocity')
layout.slice("robot.joint_position")   # slice(9, 16)
layout.field("robot.joint_position").element_names
#   ('rb_joint1', ..., 'rb_joint7')  —— 不用数偏移量
```

每个字段带自己的 `size`、`low`/`high`、`source`（`task` 还是 `robot`），扁平向量的边界由
`layout.bounds()` 拼出来，Gymnasium 的 Box space 由它派生。**没有任何一处再手写 33。**
`OBSERVATION_DIM` 现在是 `observation_layout().size`，`ACTION_DIM` 是 `joint_action.dof`。

layout 会进每份报告的 `rule_geometry.observation_layout`，所以读分数的人能看到策略当时到底
看的是什么——这是分数可审计的前提。

## 三、策略按名字要字段，不按下标

这是整套设计的收益所在：

```python
WANTED = ("ball.position", "ball.linear_velocity", "effector.position")

class MyPolicy:
    def __init__(self, layout):
        self.index = layout.view(WANTED)   # 这台机器人上这些字段的扁平下标
        ...
    def act(self, observation):
        return self.net(observation[self.index])
```

同一份策略代码接到六轴臂上，`layout.view(WANTED)` 返回的下标不同，**读到的数含义完全相同**，
不需要改一行、不需要重训。而要了 `robot.joint_position` 的策略就是诚实地绑定机体的，layout
把这件事说出来（`layout.robot_independent()` 列出不绑机体的字段）。

要一个这台机器人没有的字段会**当场报错**，而不是补零：

```
ObservationLayoutError: this robot does not publish 'robot.tactile';
it publishes ['ball.position', ..., 'effector.linear_velocity']
```

补零的向量能训练、能收敛、也能拿到分数，但那个分数不代表任何东西。这是本设计唯一一条硬规则。

## 四、跨机器人比较：两条轨道

一个任务可以发布两种动作空间，共享同一个 Judge、同一份固定集、同一批阈值：

| 轨道 | 观测 | 动作 | 维度随机体变吗 | 用途 |
|---|---|---|---|---|
| **关节空间** | 全部字段（含 `robot.joint_*`） | `dof` 维关节位置/速度/力矩设定值 | 会 | 单机体上的真实控制能力 |
| **末端空间** | 只用 `layout.robot_independent()` | `EffectorPoseCommand`（7 维位姿） | **不会** | 跨机体可比：同一份策略打分在多台机器人上 |

末端空间那条是"同一个策略在 Panda 和 UR5 上各得多少分"唯一能成立的口径。代价是它把 IK
和冗余度处理交给了 adapter，测的是任务能力而不是全部控制能力——这个取舍要写在报告里，不能
默认。`RobotAdapter.control_modes` 声明支持哪些，不支持的模式抛
`UnsupportedControlMode` 而不是静默降级。

## 五、命名与版本

- **任务 id 不带机体**：`table-tennis-return-*` 共享 Judge、固定集、L0–L5 阈值，所以不同机体的
  结果落在同一张表里，多一列 `robot` 而不是多一个 benchmark。
- **环境 id 带机体和轨道**：`MultiSportRobot/TableTennisReturn-Panda-v1`、
  `...-Panda-Vision-v1`。策略是对着环境 id 训练的，机体换了就必须是另一个 id。
- **字段名是发布词表**，不是自由字符串。两个任务都叫 `ball.position` 就必须指同一件事、同一个
  坐标系。新增字段是加词表条目，改一个字段的含义是破坏性变更。

## 六、加一台新机器人要做什么

1. 实现 `RobotAdapter`（有序 `joint_names`、`dof`、`effector_name`、`control_modes`、
   `safety_limits`）。
2. 给一份 `JointActionLimits`（关节名 + 每关节上下限 + 控制模式）。
3. 复制一个任务配置，换 `robot_id`、`joint_action`、`workspace`；
   `observation_layout()` / `OBSERVATION_DIM` / `ACTION_DIM` **自动跟着变**，不用改。
4. 用 [`scripts/calibrate_reachability.py`](../scripts/calibrate_reachability.py) 在**该任务实际
   打分的固定集**上重跑标定——底座、ready pose、击球区都从这里出。用错固定集标定过一次，代价见
   [`M2_STATUS.md`](M2_STATUS.md) 里记录的那次：命中率从 100% 掉到 63%，看起来像控制坏了，实际
   是标定对着另一个分布做的。
5. 注册 `TaskEntry`，环境 id 带机体名。

不需要动的：Judge、固定集、指标、阈值、报告 schema。这正是"同一张表"能成立的原因。

## L3 目标：任务信息，不是特权信息

L3 要求把球放进一个目标圆里。**目标和球场一样是任务的一部分**，策略有权知道。2026-09-28 之前没有任何
渠道把它交给策略——观测、`info`、`reset` 参数里都没有——于是只有读 `ShotSpec` 的参考夹具能拿 L3 分，
学习策略在 L3 上只能瞎猜。现在有三条渠道，内容相同：

| 渠道 | 形式 |
|---|---|
| Gymnasium `info["target"]` | `{"center": [u, v], "radius_m": r}`，无目标的级别为 `None`；`reset` 与每个 `step` 都给 |
| `multisport_sim.benchmark.wrappers.TargetObservation` | 在扁平观测后追加 `[present, u, v, r]` 四个数（无目标时全 0），给不读 `info` 的 RL 库用 |
| 离线评测 `PolicyController` | 策略的 `reset` 若接受 `target` 参数（或 `**kwargs`），会收到同一个字典；老策略的 `reset(seed=...)` 不受影响 |

`(u, v)` 在任务的放置平面里：回球任务与羽毛球发球是场地 `(x, y)`，足球射门是球门口 `(y, z)`，篮球投篮是
篮圈平面 `(x, y)`；每份固定集的 `manifest.targets` 写明了它是哪个平面。
