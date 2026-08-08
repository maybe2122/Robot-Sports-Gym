# Robot Sports Gym 机器人球类能力分级、自动发球与自动对打设计

## 结论与总体方案

**是的，建议分级，而且不要一上来就做“双机器人完整对打”。** 对你现在的 Robot Sports Gym，最合理的演进顺序是：

**物理可信度 → 单球拦截 → 合法回球 → 定点回球 → 鲁棒回球 → 连续回合 → 固定对手池比赛 → 人类/真实机器人比赛。**

机器人球类研究里，“碰到球”和“真正把球合法打回去”本来就应该是两个指标。例如 2025–2026 年的 PACE 人形乒乓球工作明确区分了 **Hit Rate** 和 **Success Rate**：前者只是球拍截击到球，后者要求球被合法打回对方球台；其仿真测试在超过 25 万次发球上分别统计这两个指标。citeturn6view3turn6view2 Google DeepMind 的竞技机器人乒乓球系统进一步把能力拆成不同低层技能，并最终通过对不同水平真人选手的正式比赛判断水平；机器人对初学者赢下全部比赛、对中级选手赢下 55%，但没有战胜高级组选手，因此作者将其定位在业余中级附近。citeturn2view3turn4search6

所以，对你的项目，我建议把 Benchmark 分成两条互相独立的轴：

| 轴 | 测什么 | 解决的问题 |
|---|---|---|
| **Shot Skill / 单球能力** | 给机器人固定分布的球，看它能否碰到、回过去、打准 | “机器人基本技术到底怎么样？” |
| **Game Skill / 对抗能力** | 连续对打、固定机器人对手、比赛 | “它真正会不会打球？” |

这两个维度最后可以产生一个“等级”，但**底层报告一定保留原始指标**。不要只输出一个 `score=83.6`，否则你不知道机器人到底是“碰不到高速球”，还是“能碰到但控球差”，还是“单球很强但连续恢复动作差”。

你的 README 已经明确说目前项目仍然是 **physics foundation**，而不是完成的 robot benchmark。这个定位其实非常合适。下一步不应马上堆一个 RL 算法，而应该先增加一个**与 MuJoCo / Isaac 后端解耦的 Benchmark Core**：

```text
Physics scenes
     │
     ▼
Shot Generator / Shot Bank
     │
     ▼
Robot Adapter ───────── Policy
     │
     ▼
Event Detector
(ball/racket/table/net/floor)
     │
     ▼
Sport Rule / Episode Judge
     │
     ▼
Metrics + JSON Report
     │
     ├── Single-shot Benchmark
     ├── Gymnasium Env
     ├── Isaac Lab Vector Env
     └── Rally / Match Benchmark
```

这样你的 `multisport-fidelity-v1` 继续负责：

> **“球和场地物理是不是靠谱？”**

而新的机器人 benchmark 负责：

> **“在这套物理条件下，机器人到底会不会打？”**

这两者必须分开。否则机器人失败时，很难知道到底是控制器失败还是接触模型失败。

机器人乒乓球已有研究也支持这种逐层评价方式。一类研究直接用发球机大量重复产生可控球路，例如 AIMY 开源三轮乒乓球发球机可以自动控制速度、方向、旋转和发射时间，其论文特别强调“可重复、可控、多样化球路”对于机器人训练和评价的重要性；另一类工作才进一步测试连续 rally 和正式比赛。citeturn8search0turn4search5

**因此，你现在最值得做的不是“先实现自动比赛”，而是先实现一个非常扎实的 `table_tennis_return-v0`。**

之后 tennis、badminton 基本复用这个框架。

## 机器人水平应该怎样分级

我建议不要直接定义“青铜/白银/黄金机器人”，而是先做**能力等级 + 对抗等级**。

下面这一套可以直接发展成你未来 `BENCHMARK_SPEC.md` 的核心。

| 等级 | 测试内容 | 主要变化 | 主要指标 | 推荐通过条件 |
|---|---|---|---|---|
| **L0 Physics** | 不放机器人，只发球 | 固定轨迹 | 轨迹/落点是否有效 | Shot Bank 全部合法 |
| **L1 Intercept** | 简单球，看能否碰到 | 中路、低速、低旋 | Hit Rate | ≥ 90% |
| **L2 Return** | 把球合法打回 | 横向位置、速度变化 | Valid Return Rate | ≥ 80% |
| **L3 Placement** | 打到指定区域 | 左/中/右目标 | Target Rate、落点误差 | ≥ 70% |
| **L4 Robustness** | 困难组合球 | 快球、旋转、边线、低球 | 分桶 Return Rate | ≥ 60% |
| **L5 Generalization** | 未见过的组合 | held-out 球路、物理随机化、噪声/延迟 | Worst-bucket、总体成功率 | ≥ 50–60% |

这里的数字不是国际标准，而是**我建议你的第一版 benchmark 门槛**；更重要的是分布必须版本化。一旦发布 `table-tennis-return-v1`，就不要随着算法进步偷偷修改发球范围，否则不同论文和不同机器人之间无法比较。

乒乓球尤其适合这样分级，因为难度本身就是多维的。DeepMind 在竞技系统的数据集中就把来球区分为 Fast、Normal、Slow、Topspin、No-spin、Underspin、Lob 等类别，例如其数据分析中把前向速度大于 7 m/s 视为 Fast，3.5–7 m/s 视为 Normal；旋转中大于 50 rad/s 属于 Topspin，低于 −25 rad/s 属于 Backspin。citeturn6view1 他们还发现机器人对不同旋转的表现差别很大：约 80 rad/s 的上旋仍能保持约 60% 回球率，而下旋增加后回球率迅速下降。citeturn6view1turn4search6

所以不要只把 difficulty 定义为：

```python
difficulty = ball_speed
```

应该至少保存：

\[
D =
(v,\;\omega,\;x_{\text{landing}},\;
z_{\text{hit}},\;TTC,\;
d_{\text{net}},\;d_{\text{edge}})
\]

也就是：

```text
速度
旋转
横向位置
预计击球高度
Time-to-contact
落点离网距离
落点离边线距离
```

以后 Vision Track 再加入：

```text
camera latency
observation noise
ball occlusion
frame rate
lighting variation
```

这也意味着你 README 中已经规划的：

```text
state
vision
robustness
sim-to-real
```

**不应该成为四套完全不同的任务。**

更好的关系是：

```text
                     L1   L2   L3   L4   L5
State Track           ✓    ✓    ✓    ✓    ✓
Vision Track          ✓    ✓    ✓    ✓    ✓
Robustness Track           ✓    ✓    ✓    ✓
Real Robot Track      ✓    ✓    ✓    ✓
```

例如：

```text
TableTennis-Return-State-L3-v1
TableTennis-Return-Vision-L3-v1
TableTennis-Return-Real-L3-v1
```

这样“机器人等级”和“传感条件”不会混在一起。

### 单球指标也必须拆开

以乒乓球为例，我建议一次 episode 输出至少：

```json
{
  "hit": true,
  "valid_return": true,
  "target_hit": false,
  "target_error_m": 0.184,
  "time_to_contact_s": 0.431,
  "outgoing_speed_mps": 6.21,
  "incoming_speed_mps": 5.77,
  "incoming_spin_radps": [48.2, -3.1, 6.2],
  "landing_xy": [0.31, 0.84],
  "failure_reason": null
}
```

这与现有机器人乒乓球研究的评价方式吻合：PACE 明确把 racket interception 和 valid return 分离；SpikePingpong 又把落点精度单独测量，例如按距离目标中心 30 cm 和 20 cm 的区域统计成功率。citeturn6view3turn4academia20 DeepMind 的竞技系统则进一步记录不同低层击球策略的 land rate、回球速度和落点分布，而不是只记录比赛胜负。citeturn1view0turn6view1

这才真正能回答“机器人水平是多少”。

比如两个机器人：

```text
Robot A
Hit Rate:           97%
Valid Return Rate:  61%
Target Rate:        28%

Robot B
Hit Rate:           86%
Valid Return Rate:  80%
Target Rate:        68%
```

A 的问题明显是**击球质量/控制**，B 的问题偏向**拦截覆盖范围**。

一个总分完全看不出这种差别。

## 自动发球与“机器人是否接住”的实现

你问的：

> “发一些球，看机器人是否可以接住”

实际上应该是你第一版机器人 benchmark 的核心，而且实现难度远低于完整对打。

这里我特别建议引入：

```text
Shot Generator
        ↓
Shot Bank
        ↓
Episode Runner
        ↓
Rule Judge
        ↓
Metrics
```

而不是直接在 `reset()` 里：

```python
ball_vel = np.random.uniform(...)
```

因为后者很难保证两台机器人或 MuJoCo / Isaac Sim 得到**完全相同的一组测试球**。

### Shot Bank

建议加入：

```text
benchmarks/
└── table_tennis/
    └── return-v1/
        ├── train.jsonl
        ├── dev.jsonl
        └── test.jsonl
```

一条记录：

```json
{
  "shot_id": "tt-return-l2-000314",
  "sport": "table_tennis",
  "level": "L2",
  "position": [0.10, 1.80, 1.10],
  "linear_velocity": [-0.35, -5.60, 1.62],
  "angular_velocity": [55.0, 4.0, 2.0],
  "tags": [
    "topspin",
    "forehand",
    "normal-speed"
  ]
}
```

这里我建议：

**训练集可以随机生成；正式 test set 必须固定。**

Gymnasium 本身支持通过 `reset(seed=...)` 建立确定性随机生成流程；当前 API 要求 `reset()` 返回 `(observation, info)`，而 `step()` 返回 observation、reward、`terminated`、`truncated` 和 info，这正好很适合你把“成功/失败”与“超时”分开。citeturn1view3

首先定义通用数据结构：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    sport: str
    level: str

    position: Vec3
    linear_velocity: Vec3
    angular_velocity: Vec3

    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BallState:
    position: npt.NDArray[np.float64]
    linear_velocity: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]


@dataclass
class EpisodeResult:
    shot_id: str

    hit: bool = False
    valid_return: bool = False
    target_hit: bool = False

    contact_time_s: float | None = None
    landing_xy: tuple[float, float] | None = None
    target_error_m: float | None = None

    failure_reason: Literal[
        "miss",
        "net",
        "own_side",
        "out",
        "floor",
        "timeout",
        "safety",
    ] | None = None
```

然后给所有后端一个统一接口：

```python
from typing import Protocol


class SportSimAdapter(Protocol):
    def reset(self) -> None:
        ...

    def launch_ball(self, shot: ShotSpec) -> None:
        ...

    def get_ball_state(self) -> BallState:
        ...

    def semantic_contacts(self) -> set[frozenset[str]]:
        """
        Example:
        {
            frozenset(("ball", "robot_racket")),
            frozenset(("ball", "table")),
        }
        """
        ...

    def step(self, action) -> None:
        ...

    @property
    def time(self) -> float:
        ...
```

这样上面的 Benchmark 完全不知道底下是：

```text
MuJoCo
Isaac Sim
真实机器人
```

这点对 Robot Sports Gym 尤其重要。

### MuJoCo 中怎样发球

MuJoCo 官方 Python API 可以通过 object name 查询模型对象，而运行时接触信息保存在 `mjData.contact` 中，因此非常适合做这种事件判定。citeturn5search0turn5search5

假设球有一个 free joint：

```xml
<body name="ball">
    <freejoint name="ball_free"/>
    ...
</body>
```

可以这样直接设置发射状态：

```python
import mujoco
import numpy as np


class MujocoBallLauncher:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ball_joint: str = "ball_free",
    ):
        self.model = model
        self.data = data

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            ball_joint,
        )

        if joint_id < 0:
            raise ValueError(f"Unknown ball joint: {ball_joint}")

        self.qpos_adr = model.jnt_qposadr[joint_id]
        self.qvel_adr = model.jnt_dofadr[joint_id]

    def launch(self, shot: ShotSpec) -> None:
        p = np.asarray(shot.position, dtype=np.float64)
        v = np.asarray(shot.linear_velocity, dtype=np.float64)
        w = np.asarray(shot.angular_velocity, dtype=np.float64)

        # free joint qpos:
        # [x, y, z, qw, qx, qy, qz]
        self.data.qpos[self.qpos_adr : self.qpos_adr + 3] = p
        self.data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7] = (
            1.0, 0.0, 0.0, 0.0
        )

        # free joint qvel:
        # [vx, vy, vz, wx, wy, wz]
        self.data.qvel[self.qvel_adr : self.qvel_adr + 3] = v
        self.data.qvel[self.qvel_adr + 3 : self.qvel_adr + 6] = w

        mujoco.mj_forward(self.model, self.data)
```

你当前已经实现了 sport-specific drag / spin / Magnus，所以 benchmark **不要自己重新计算飞行轨迹**。

只负责初始化：

\[
p_0,\;v_0,\;\omega_0
\]

随后全部交给你的现有 physics layer。

这会让架构非常干净：

```text
Benchmark:
initial conditions

Physics Sim:
trajectory + collision + aerodynamics

Robot:
action

Judge:
outcome
```

### Isaac Sim / Isaac Lab 中怎样做同一件事

Isaac Lab 当前 `RigidObject` API 提供了 `write_root_pose_to_sim()` 和 `write_root_velocity_to_sim()`；官方文档规定 pose 是位置加 `(w,x,y,z)` 四元数，velocity 是前三维线速度、后三维角速度。citeturn10search0turn10search1

因此同一个 `ShotSpec` 可以变成：

```python
import torch


def launch_isaac_ball(ball, shot: ShotSpec, device: str) -> None:
    pose = torch.tensor(
        [[
            *shot.position,
            1.0, 0.0, 0.0, 0.0,
        ]],
        dtype=torch.float32,
        device=device,
    )

    velocity = torch.tensor(
        [[
            *shot.linear_velocity,
            *shot.angular_velocity,
        ]],
        dtype=torch.float32,
        device=device,
    )

    ball.write_root_pose_to_sim(pose)
    ball.write_root_velocity_to_sim(velocity)
```

这正是为什么我建议 **Shot Bank 放在 backend 之上**。

同一条：

```json
tt-return-l3-001842
```

可以分别跑：

```text
MuJoCo
PhysX
Real launcher
```

然后你甚至可以报告：

```text
shot_id                 mujoco   physx   real
tt-return-l3-001842       PASS    PASS    PASS
tt-return-l3-001843       PASS    FAIL    FAIL
tt-return-l3-001844       PASS    PASS    FAIL
```

这会把你现有的“物理 fidelity benchmark”真正连接到“robot task benchmark”。

### 不要用“球离球拍很近”判断 Hit

正确做法应当监听真正 contact。

MuJoCo 可以直接读取 `mjData.contact`；较新的 MuJoCo 还增加了固定尺寸 contact sensor，官方明确将其定位于 learning agent input 和 environment logic。citeturn5search3turn5search18 Isaac Lab 也有 ContactSensor，可以按 rigid body / filtered body 读取接触力。citeturn5search2turn5search9

MuJoCo 最简单可以：

```python
def active_contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> set[frozenset[str]]:
    result: set[frozenset[str]] = set()

    for i in range(data.ncon):
        c = data.contact[i]

        name1 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            c.geom1,
        )
        name2 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            c.geom2,
        )

        if name1 is None or name2 is None:
            continue

        result.add(frozenset((name1, name2)))

    return result
```

实际项目中再把：

```text
ball_geom              → ball
robot_paddle_surface   → robot_racket
table_top_*            → table
net_*                  → net
floor                   → floor
```

映射成 semantic contact。

而且要做 **contact edge detection**：

```python
new_contacts = current_contacts - previous_contacts
```

否则一次持续 4 个 physics step 的球拍碰撞会被记成四次击球。

### 合法回球判定最好用状态机

对于乒乓球，不要写成：

```python
success = touched_racket
```

而应类似：

```text
INCOMING
   │
   ├── racket contact
   ▼
HIT
   │
   ├── crosses net
   ▼
CROSSED_NET
   │
   ├── opponent table contact
   ▼
SUCCESS
```

示例：

```python
class TableTennisJudge:
    def __init__(
        self,
        *,
        robot_racket_name: str = "robot_racket",
        timeout_s: float = 2.0,
    ):
        self.robot_racket_name = robot_racket_name
        self.timeout_s = timeout_s

        self.hit = False
        self.crossed_net = False
        self.done = False

        self.result: EpisodeResult | None = None

    def reset(self, shot_id: str) -> None:
        self.hit = False
        self.crossed_net = False
        self.done = False

        self.result = EpisodeResult(shot_id=shot_id)

    def update(
        self,
        *,
        ball: BallState,
        contacts: set[frozenset[str]],
        time_s: float,
    ) -> None:
        if self.done:
            return

        assert self.result is not None

        racket_contact = frozenset(
            ("ball", self.robot_racket_name)
        ) in contacts

        table_contact = frozenset(
            ("ball", "table")
        ) in contacts

        floor_contact = frozenset(
            ("ball", "floor")
        ) in contacts

        if racket_contact and not self.hit:
            self.hit = True
            self.result.hit = True
            self.result.contact_time_s = time_s

        # Example convention:
        # net is y == 0
        # robot is y < 0
        # opponent is y > 0
        if self.hit and ball.position[1] > 0.0:
            self.crossed_net = True

        if self.hit and table_contact:
            x, y = ball.position[:2]

            # User README table dimensions:
            # width = 1.525 m
            # length = 2.74 m
            inside_x = abs(x) <= 1.525 / 2
            opponent_y = 0.0 < y <= 2.74 / 2

            if inside_x and opponent_y:
                self.result.valid_return = True
                self.result.landing_xy = (float(x), float(y))
                self.done = True
                return

            if y < 0:
                self.result.failure_reason = "own_side"
                self.done = True
                return

        if floor_contact:
            self.result.failure_reason = (
                "miss" if not self.hit else "out"
            )
            self.done = True
            return

        if time_s >= self.timeout_s:
            self.result.failure_reason = "timeout"
            self.done = True
```

真正提交到项目时，table bounds、net plane、robot side 不应硬编码，而应该来自你现有：

```python
src/multisport_sim/specs.py
```

另外，rally 中球擦网以后仍可能形成合法回球，因此不要简单写：

```python
if net_contact:
    failure
```

规则判断应该由各 sport 的 `RuleEngine` 处理。

最后得到：

```python
hit_rate = hits / total
return_rate = valid_returns / total
```

这正好对应现代机器人乒乓球论文中常见的 hit / successful return 分离方式。citeturn6view3

## 自动对打应该怎样实现

“自动对打”其实有三种完全不同的东西，我建议你全部支持，但按复杂度逐级实现。

| 类型 | 对手是什么 | 是否真实碰撞 | 用途 |
|---|---|---:|---|
| **Synthetic Rally** | 程序生成下一颗球 | 否 | benchmark 首选 |
| **Scripted Physical Opponent** | 程序控制球拍 | 是 | 连续物理 rally |
| **Robot vs Robot** | 第二台机器人 + policy | 是 | 最终竞技测试 |

### Synthetic Rally 最值得先做

第一次不要真的模拟另一台机器人。

流程：

```text
Robot A 接到球
      ↓
合法回到对方场
      ↓
Judge 判 success
      ↓
Opponent Generator 选择下一颗来球
      ↓
重新 launch
      ↓
Robot A 再接
```

于是：

```python
rally_length = consecutive_successful_returns
```

例如：

```python
def run_synthetic_rally(
    adapter,
    policy,
    shot_bank,
    *,
    max_returns: int = 50,
) -> int:
    rally = 0

    for _ in range(max_returns):
        shot = shot_bank.next_for_rally(rally)

        adapter.reset_ball_only()
        adapter.launch_ball(shot)

        judge = TableTennisJudge()
        judge.reset(shot.shot_id)

        previous_contacts: set[frozenset[str]] = set()

        while not judge.done:
            obs = adapter.get_observation()
            action = policy(obs)

            adapter.step(action)

            contacts = adapter.semantic_contacts()
            new_contacts = contacts - previous_contacts
            previous_contacts = contacts

            judge.update(
                ball=adapter.get_ball_state(),
                contacts=new_contacts,
                time_s=adapter.time,
            )

        assert judge.result is not None

        if not judge.result.valid_return:
            break

        rally += 1

    return rally
```

严格来说，这不应该叫“真实物理连续对打”，应该在报告中命名：

```text
synthetic_rally_length
```

而不是：

```text
physical_rally_length
```

但它有一个巨大优点：

**对手不会成为额外变量。**

例如同一组 benchmark：

```text
Robot A: median rally = 14
Robot B: median rally = 8
```

说明差别基本来自被测机器人。

而不是：

```text
Robot A 可能碰巧遇到一个很弱的 Robot B policy
```

这一思想与已有机器人乒乓球的 cooperative rally 测试非常接近，只不过真实系统使用真人作为另一方。i-Sim2Real 就把 rally length 当作核心目标，其真实机器人最终达到平均 22 次连续击球、最高 150 次。citeturn4search5

你的 synthetic opponent 还可以根据 rally 逐渐变难：

```python
def difficulty_from_rally(rally: int) -> str:
    if rally < 3:
        return "L1"
    if rally < 8:
        return "L2"
    if rally < 15:
        return "L3"
    if rally < 25:
        return "L4"
    return "L5"
```

于是变成一个很有意思的指标：

```text
maximum sustained difficulty
```

### 第二阶段做真实物理 Scripted Opponent

如果你想真正看到：

```text
机器人 → 球 → 对面球拍 → 球 → 机器人
```

那么对面不一定需要完整机器人。

可以先做：

```text
kinematic opponent paddle
```

MuJoCo 很适合这个方案。官方 `mocap` body 可以通过 `mjData.mocap_pos` 和 `mjData.mocap_quat` 在运行过程中控制六维位姿，并继续与仿真对象发生交互；官方文档明确把这种机制描述为可以移动模拟对象并与其他模拟对象互动。citeturn9search0turn9search2

场景大概：

```xml
<body name="opponent_racket" mocap="true">
    <geom
        name="opponent_racket_geom"
        type="..."
        ...
    />
</body>
```

Opponent Planner 做三件事：

```text
预测球到达对方击球区的位置
            ↓
规划 opponent paddle impact pose
            ↓
执行短 swing trajectory
```

结构：

```python
class ScriptedOpponent:
    def update(self, ball_state, time_s):
        prediction = self.predict_intercept(ball_state)

        if prediction is None:
            return

        target = self.tactic.sample_target()

        strike = self.strike_planner.solve(
            incoming=prediction,
            desired_landing=target,
        )

        self.paddle_controller.execute(strike)
```

这里我不建议第一版就求解析的“球拍角度 → 精确落点”公式。

你已经有 MuJoCo / PhysX 接触模型、drag、spin 和 Magnus，更稳妥的方法是：

```text
candidate racket pose/velocity
             ↓
短时间 forward simulation
             ↓
预测 landing point
             ↓
优化
```

即：

\[
u^*
=
\arg\min_u
\left\|
p_{\text{landing}}(u)-p_{\text{target}}
\right\|^2
+
\lambda C(u)
\]

其中：

```text
u =
racket position
racket orientation
racket linear velocity
racket angular velocity
```

这是一个小型 shooting / MPC 问题。

你甚至可以离线生成：

```text
incoming ball state
    →
optimal paddle state
```

再训练一个简单网络作为：

```python
opponent_striker(incoming_state) -> racket_impact_state
```

这样 runtime 就快很多。

已有乒乓球研究确实经常依赖精确轨迹预测来解决 interception；一项结合物理模型和数据学习的工作在真实机器人上实现过 29/30 的回球表现，并特别强调旋转初态对于长期轨迹预测的重要性。citeturn8academia14

### 最后才做 Robot vs Robot

完整对打可以复用同一个 policy interface：

```python
action_a = policy_a(obs_a)
action_b = policy_b(obs_b)

env.step({
    "player_a": action_a,
    "player_b": action_b,
})
```

比赛层再增加：

```text
serve possession
score
game
match
side switching
timeout
safety stop
```

但这时不要再用：

```text
return_rate
```

判断唯一水平。

应该报告：

```text
point win rate
game win rate
match win rate
average rally
serve return rate
first-three-ball win rate
failure breakdown
```

DeepMind 的机器人乒乓球实验很有参考价值：他们最终不是靠单球成功率宣称“human-level”，而是让 29 名从 beginner 到 tournament-level 的未见真人选手实际比赛；与此同时仍保留 return、spin 和不同技能策略等诊断指标。citeturn0search1turn2view3

这里还可以进一步引入一个**固定 opponent pool**：

```text
Opponent-A: safe / slow
Opponent-B: topspin
Opponent-C: underspin
Opponent-D: fast attacker
Opponent-E: corner targeting
Opponent-F: mixed/adaptive
```

然后得到：

```text
            A      B      C      D      E      F
Robot X    93%    84%    61%    48%    53%    55%
Robot Y    88%    80%    76%    62%    69%    65%
```

甚至再计算 Elo / Bradley–Terry rating。

但是 **rating 只做 summary，不替代这个矩阵。**

因为一个机器人可能很擅长快球但完全怕下旋。

DeepMind 的真实实验就发现这种明显的 skill-specific weakness，而且高级真人选手会在比赛过程中主动寻找并利用机器人对下旋、快球、低球和近网球的弱点。citeturn4search6

这正是“对打 benchmark”比单球 benchmark 更高级的原因：

> 单球测试检查 capability；
> 对抗测试检查 opponent exploitation 下的 capability。

## 代码架构与 Gymnasium / Isaac Lab 集成

按照你的现有仓库，我建议不要把这些东西塞进 `scenes/`。

可以新增：

```text
src/multisport_sim/
├── benchmark/
│   ├── __init__.py
│   ├── types.py
│   ├── shot_bank.py
│   ├── runner.py
│   ├── metrics.py
│   │
│   ├── rules/
│   │   ├── base.py
│   │   ├── tennis.py
│   │   ├── table_tennis.py
│   │   ├── badminton.py
│   │   ├── football.py
│   │   └── basketball.py
│   │
│   ├── opponents/
│   │   ├── synthetic.py
│   │   ├── scripted.py
│   │   └── policy.py
│   │
│   └── backends/
│       ├── base.py
│       ├── mujoco.py
│       └── isaac.py
│
├── envs/
│   ├── table_tennis_return.py
│   ├── tennis_return.py
│   └── ...
│
└── specs.py
```

再增加：

```text
benchmarks/
├── table_tennis/
│   └── return-v1/
│       ├── manifest.json
│       ├── dev.jsonl
│       └── test.jsonl
└── ...
```

其中最重要的原则是：

```text
rules != rewards
```

**Rules 定义成功与失败；Rewards 只是帮助 RL 学习。**

例如真实评价：

```python
success = (
    racket_hit
    and crossed_net
    and first_landing_on_opponent_table
)
```

这个不能因为训练算法不同而变化。

但 reward 可以：

\[
r =
0.2r_{\text{approach}}
+1.0r_{\text{hit}}
+3.0r_{\text{return}}
+1.0r_{\text{target}}
-0.1r_{\text{jerk}}
-1.0r_{\text{safety}}
\]

这么做有很好的研究依据。PACE 的消融实验发现，如果只给“球有没有成功落到对方桌面”这种稀疏反馈，policy 很难学习到合适的回球速度和球拍姿态，因此他们加入了 hit guidance 和 return guidance。citeturn6view2 DeepMind 的训练 reward 同样不只包括落台，还包含击球、速度/加速度/jerk、安全碰撞、球拍高度和 style 等项。citeturn6view1

但正式 benchmark 最终仍只看：

```text
Hit Rate
Return Rate
Target Rate
Landing Error
Safety Violations
Rally Length
Match Win Rate
```

而不是 RL reward。

### Gymnasium 环境可以非常薄

你 README 里说 versioned Gymnasium env 仍未完成，这正好可以用上述 Benchmark Core 实现，而不是再写一套判断代码。

Gymnasium 官方 API 当前要求自定义环境实现 `reset()` 和 `step()`；`terminated` 表示任务自身结束，而 `truncated` 可用于超时等外部终止条件。citeturn1view3

因此：

```python
import gymnasium as gym


class TableTennisReturnEnv(gym.Env):
    def __init__(
        self,
        adapter,
        shot_bank,
        policy_observation_builder,
    ):
        super().__init__()

        self.adapter = adapter
        self.shot_bank = shot_bank
        self.obs_builder = policy_observation_builder

        self.judge = TableTennisJudge()

        # Define these from the robot adapter.
        self.observation_space = ...
        self.action_space = ...

        self._prev_contacts = set()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.adapter.reset()

        level = (
            options.get("level", "L2")
            if options
            else "L2"
        )

        shot = self.shot_bank.sample(
            level=level,
            rng=self.np_random,
        )

        self.adapter.launch_ball(shot)

        self.judge.reset(shot.shot_id)
        self._prev_contacts.clear()

        obs = self.obs_builder(self.adapter)

        info = {
            "shot_id": shot.shot_id,
            "level": shot.level,
            "tags": shot.tags,
        }

        return obs, info

    def step(self, action):
        self.adapter.step(action)

        contacts = self.adapter.semantic_contacts()
        new_contacts = contacts - self._prev_contacts
        self._prev_contacts = contacts

        self.judge.update(
            ball=self.adapter.get_ball_state(),
            contacts=new_contacts,
            time_s=self.adapter.time,
        )

        result = self.judge.result
        assert result is not None

        reward = self._training_reward(result)

        terminated = (
            self.judge.done
            and result.failure_reason != "timeout"
        )

        truncated = (
            self.judge.done
            and result.failure_reason == "timeout"
        )

        obs = self.obs_builder(self.adapter)

        info = {
            "hit": result.hit,
            "valid_return": result.valid_return,
            "landing_xy": result.landing_xy,
            "failure_reason": result.failure_reason,
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
```

于是你的 Benchmark Runner 和 RL environment 使用**同一个 Rule Judge**：

```text
               ┌── benchmark CLI
Shot Bank ─────┤
               └── Gymnasium reset()

Physics ───────── Sport Rule Judge
                      │
                      ├── benchmark metrics
                      └── Gym reward/termination
```

不会发生：

```text
Benchmark 认为成功
RL Env 认为失败
```

这种很常见但很难排查的问题。

### Isaac Lab 也不应该重新写一套逻辑

Isaac Lab 当前官方推荐的 `ManagerBasedRLEnv` 把 observation、action、reward、termination、event、curriculum、command 等功能拆成可配置 manager；官方尤其建议通过 `ManagerBasedRLEnvCfg` 描述 task，而不是去修改 base class。citeturn1view1turn11search14

这非常适合你的项目：

```text
Shot Generator
    →
CommandManager

Ball reset
    →
EventManager

robot / ball state
    →
ObservationManager

hit + legal return + target
    →
RewardManager

success / miss / timeout
    →
TerminationManager

L1 → L2 → L3 → L4
    →
CurriculumManager
```

Isaac Lab 官方也专门提供 curriculum mechanism，可以在训练过程中逐步修改 reward 或环境参数。citeturn1view2

你就可以定义：

```python
@configclass
class TableTennisCurriculumCfg:
    ball_difficulty = CurriculumTermCfg(
        func=increase_ball_difficulty,
        params={
            "initial_level": 1,
            "max_level": 5,
        },
    )
```

训练逻辑可以是：

```text
L1 success > 90%
        ↓
扩大 lateral range
        ↓
L2

L2 success > 85%
        ↓
增加速度 range
        ↓
L3

L3 success > 80%
        ↓
加入 spin
        ↓
L4

L4 success > 70%
        ↓
noise + latency + contact randomization
        ↓
L5
```

这与机器人乒乓球研究里已经被验证过的 curriculum 思路一致。较早的 model-free RL 机器人乒乓球工作就报告了利用适当 curriculum 和 reward 后，在广泛来球上获得约 80% return rate；更近的工作同样从较简单情况逐步增加来球复杂度。citeturn4academia23turn4academia21

### Isaac 后端尤其适合大规模自动发球

Isaac Lab 当前 `ManagerBasedRLEnv` 本身就是面向 vectorized environment 设计的，`num_envs` 指定并行子环境数量，observation 和 action 都以 batch 形式传递；`InteractiveScene` 也会根据 `num_envs` 克隆场景。citeturn11search0turn11search1

这意味着未来可以：

```text
env 0    L1 center
env 1    L1 center
env 2    L2 forehand
env 3    L2 backhand
...
env 511  L5 underspin
```

一次跑几百个发球测试。

从 benchmark 的角度这非常重要。

MuJoCo 先负责：

```text
容易调试
精细分析
reference implementation
```

Isaac Lab 再负责：

```text
parallel training
large-scale evaluation
```

这比让 MuJoCo 和 Isaac 各自发展一套任务定义更合理。

## 评测协议、自动报告与推荐落地顺序

真正把项目变成可引用 benchmark，最后一个关键不是 RL，而是**评测协议稳定性**。

我建议一次正式单球等级测试至少使用约 **1,000 个固定 test shots**。

原因很直观。假设机器人观察到：

```text
80 / 100 = 80%
```

二项成功率的 Wilson 95% 区间大约仍为：

```text
71.1% – 86.7%
```

不确定性相当大。

而：

```text
800 / 1000 = 80%
```

区间大约缩至：

```text
77.4% – 82.4%
```

所以可以设计成：

```text
开发:
200 shots / level

正式 benchmark:
1000 shots / level

论文/leaderboard:
固定 test split
```

而且不能只报告 aggregate：

```text
Return Rate = 73%
```

必须同时报告：

| Bucket | Return |
|---|---:|
| Center | 95% |
| Forehand | 84% |
| Backhand | 77% |
| Fast | 58% |
| Topspin | 69% |
| Underspin | 31% |
| Short | 62% |
| Deep | 81% |

这类分桶尤其重要，因为真实机器人已有研究反复发现，整体平均成功率很容易掩盖对特定旋转、速度和来球区域的巨大弱点。DeepMind 的比赛分析就是通过 spin-specific return rate 和真人策略反馈发现机器人明显怕强下旋。citeturn4search6turn6view1

正式报告我建议变成：

```json
{
  "benchmark": "multisport-robot-benchmark-v1",
  "task": "table-tennis-return-v1",
  "backend": "mujoco",
  "robot": "example-arm-v1",
  "policy": "policy-sha256:...",
  "seed": 12345,

  "episodes": 1000,

  "metrics": {
    "hit_rate": 0.921,
    "valid_return_rate": 0.784,
    "target_rate": 0.631,
    "mean_target_error_m": 0.217,
    "safety_violation_rate": 0.002
  },

  "buckets": {
    "fast": {
      "episodes": 150,
      "valid_return_rate": 0.613
    },
    "underspin": {
      "episodes": 150,
      "valid_return_rate": 0.427
    }
  },

  "failures": {
    "miss": 109,
    "net": 31,
    "own_side": 26,
    "out": 45,
    "timeout": 5
  }
}
```

然后生成 Markdown：

```text
Table Tennis Return v1

Level passed: L3

Hit rate             92.1%
Valid return rate    78.4%
Target rate          63.1%
Safety violation      0.2%

Weakest buckets
Underspin             42.7%
Fast                  61.3%
Backhand              68.9%
```

你的现有：

```bash
make evaluate
make evaluate-isaac
```

可以继续用于 physics。

增加：

```bash
make benchmark
make benchmark-isaac
```

以及 CLI：

```bash
multisport-benchmark \
  --sport table-tennis \
  --task return \
  --level L2 \
  --episodes 1000 \
  --robot my_robot \
  --policy checkpoint.pt \
  --report reports/robot/tt-l2.json
```

连续 rally：

```bash
multisport-benchmark \
  --sport table-tennis \
  --task rally \
  --opponent synthetic \
  --episodes 200
```

真正物理 scripted opponent：

```bash
multisport-benchmark \
  --sport table-tennis \
  --task rally \
  --opponent scripted
```

最终：

```bash
multisport-benchmark \
  --sport table-tennis \
  --task match \
  --player-a policy_a.pt \
  --player-b policy_b.pt
```

### 五项运动不要强行使用完全相同的“接球”定义

你的 benchmark core 可以共用，但每项运动的 skill progression 应不同：

| Sport | 基础能力 | 高级能力 | 最终交互 |
|---|---|---|---|
| Tennis | intercept → legal return | target / spin / speed | rally / match |
| Table tennis | hit → valid return | placement / spin | rally / match |
| Badminton | contact → over-net return | placement / smash defense | rally / match |
| Football | intercept / receive | control + pass/kick target | passing / game |
| Basketball | receive/control | pass / shoot / rebound | possession/game |

特别是 badminton，shuttle 的飞行模型和球类差别很大，你现有 orientation-dependent drag 和 stabilizing torque 正好可以产生很有价值的难度维度；但其 `RuleEngine` 不能照搬乒乓球的 bounce state machine。

所以最终抽象应该是：

```python
class SportRule:
    def reset(self) -> None:
        ...

    def update(
        self,
        state,
        contacts,
    ) -> None:
        ...

    def metrics(self) -> dict:
        ...
```

而不是：

```python
class BallGameRule:
    # one rule fits everything
```

### 最推荐的实际开发顺序

你现在的 README 已经强调：

> 还没有 robot assets、versioned Gymnasium environments、vectorized Isaac Lab environments、fixed evaluation splits 和 reference policies。

因此我建议下一阶段严格按这个依赖关系推进：

```text
当前
Physics foundation
      │
      ▼
Benchmark Core
ShotSpec / ShotBank / EventJudge
      │
      ▼
TableTennis-Return-v0
scripted paddle 先验证 Judge
      │
      ▼
RobotAdapter
接入第一台真实 robot asset
      │
      ▼
TableTennis-Return-v1
固定 dev/test split
      │
      ├─────────────┐
      ▼             ▼
Gymnasium Env    Isaac Lab Env
      │             │
      ▼             ▼
Reference policy / RL baseline
      │
      ▼
Synthetic Rally
      │
      ▼
Physical Scripted Rally
      │
      ▼
Policy Opponent Pool
      │
      ▼
Robot-vs-Robot Match
      │
      ▼
Real Robot / Human Evaluation
```

**这里最关键的节点是 `TableTennis-Return-v1`，不是 Robot-vs-Robot。**

只要这个任务做扎实，你就已经可以非常严谨地回答：

> “这个机器人会不会打乒乓球？”

因为报告会告诉你：

```text
它能不能碰到球
        ↓
碰到以后能不能回去
        ↓
能不能控制落点
        ↓
怕什么速度
        ↓
怕什么旋转
        ↓
覆盖多大区域
        ↓
在噪声和延迟下还能不能打
```

随后再回答更高级的问题：

> “它能不能持续打？”

用：

```text
synthetic rally length
physical rally length
```

最后才回答：

> “它能打到什么人类水平？”

用：

```text
fixed opponent pool
robot-vs-robot matches
human skill groups
real-world matches
```

已有研究很好地说明了为什么最后一步必须单独存在。DeepMind 的机器人单项技能本身已经拥有不同的落台率、回球速度和目标策略，但真正的“水平”仍然需要让它面对不同水平的人类、统计 point/game/match 结果，并观察对手是否能够主动利用它的弱点。citeturn1view0turn2view3

**因此，对你的 Robot Sports Gym，我会把下一版 benchmark 的核心定义成：**

```text
multisport-robot-benchmark-v1

Core metrics
────────────
Hit
Valid Return
Target Accuracy
Robustness
Safety

Interaction metrics
───────────────────
Synthetic Rally
Physical Rally
Point Win Rate
Match Win Rate

Tracks
──────
State
Vision
Robustness
Sim-to-Real

Difficulty
──────────
L1 → L5
```

其中：

> **“自动发一些球，看机器人能不能接住”应该成为标准化 Single-Shot Benchmark；
> “自动对打”应该建立在它之上，而不是替代它。**

这是目前最适合你这个项目阶段、也最容易进一步发展成可重复机器人球类 benchmark 的路线。
