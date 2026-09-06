# 固定集：规模、生成与 L5 扰动

**日期：2026-08-30**
**范围：`BENCHMARK_SPEC` §8 的评测协议**

`return-v0` 是手写的：dev 12 条、test 24 条，每级 2–4 球。它足以验证规则引擎、报告链路和机器人适配层，
但**远不足以支撑任何关于策略的结论**——4 个 episode 的 95% Wilson 区间宽约 60 个百分点。

规范要求：每个测试条件 ≥100 episodes，train/dev/test 种子互不重叠。这份文档说明现在有哪些固定集、
它们是怎么生成的，以及 L5 的扰动到底扰动了什么。

---

## 一、现有固定集

| 资源 | split | 每级 | 总数 | 说明 |
|---|---|---|---|---|
| `table_tennis/return-v0` | dev / test | 2 / 4 | 12 / 24 | **冻结**。规则引擎的回归夹具，摘要永不改动，历史分数因此全部有效 |
| `table_tennis/return-v1` | train / dev / test | 200 / 50 / 100 | 1200 / 300 / 600 | 统计充分；L5 声明真实扰动 |
| `tennis/return-v0` | train / dev / test | 200 / 50 / 100 | 1200 / 300 / 600 | 网球任务的首个固定集，见 [`TENNIS.md`](TENNIS.md) |

选哪一份由命令行决定：mocap 回归夹具默认冻结的 v0，Panda 机体任务默认统计充分的 v1：

```bash
multisport-benchmark --robot panda --level L2 --split test
python scripts/eval_policy.py --policy my_pkg:load_policy --bank table_tennis/return-v1 \
  --split test --levels all
```

**v0 一个字节都没改。** 报告里记录的 `shot_bank_digest` 因此仍然对得上，任何在 v0 上产生过的分数
都不会因为新增固定集而失效。

### 不同任务的发射方向

早期 return bank 固定为 opponent side `x>0`、`vx<0`。发球/投篮任务不能伪装成来球：新 manifest 可在
`coordinate_system` 声明 `shot_origin`（`robot_side` / `opponent_side`）和 `initial_motion`
（朝任一侧，或“静止或朝该侧”）。loader 同时验证起点与速度符号；两个字段必须一起出现。未声明字段的
旧 return manifest 继续走原有严格规则，因此其 digest 和行为都不变。

## 二、生成方式

`scripts/generate_shot_bank.py`（`make shot-bank`）。两条性质比数量更重要：

**1. 每条球都是仿真验证过的，不是算出来的。** 候选球在真实 MuJoCo 场景里发射，只有 Judge 判定
`incoming_valid` 才保留。空气阻力和 Magnus 力让弹道估算和真实轨迹的差别大到——手推的固定集里会混进
根本无法合法回击的球。网球在 L5 速度下的接受率只有约 29%，这些被拒的球用公式是看不出来的。

**2. 标签来自球实际做了什么。** `short` / `deep` / `edge` 读的是**实测第一落点**，不是采样参数。
所以同一个 bucket 在不同 split 里含义相同。速度/横向/旋转标签来自采样值，因为那本来就是发球设定。

**分层保证 bucket 覆盖。** L4/L5 的通过标准是"最差 bucket 的回球率"——某个 bucket 空了，这一级就
**根本无法评估**。所以每个 pass_bucket 都有对应的采样条件，而且条件声明 `require_any`：测得的标签
不含要求的 bucket 就重采，直到出现为止。覆盖是构造出来的，不是碰运气。

**种子不重叠。** 每个 split 的种子来自 `sha256("multisport-shot-bank-v1:{sport}:{split}")`，写在
manifest 的 `generation.split_seeds` 里。测试直接断言三个 split 的**球本身**互不相同，不只是种子不同。

## 三、还差多少

| 条件 | 规范要求 | 现状 |
|---|---|---|
| 每级 test episodes | ≥100 | **✅ 100** |
| train split | 需要 | **✅ 1200 条** |
| 种子不重叠 | 需要 | **✅ 并有测试断言** |
| 每 **bucket** episodes | ≥100 | ❌ 14–68：**bucket 级结论仍不可发表** |
| 5 个训练种子 | 需要 | ❌ 属于提交方，工具不强制 |

## 四、L5 扰动

v0 把最难的一级叫 "Generalization"，然后在 manifest 里承认
`domain_randomization` / `observation_noise` / `latency` 三项都是 `not_implemented_in_v0`。
只有留出轨迹的泛化不是泛化，是在稍宽一点的分布上做插值。

`return-v1` 的 L5 声明了真实条件（`multisport_sim.benchmark.perturbations`）：

| 条件 | 参数 |
|---|---|
| 观测噪声 | 球位置 σ=6 mm、速度 σ=0.15 m/s、旋转 σ=2 rad/s、关节位置 σ=2 mrad、相机 σ=4 个灰度级 |
| 延迟 | 观测 2 个控制步、动作 1 个控制步（200 Hz 下是 10 ms 和 5 ms） |
| 域随机化 | 球质量 ±4%、空气密度 ±10%（等价于阻力与 Magnus 同步缩放）、相机安装位置 σ=10 mm |

三条设计原则：

**1. 扰动属于任务，不属于运行器。** 施加什么由 Shot Bank 的 manifest 决定，所以同一个 bank 同一级的
两次运行扰动方式相同，报告也能如实写出策略到底承受了什么。v0 的 manifest 里那三个
`"not_implemented_in_v0"` 字符串被解释为"什么都不加"，所以 **v0 依然一点扰动都没有**。

**2. 由 episode 种子决定，可以精确重放。** 同一条命令跑两次，结果逐字节相同（有测试）。

**3. 只扰动策略看到的东西，绝不扰动 Judge 看到的东西。** 观测上的噪声不能改变"球是否过网"。Judge 直接
读后端，只有 `PerturbedBackend.observe()` 被拦截——这是"任务更难了"和"评分算错了"的区别。

延迟分两个方向，因为它们的失败方式不同：观测延迟意味着策略瞄准的是球**曾经**在的地方；动作延迟意味着
手臂在策略决定之后才开始动。补偿了其中一个不代表能扛住另一个。

域随机化只随机化真实部署里真会变的量：球质量在 ITTF 公差带内、空气密度、以及相机安装位置。
**球台/球场几何不随机化**——那是规定尺寸的器材，动了它就等于改变了 Judge 对"合法回球"的定义。

实测落差（同一条基线，L4 无扰动 → L5 有扰动，两级的分布也不同，所以这是两者的合计）：

| 任务 | L4 命中 / 回球 | L5 命中 / 回球 |
|---|---|---|
| `table-tennis-return-panda-v1`（`intercept`，test 每级 100 球） | 52% / 12% | 24% / 6% |
| `tennis-return-v0`（`scripted`，test 每级 100 球） | 90% / 56% | 80% / 46% |

## 五、明确没有做的

| 项目 | 说明 |
|---|---|
| bucket 级样本量 | 每 bucket 14–68 条，距离 100 还差一截 |
| 更强的视觉域随机化 | 光照、材质、纹理都没有随机化，只有相机外参 |
| 执行器噪声 | 只扰动观测与延迟，没有扰动力矩输出 |
| 扰动与分布的解耦 | L5 同时换了分布**和**加了扰动，所以"L4→L5 的落差"里两者的贡献分不开。要分开需要在同一批球上跑有扰动/无扰动两次，工具支持但还没跑 |
