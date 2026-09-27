# 结果包与完整指标

**日期：2026-08-30**
**范围：`BENCHMARK_SPEC` §7 的剩余指标与 §9 的提交产物**

终端里的一行分数不是结果。结果是**别人能核对的东西**：哪个 commit、哪份固定集、哪个策略、跑在什么
环境里、每一回合的原始结果，以及失败样本的录像——不只是成功的。

---

## 一、指标现在齐了

`BENCHMARK_SPEC` §7 要求八项原始指标。补齐前缺三项，现在全部在报告里：

| 指标 | 位置 | 说明 |
|---|---|---|
| `success_rate` | `metrics.*_rate` + `confidence_intervals_95` | Wilson 区间 |
| `target_error_m` | `metrics.mean_target_error_m` | 首次合法落点误差 |
| **`contact_error`** | `contact_error` | 见下 |
| `safety_violations` | `robot_metrics` | 五类违规，逐物理步检查 |
| `energy_joule` | `robot_metrics` | 执行器机械功积分 |
| **`episode_time_s`** | `metrics.mean_episode_time_s` | 每回合已有，现在也进汇总 |
| **`robustness_gap`** | `robustness_gap` | 见下 |
| **`inference_latency_ms`** | `inference_latency_ms` | 见下 |

### `contact_error`：拍面上的偏心距离

定义：**首次**球—拍接触点到拍面中心的距离，**只算拍面平面内的分量**。第三个分量是球半径加拍面
半厚，是个和瞄准无关的常数；平面内距离才是"离甜点多远"。

- 没有击球的回合报 `null` 而不是 0——0 表示"正中甜点"，而它根本没碰到球。
- 只取第一次接触。后面的接触是球离开拍面，把它们算进去会让瞄准看起来更准。
- 同时报 `blade_speed_mps`：接触瞬间的拍面速度，这是命中能否变成合法回球的关键量。

实测（`intercept` 基线，dev L2）：平均偏心 **3.9 cm**，最大 5.4 cm。拍面半轴是 8.5×10 cm，所以是
偏心但仍在拍上。

### `robustness_gap`：分布内减扰动集

定义：`valid_return_rate` 在 **L2–L3**（能产生成功的标称分布；L1 在击球瞬间结束、成功率恒为 0，只保留在逐级明细里）上的值，减去在 **L4–L5**（快球/旋转/边线/短长/
低平/留出组合）上的值。

- 单个 level 的报告里这项是 `null`：这个量本身就是两组之间的比较，用一组算出来的数是和空气比较。
- 跨 level 的运行（`scripts/eval_policy.py --levels all`、`make baselines`）才会给出它。
- `level_breakdown` 单列 L4 与 L5 的点估计、样本量和 Wilson 区间。L4 是分布变化；L5 重新分层采样并
  额外施加观测噪声、动作延迟和域随机化。两者不是同一批样本的逐级加难，经验成功率不保证单调。
- Panda 默认使用 `table_tennis/return-v1`，test 每级 100 球。L4/L5 的最弱 bucket 仍只有 14–68 球，
  因而 `assessment.bucket_confidence_intervals_95` 会逐 bucket 报 Wilson 区间；点估计不能脱离区间引用。

### `inference_latency_ms`：策略自己想了多久

`run_shots` 只对 `controller.act()` 计时——不含物理、不含渲染。报 `mean` / `p50` / `p95` / `max`。

这是**这台机器上**的数字，和报告的 `software` 块放在一起读。不比较硬件就比较两份提交的延迟没有意义，
所以这一项明确写了 `measured_on`。

## 二、结果包

```bash
python scripts/package_submission.py \
  --policy my_pkg:load_policy --policy-id my-sac-v3 \
  --split test --track vision \
  --policy-file weights/actor.pt --policy-file weights/critic.pt \
  --out submission/
```

或 `make submission POLICY=my_pkg:load_policy POLICY_ID=my-sac-v3`。产出：

```text
submission/
├── manifest.json     commit、任务与资产版本、后端、硬件、轨道、复现命令
├── config/           完整任务配置 + 本次运行参数
├── metrics.json      每回合原始结果（按 level）+ 各级汇总 + robustness_gap
├── policy/           权重（sha256 内容哈希）或 SOURCE.md 说明如何获取
├── reports/          每个 level 的完整 JSON/Markdown 报告
├── videos/           固定的成功**和**失败样本
└── environment.txt   Python、平台、MuJoCo、全部已安装依赖版本
```

四条设计决定：

**1. manifest 会承认工作区是脏的。** `repository.dirty` 和 `uncommitted_files` 如实记录。脏树意味着
"只看 commit 复现不出这个结果"——说出来才是这个字段的意义，藏起来 manifest 就成了假话。

**2. 录像的选择是规则，不是品味。** 按 shot_id 排序后取前 N 个成功和前 N 个失败。"不得只挑成功案例"
是关于**选择方式**的规则，排序取前几个就消除了挑选的余地。

**3. 没带权重的包会明说。** `policy/SOURCE.md` 写清楚这个包**不含**策略，而不是让它看起来很完整。
`policy.complete` 字段同样如实。

**4. 包完整不等于可上榜。** `leaderboard_eligible` 直接取自 Shot Bank manifest 的判断：建立在实验性
固定集上的包，无论多完整都不是排行榜条目。

录像需要可选依赖 Pillow（`uv pip install pillow`，或装 `submission` extra）。没装的话包照样生成，
manifest 里 `videos.status` 写 `skipped` 并给出原因——而不是悄悄产出一个看起来完整的包。

## 三、明确没有做的

| 项目 | 说明 |
|---|---|
| 训练配置 | `config/` 目前只有任务配置和评测参数；训练超参需要提交者自己放进去 |
| 大文件策略 | 权重按内容哈希记录，但没有实现 LFS / 外部存储的获取协议 |
| 多种子协议 | `BENCHMARK_SPEC` §8 要求 5 个训练种子、每条件 ≥100 episodes；打包器不强制，因为固定集本身还不够大 |
| 审计流程 | 谁来验、验什么、榜怎么开，都还没定 |
