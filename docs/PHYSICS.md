# 物理模型与标准来源

## 坐标和积分

- 全部输入使用 SI 单位：米、千克、秒、牛顿。
- 世界坐标 `+Z` 向上，重力为 `9.81 m/s²`。
- MuJoCo 使用 1 ms 固定步长和 `implicitfast` 积分器。
- Isaac Sim 使用 240 Hz 固定步长、PhysX TGS 求解器与连续碰撞检测（CCD）。
- 球体使用六维接触约束，分别设置滑动、扭转和滚动摩擦。

## 球与地面/球桌回弹

MuJoCo 的 `solref=(timeconst, dampratio)` 描述柔性接触动态，而不是直接输入恢复系数。本项目先以规则给出的质量、半径和落下高度建立模型，再在固定 1 ms 步长下校准 `solref`。`tests/test_bounce.py` 每次真实执行落球并检查第一次反弹最高点，避免“配置看似合理、运行结果不合理”。

当前自动验证目标：

| 接触 | 落下高度（球底到表面） | 首次回弹（球底到表面） |
|---|---:|---:|
| 网球—硬地 | 2.54 m | 1.41 m（容差 ±0.08 m） |
| 乒乓球—球台 | 0.30 m | 0.23 m（容差 ±0.025 m） |
| 篮球—木地板 | 1.80 m | 1.06 m（容差 ±0.04 m） |
| 足球—草地 | 2.00 m | 0.85–1.15 m（模型校准范围） |
| 羽毛球软木—地面 | 1.80 m | 小于 0.06 m |

场地表面使用不同摩擦与滚阻；乒乓球桌拥有独立于周围地面的接触参数。

Isaac 后端用 PhysX 恢复系数和 `max` 恢复系数组合规则实现同一目标，并提供 `--drop-test` 实测入口。在当前 Isaac Sim 5.0/CPU PhysX 验收中：

- 网球：2.54 m 落下，首次回弹 1.408 m。
- 乒乓球：球桌上方 0.30 m 落下，首次回弹 0.244 m。
- 篮球：1.80 m 落下，首次回弹 1.059 m。

## 真实度评分方法

`multisport-fidelity-v1` 对每项实测值使用以下可复现评分：

```text
tolerance_ratio = abs(measured - ideal) / tolerance
score = 100 * exp(-ln(2) * tolerance_ratio^2)
```

参考区间中点得到 100 分，落在接受边界时得到 50 分，超出区间即 FAIL。报告同时保留原始实测值，因此使用者可以采用自己的评分函数重新计算。回弹报告还给出由高度比推导的等效恢复系数：

```text
effective_restitution = sqrt(rebound_height / drop_height)
```

网球采用 ITF Type 2 的 1.35–1.47 m 区间；乒乓球桌采用 ITTF 从球底 0.30 m 释放、回弹 0.230–0.260 m 的区间；篮球采用 FIBA 从球底 1.80 m 释放、回弹 1.035–1.085 m 的区间。足球规则与羽毛球规则没有同类的比赛表面落球标准，因此报告将它们标记为 `engineering`，不会计入 `official_metrics_score`。

当前 v1 评分覆盖接触回弹。它不应被解释为整体仿真已达到同等精度；球拍—球、篮圈—球、飞行轨迹、滚动减速、柔性网和机器人接触需要在获得实验轨迹或测力数据后增加独立指标。

当前机器上的可复现基线为：MuJoCo 5/5 通过，总分 98.08、官方指标分 98.98；Isaac Sim/CPU PhysX 5/5 通过，总分 98.67、官方指标分 99.93。完整原始值保存在 `reports/`，后续任何物理参数修改都可以用相同命令检测回归。

## 空气动力

普通球体的阻力为：

```text
F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel
```

其中 `rho=1.225 kg/m³`，相对风速 `v_rel = v_ball - v_wind`。网球、乒乓球、足球和篮球还使用有界升力系数计算 `ω × v` 方向的 Magnus 力，所以旋转会形成可见弧线。MuJoCo 通过 passive callback 施力，Isaac Sim 通过 `RigidObject.set_external_force_and_torque()` 每步施力。

羽毛球采用相同的二次阻力基础式，但投影面积会随羽轴和来流夹角变化。阻力作用点放在质心后方 45 mm，因此压心偏置产生稳定力矩，使软木头自然朝飞行方向；另有小幅角速度阻尼。羽毛裙只参与视觉渲染，软木球头承担接触，从而避免细羽毛产生数值不稳定碰撞。

## 标准来源

- [ITF Rules and Regulations / Technical Booklet](https://www.itftennis.com/en/about-us/governance/rules-and-regulations/?type=rules)：网球场、球质量/直径和回弹测试范围。
- [ITTF Handbook](https://documents.ittf.sport/sites/default/files/public/2022-02/ITTF_HB_2022_clean_v1_0.pdf)：球台 2.74 × 1.525 m、高 0.76 m、球网 15.25 cm、40 mm / 2.7 g 球和 30 cm 落下约回弹 23 cm。
- [IFAB Laws of the Game](https://downloads.theifab.com/downloads/laws-of-the-game-202627-double-pages?l=en)：足球 68–70 cm 周长、410–450 g 和球门尺寸。
- [BWF Laws of Badminton](https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf)：场地尺寸、16 根羽毛、4.74–5.50 g、球头和羽裙尺寸。
- [FIBA Official Basketball Rules and Equipment](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)：28 × 15 m 场地、3.05 m 篮圈、7 号球质量/周长和 1.8 m 落球回弹范围。

气动系数是面向交互式刚体模拟的代表值，不是对特定品牌、缝线、湿度或雷诺数的风洞拟合。所有系数都集中在 `specs.py`，便于实验校准和域随机化。
