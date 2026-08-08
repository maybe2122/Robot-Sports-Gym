# 物理模型与标准来源

## 坐标和积分

- 全部输入使用 SI 单位：米、千克、秒、牛顿。
- 世界坐标 `+Z` 向上，重力为 `9.81 m/s²`。
- MuJoCo 使用 1 ms 固定步长和 `implicitfast` 积分器。
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

## 空气动力

普通球体的阻力为：

```text
F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel
```

其中 `rho=1.225 kg/m³`，相对风速 `v_rel = v_ball - v_wind`。网球、乒乓球、足球和篮球还使用有界升力系数计算 `ω × v` 方向的 Magnus 力，所以旋转会形成可见弧线。

羽毛球采用相同的二次阻力基础式，但投影面积会随羽轴和来流夹角变化。阻力作用点放在质心后方 45 mm，因此压心偏置产生稳定力矩，使软木头自然朝飞行方向；另有小幅角速度阻尼。羽毛裙只参与视觉渲染，软木球头承担接触，从而避免细羽毛产生数值不稳定碰撞。

## 标准来源

- [ITF Rules and Regulations / Technical Booklet](https://www.itftennis.com/en/about-us/governance/rules-and-regulations/?type=rules)：网球场、球质量/直径和回弹测试范围。
- [ITTF Handbook](https://documents.ittf.sport/sites/default/files/public/2022-02/ITTF_HB_2022_clean_v1_0.pdf)：球台 2.74 × 1.525 m、高 0.76 m、球网 15.25 cm、40 mm / 2.7 g 球和 30 cm 落下约回弹 23 cm。
- [IFAB Laws of the Game](https://downloads.theifab.com/downloads/laws-of-the-game-202627-double-pages?l=en)：足球 68–70 cm 周长、410–450 g 和球门尺寸。
- [BWF Laws of Badminton](https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf)：场地尺寸、16 根羽毛、4.74–5.50 g、球头和羽裙尺寸。
- [FIBA Official Basketball Rules and Equipment](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)：28 × 15 m 场地、3.05 m 篮圈、7 号球质量/周长和 1.8 m 落球回弹范围。

气动系数是面向交互式刚体模拟的代表值，不是对特定品牌、缝线、湿度或雷诺数的风洞拟合。所有系数都集中在 `specs.py`，便于实验校准和域随机化。

