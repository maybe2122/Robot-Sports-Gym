# Contributing

感谢你改进 MultiSport Physics Sim。这个项目同时维护 MuJoCo 和 Isaac Sim/PhysX 后端；场景、物理或评测改动必须说明对两个后端的影响。

## 开始之前

- 小型修复可以直接提交 Pull Request。
- 新运动、新机器人、新 benchmark 任务或破坏兼容性的 API 变更，请先创建 issue，写明使用场景、标准来源和拟议接口。
- 不要提交来源或许可证不明确的网格、纹理、机器人模型、数据集或预训练权重。

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"

make lint
make test
make verify
make evaluate
```

Isaac Sim 需要独立的 Isaac Lab Python 环境：

```bash
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=.venv/bin/python
```

## 变更要求

### 场景和物理

1. 使用 SI 单位。
2. 在 `specs.py` 集中维护共享参数，避免两个后端各自出现不透明常量。
3. 给出标准、论文或实验数据来源，并说明它是官方指标还是工程参考。
4. 更新或新增数值测试；物理参数变更必须重新生成 fidelity 报告。
5. 若只支持一个后端，必须在文档和能力矩阵中明确标注。

### Benchmark 任务

任务需要遵守 [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md)，至少定义：

- 环境 ID 和版本；
- 机器人与控制频率；
- 动作、观测、奖励、终止和截断；
- 随机化分布与固定评测种子；
- 主指标、辅助指标和成功判据；
- state、vision、sim-to-real 轨道中适用的轨道。

### 代码和文档

- Python 3.10+，Ruff 检查，新增行为需要 pytest。
- 公共 API 应包含类型标注和简洁 docstring。
- 用户可见变化写入 `CHANGELOG.md`。
- 文档中的命令必须能从仓库根目录执行。

## Pull Request 检查表

- [ ] 变更范围单一，提交信息清晰。
- [ ] 测试和 lint 通过。
- [ ] MuJoCo/Isaac 对应关系已说明。
- [ ] 新资产和数据包含许可证、来源及校验信息。
- [ ] 物理改动附有量化前后对比。
- [ ] README、benchmark 协议或 changelog 已按需更新。

## 报告问题

Bug 报告请包含操作系统、Python/MuJoCo/Isaac Sim/Isaac Lab 版本、完整命令、随机种子、日志和最小复现。安全问题不要公开披露，参见 [`SECURITY.md`](SECURITY.md)。

