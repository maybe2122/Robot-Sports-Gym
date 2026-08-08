# Reproducibility

## 当前已验证环境

| 组件 | 已验证版本 |
|---|---|
| Python | 3.10（MuJoCo）、3.11（Isaac 环境） |
| MuJoCo | 3.x（项目约束 `>=3.2,<4`） |
| Isaac Sim | 5.0.0 |
| Isaac Lab | 0.46.2 |
| Isaac physics | CPU PhysX, 240 Hz, TGS, CCD |

“支持范围”和“已验证组合”是两件事。论文或结果包必须记录实际解析后的完整依赖版本，不能只引用上表。

## 最小复现记录

每次 benchmark 运行至少保存：

- Git commit 和工作树是否干净；
- task、physics、asset、randomization 和报告 schema 版本；
- Python、MuJoCo、Isaac Sim、Isaac Lab、CUDA、驱动和操作系统版本；
- CPU/GPU 型号、并行环境数和线程设置；
- 全部随机种子和确定性选项；
- 完整命令与解析后的配置；
- 每个 episode 的原始结果，不只保存平均值；
- 训练曲线、最终权重哈希、评测日志和失败样本。

## 随机性

- 环境通过 `reset(seed=...)` 接收种子。
- Python、NumPy、PyTorch、环境随机化和策略采样使用可追踪的独立 RNG stream。
- 训练、验证和测试 seed 空间互不重叠。
- 并行环境的子种子由主种子确定性派生，不能依赖进程启动时间。
- GPU/PhysX 的并行规约可能不是逐位确定；因此报告统计分布，而不是只比较单条轨迹哈希。

## 物理回归

```bash
make test
make verify
make evaluate
make test-isaac ISAAC_PYTHON=/path/to/isaac/python
make evaluate-isaac ISAAC_PYTHON=/path/to/isaac/python PYTHON=/path/to/python
```

修改质量、尺寸、材质、时间步、求解器、气动或碰撞几何后，必须重新生成并审查 `reports/`。已有 fidelity 分数只覆盖报告中列出的指标。

## 发布归档

公开版本应：

1. 创建带签名或受保护的 Git tag；
2. 发布源代码归档、wheel、结果 schema 和基线报告；
3. 将数据与权重上传到可长期访问、带校验和的存储；
4. 使用 Zenodo 等归档服务获得 DOI，并回填 `CITATION.cff`；
5. 在 changelog 中记录所有影响成绩可比性的变化。

