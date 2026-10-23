# 015：无历史融合分支与联合 dK/dV 输出

性能基线为用户已提交的 **`9b9fc8f96` 拆分版本**，不是关闭拆分的 FA。三组分别是 baseline（优化前拆分）、nohistory（仅无历史任务走融合）、both（再加入联合 dK/dV 输出）。默认训练接口与 B→交付 dx→H 顺序不变。

- 设计、源码校验和与隔离版本：`design.json`；最终复现记录见 `manifest.json`。
- 正确性：`unit-evidence.json`、`validation-summary.json`、`memcheck.log`、`pp-validation/`。融合 FA 仅作为数值参考，不是本归档性能基线。
- 性能：`run_comparison.py` 顺序运行三组；任务边界计时和无采样端到端分别归档。`summary.json` 保留配对原始数据、范围和每迭代差值，不把跨 rank 的 B/H 总和解释为 iteration time。
- 生产源码和相对基线补丁见 `source/` 与 `implementation.patch`。本轮没有修改 scheduler、timeline 或历史缓存累加算法。

## 中断及测试判据修正

首次回归243项通过、3项 profiler 用例失败：判据误把必需的 FP32 dQ 工作区清零当作多余输出归约。修正判据后仅重跑这3项，全部通过；未为此修改生产数值代码。原始失败日志 `unit-tests.log` 与修正后 `launch-tests-corrected.log` 均保留。

`tasks/` 首次隔离基线运行因默认编译多个 GPU 架构触发360秒超时，未产生可用训练样本。终止进程后清理了该次构建锁，将所有版本统一设为 `TORCH_CUDA_ARCH_LIST=8.9`，在 `tasks-retry/` 重新运行。编译发生在训练迭代之前，失败运行不纳入统计；后续运行参数完整保留在各 `*-configuration.json`。

## 最终结果

正序 `tasks-retry/` 与唯一一次反序 `tasks-reverse/` 合计每模式216个B、144个H。`e2e/` 为独立无采样六轮运行，比较配对第3–6轮。无历史B变化小，中等历史部分样本变快，长历史H出现重复回退；端到端平均1894.25→1902.55ms（+0.44%），没有测到加速。详见 `summary.json`；单组端到端不支持稳定回退幅度的断言。

反序复测由首轮不同历史形状方向不一致触发，之后未继续扩大实验。恢复中断后没有重复正确性实验。原始失败运行、测试判据修正与全部有效配对运行均保留。
