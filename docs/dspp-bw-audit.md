# DSPP B/W 分解审计、实验与回退记录

日期：2026-09-03

## 1. 审计范围

- DSPP：`Megatron-LM-kwai`，当前分支 `cuiym-dspp`。
- 参考实现：`zero-bubble-pipeline-parallelism`，本地提交
  `39f2c186`。
- 真机 case：3 x L40，PP=3、VPP=2、24 layers、每个 physical stage
  每个 chunk 4 layers、hidden size 2048、chunk size 256。
- 输入仍为同一份本地 synthetic mmap 数据集，长度循环为
  `768, 512, 192, 128`。本次没有更换构造方法或 ordering。

## 2. Zero-Bubble 的 B/W 分解方式

参考实现没有实现一个新的、整体 fused 的 B kernel。它仍通过 PyTorch
autograd 执行 B，只在 tensor-parallel Linear backward 中截获 wgrad：

1. B 立即计算并传播 dgrad；LayerNorm、attention backward、激活函数等普通
   autograd 节点也仍在 B 中执行。
2. Linear 保存本次 microbatch 的 input 与 grad-output，不执行 fused wgrad
   GEMM。
3. B 完成后将该 microbatch 捕获的所有 Linear wgrad 工作 flush 到队列。
4. W 从队列取出这一批工作，准备二维 GEMM layout，然后通过
   `fused_weight_gradient_mlp_cuda` 累加到 `weight.main_grad`。
5. 新版实现还将只服务于 wgrad 的 sequence-parallel all-gather、
   `contiguous` 和 reshape 延迟到 W。

参考实现的队列按固定 `chunk x seq_split_idx` 建立。这一点不能直接复制到
DSPP：DSPP 的 chain split count 可变，且一个 physical task 可能是一个长
sequence segment，也可能是若干完整短 sequence 的 pack。

## 3. DSPP correctness 结论

DSPP 保留按 model chunk 的 FIFO 队列，每次 physical B 独立 flush 一项。队列
项直接持有该 physical task 的 activation 与 grad-output，所以 wgrad 不依赖
等长 microbatch，也不需要知道该任务内部是长 segment 还是短序列 pack。

以下工作必须留在 B，不能随 Linear wgrad 移到 W：

- 当前 segment 的 activation dgrad；
- FlashAttention backward；
- 历史 K/V 的梯度累积，以及把 successor 的 pending K/V gradient 返回给
  predecessor segment。

它们位于 segment chain 的 activation-gradient critical path。W 只处理参数的
Linear wgrad，因此不会改变
`B(sequence, segment i) -> B(sequence, segment i-1)`。

本次修正了一个队列正确性风险：旧的 `pop()` 在 W 数量大于 pending B 数量
时会用 `min()` 静默少执行 W。现在 underflow 立即报出 rank、chunk、请求数和
可用数；迭代结束仍调用 `assert_empty()` 检查遗漏项。每个 W 仅增加一次
`len(deque)` 比较，开销可忽略。

没有引入跨 physical task 的 wgrad 合并。不同有效 token 数、padding 方式或
KV chain 状态的任务保持独立；这比照搬参考实现的固定 split lane 更适合当前
变长训练。

## 4. 最终实现变化

曾把新版 Zero-Bubble 的 deferred SP/layout preprocessing 移植到普通 Linear
和 Kwai QKV Linear。真机结果显示当前 TP=1、SP=false case 没有减少 kernel，
反而把额外 Python/dispatcher 开销放入 W，使 profiled critical span 增加约
3.8%。该负优化已经完整撤销；`layers.py` 与 `warmup_fix` 时逐字一致。

`megatron/core/zbpp_utils.py`：

- W 队列由静默截断改为严格 underflow 错误。

`tests/unit_tests/dspp/test_weight_grad_store.py`：

- 覆盖不同 physical shape 在同一 chunk 中的 FIFO 顺序；
- 覆盖缺失 W 的 underflow。

## 5. 真机 profile 结果

可复查的 PyTorch profiler 实验结果：

- 修改前：`src/results/dspp_bw_audit_before/traces/`
- 已撤销实验：`src/results/dspp_bw_audit_after/traces/`
- 已撤销实验 timeline：
  `src/results/dspp_bw_audit_after/timeline/iteration_000002.svg`
- 训练日志：`src/results/dspp_bw_audit_after_train.log`

在本 case 中 TP=1、SP=false，而且进入 wgrad 的 tensor 已经 contiguous。因此
该参考实现移植没有减少 CUDA kernel：

| Stage | B1 修改前 kernel median | B1 修改后 kernel median |
| ---: | ---: | ---: |
| 0 | 274 | 274 |
| 1 | 260 | 260 |
| 2 | 260 | 260 |

B1 的 non-NCCL kernel 总时间 median 约为 1.4--1.8 ms，但启用
`with_stack=True` 的 PyTorch profiler 下，CPU task range median 约为
35--47 ms。修改前后的 kernel 数和 GPU busy time 基本相同。说明当前主要
问题不是遗漏了 Zero-Bubble 的某个 fused B 实现，而是 B 仍包含约 260 个
由 autograd、attention、normalization、activation 和 dgrad 产生的小 kernel。
B/W 分解移走大 wgrad GEMM 后，使这些 launch 更显眼。由于实验没有收益且
端到端变慢，相关源码修改没有保留。

部分 B 看起来很短也不是少算了。例如修改前 rank 1 的第一个 B1：CPU range
为 27.9 ms，但归属于它的 GPU kernels 在 range 开始约 32.6 ms 后才执行，
GPU envelope 仅 2.1 ms；这些 kernel 已经排入 stream，只是在等待先前工作。
另一些 B 在空 stream 上执行，host dispatch 间隙直接暴露，于是 timeline
显得很长。这是“同一批工作被选择性隐藏或暴露”，不是 microbatch correctness
变化。

注意：`with_stack=True` 会显著放大 Python/autograd host 开销。该 trace 适合
定位依赖与 launch，不应直接作为无 profiler 的吞吐数字。

## 6. 测试结果

- `python -m pytest -q tests/unit_tests/dspp`：`34 passed`。
- 最终 queue-only 修复的 3 x L40 训练：3/3 iterations 完成，0 skipped，
  0 NaN；第一步 loss 仍为 `1.074023E+01`。
- 与 `warmup_fix` 相同的轻量 timeline 条件下，iteration 2 为
  `1307.9 -> 1332.2 ms`，iteration 3 为 `1665.7 -> 1667.3 ms`，median
  仅变化 `+0.1%`；属于三次短运行的正常波动，没有观察到 queue 检查的性能
  影响。最终结果位于 `src/results/dspp_bw_queue_fix/`。
- 修改前后的 peak allocated memory 相同：rank 0 为 10568.1 MiB。

## 7. 可以继续做的优化

当前不建议把固定长度 Zero-Bubble 的跨 microbatch W grouping 直接移植过来。
变长/packing 下若要合并 GEMM，至少要按 weight、dtype、有效 row layout 和
shape 分桶；它还会改变 W 插入 bubble 的位置，收益需要真机验证。

更可能有效的方向是：

1. 针对 DSPP packed-attention 的 pad/unpad 和历史 K/V gradient routing 做
   C++/Triton fusion，减少 B 中确实存在的 copy、fill、add 小 kernel。
2. 对稳定的 physical-task signature 分桶后尝试 CUDA Graph；不能用一个 graph
   覆盖所有可变 history/packing layout。
3. 在启用 specialized QKV path（例如相应 CP 配置）时，将它的 Q/K/V grad
   concat 一并延迟到 W；当前 TP=1、CP=1 case 没有走该路径。
4. CUDA Green Context/SM margin 主要解决 NCCL 与 compute 的 SM 竞争。它可能
   改善通信重叠，但不会消除当前 trace 中的 host launch starvation，应作为
   独立实验，而不是 B/W correctness 修复。

本轮曾尝试对常见 KV layout 直接返回 gradient view，单元测试正确，但真机
B kernel 数完全不变：autograd 在上游 slice 边界补回了等量 materialization。
该实验代码已撤销，避免保留无收益的复杂 fast path。
