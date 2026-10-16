# DSPP 整体报告写作上下文摘要

更新日期：2026-09-13

本文不是 DSPP 的正式论文或系统手册，而是供后续新会话继续撰写“整体报告”时使用的上下文。后续写作者应以本文为导航，并在引用实现细节或实验数字时回到文末列出的权威文档核对。

## 1. 写作任务与用户偏好

目标是撰写一份能够用于论文初稿或阶段汇报的 DSPP 整体报告，让不了解实现过程的读者快速明白：

1. 变长序列 pipeline training 面临什么问题；
2. 现有或常用实现方式为什么不足；
3. DSPP 的算法与系统设计是什么；
4. 这些设计为什么合理；
5. 哪些实验支持哪些结论；
6. 当前原型完成到了什么程度，还不能证明什么。

写作时遵循以下原则：

- 使用中文，逻辑清晰，面向不参与开发的读者；
- 按“问题—挑战—方法—依据—实验—限制”组织，不按 milestone 或 bug 修复时间线组织；
- 不描述反复尝试、错误版本和无效优化，除非某个失败揭示了可推广的系统约束；
- 不把一般工程修复包装成研究贡献，也不要为了凑贡献夸大；
- 可以写有普遍意义的实现思想，例如 stage-local 推进、双向 P2P 解耦、matched lookahead、continuation-aware packed attention 和 profile-backed chunk selection；
- 所有实验结论都要说明由哪组结果支持，以及该实验不能证明什么；
- 公式使用行内代码或代码块，不使用独立 LaTeX 公式，因为当前 CLI 不渲染 LaTeX；
- 避免人为硬换行和不自然断句；
- 当前任务是整理报告，不要为了写报告重新运行实验；
- 不要把当前 L40 上测得的最优值写成跨硬件通用常数。

## 2. 项目目标与当前实验环境

DSPP 的目标是在真实 Megatron 训练运行时中支持变长序列的 stage-local pipeline scheduling，而不是把数值仿真器中的虚拟执行时间直接搬到真机。

当前原型位于：

```text
/home/cuiym/src/Megatron-LM-kwai
```

主要开发和验证环境为 3 张 NVIDIA L40。常用模型与并行配置为：

```text
PP = 3
VPP = 2
TP = CP = DP = 1
Transformer layers = 24
每个逻辑 stage = 4 layers
hidden size = 2048
FFN hidden size = 5504
attention heads = 16
BF16
FlashAttention
```

当前实现是研究原型，不应被描述成已经覆盖任意 Megatron 配置的生产系统。真实训练时间由 CUDA kernel、依赖等待和 P2P 通信共同决定。数值仿真器中的 FLOPs 或 metadata cost 仅可作为缺少真机 profile 时的排序 fallback，不能代表真实执行时间。

## 3. DSPP 的问题定义

输入 iteration 包含不同长度的 logical sequences。长度超过 chunk size 的长序列被切成 continuation segments；短序列以及长序列最后一个未填满的 tail 可以共同 packing 成 physical microbatch。

DSPP 要同时解决四类耦合问题：

- physical task 的组成和数量随当前输入长度及 chunk size 改变；
- 长序列 segment 之间存在跨 segment KV 和反向梯度依赖；
- 不同 pipeline stage 需要按本地依赖独立推进，不能被全局通信顺序重新对齐；
- compute shape、通信数量、pipeline bubble 和显存占用会随 slicing/packing 共同变化。

同一长序列必须满足：

```text
F(sequence, segment i-1) -> F(sequence, segment i)
B(sequence, segment i)   -> B(sequence, segment i-1)
```

这里的连续性是逻辑依赖和 F 入口顺序上的连续性，不要求这些任务连续占用 GPU wall-clock。实际 stage-local schedule 可以在相邻 segment 之间插入已就绪的 B、W 或其他任务。

## 4. 最终实现采用的核心方法

### 4.1 固定 Physical Transport 与变长逻辑语义分离

固定 chunk size 已经使 activation 和 gradient 的 physical tensor shape 统一为 `[chunk_size, hidden]`。真正的困难不是不同 microbatch 的通信 shape 不一致，而是相同 shape 的 tensor 可能具有完全不同的逻辑语义，例如长序列中段、长序列 tail 或多个短序列组成的 pack。

最终实现将 transport 与 logical metadata 分离：

- transport 层只传输 padding 后的固定 shape payload，不进行逐消息动态 shape handshake；
- pipeline 各 stage 根据广播的紧凑 sequence lengths 确定性重建相同的 `DsppBatchPlan`；
- plan 保存 sequence id、segment id、offset、valid tokens、sequence boundary、cu-seqlens、position 和 loss mask 等语义；
- 非整除长度自然产生 tail，tail 可以继续与短序列 packing。

例如 `length=16384, chunk=3584` 会产生四个 3584-token segment 和一个 2048-token tail，而不是要求长度必须整除 chunk。

这一设计的意义是：通信协议保持固定和简单，同时 attention、KV state、loss mask 和梯度路由仍能恢复准确的变长语义。

### 4.2 Stage-local V-ZB 调度

实现根据同一 batch plan 为每个 physical stage 生成自己的 F/B/W 任务序列。每个 stage 只根据本地任务依赖、输入数据是否到达以及 deferred-W 状态推进：

- 没有全局 action index；
- iteration 内不使用 barrier 强制 stage 对齐；
- activation、gradient 和 deferred-W 状态使用包含 logical batch、physical microbatch 和 model chunk 的 TaskId 标识；
- stage 完成当前任务后即可推进到下一个本地可执行任务。

这把仿真器中的 stage-based fixed schedule 落实为真实的异步 pipeline，而不是由中央控制器逐步向全部 stage 发同一个动作。

### 4.3 双向 P2P 解耦与大消息有界异步协议

同一 PP edge 上 activation 和 gradient 的方向相反。如果两个方向共享一个 communicator 或通信 stream，一个方向排队的工作可能导致另一个方向 head-of-line blocking。因此，最终实现为每条物理 PP edge 的两个方向分别创建 NCCL communicator 和 CUDA stream。

仅拆分 stream 仍不足以稳定处理大量大消息。最终大消息协议为：

```text
sender ready
  -> receiver 在 lookahead 窗口内提前提交匹配 irecv
  -> receiver 通过控制面返回 credit
  -> sender 在数据面提交 isend
```

控制面和数据面职责不同：

- CPU 辅助线程阻塞接收 ready、credit 和 ack，只更新 CPU 状态，不调用 NCCL 或 CUDA；
- 主训练线程在任务边界或等待依赖时处理已经到达的 credit，并在对应 NCCL stream 上提交 isend；
- receiver 可以在真正消费 tensor 前提前 irecv，但 sender 不会无限提交 unmatched sends；
- 每条 lane 的 receive lookahead 和 send in-flight depth 都有上限。

当前默认参数是：

```text
receive lookahead depth = 2
send in-flight depth = 2
```

它应概括为“控制面有序匹配、数据面有界异步传输”。credit 约束发送 launch 的匹配条件，但不会让所有 stage 在同一点同步，也不会阻塞主线程执行无关计算。

### 4.4 Continuation-aware Packed Attention

一个 physical pack 可以同时包含长序列当前 segment 和多个独立短序列。长序列必须看到自身历史 KV，而 pack 内不同序列又必须互相隔离。

实现为 Q 和 K 分别构造 cu-seqlens。例如：

```text
Q = [A_current, B, C]
K = [A_history + A_current, B, C]
```

配合 bottom-right causal alignment，A 能看到自己的历史，B 和 C 仍是互相独立的序列。KV state 按 `(layer, sequence_id)` 管理，而不是绑定到固定 microbatch 编号。反向时，后续 segment 对历史 K/V 的梯度会被路由回前驱 segment。

这是 DSPP 支持“长序列切分 + tail/短序列混合 packing”的关键 correctness 机制。

### 4.5 面向变长 Physical Task 的 B/W 分解

DSPP 把 backward 拆成：

- B：activation dgrad、FlashAttention backward 和 KV gradient routing；
- W：可延迟执行的 Linear weight-gradient。

等长 pipeline 可以依赖固定 microbatch/slice 下标组织 deferred W，但 DSPP 的 segment 数量和 pack 组成随输入变化。当前实现让每个 physical task 在 B 完成后，把对应的 activation 和 grad-output 写入所属 model chunk 的 FIFO；W 按 stage-local 调度顺序弹出。因此 deferred-W 不依赖固定 split 数，也不关心 task 内部是长序列 segment、tail 还是短序列 pack。

iteration 结束时检查所有 W 和 KV 状态已经排空，以发现结构性遗漏。这类检查位于构造或 iteration 边界，而不是每个 GPU task 的热路径。

### 4.6 Microbatch 构造与排序

当前构造方式为：

1. 长序列按 chunk size 切分；
2. 完整 middle segment 通常单独成为 physical microbatch；
3. 长序列 tail 建立 residual bin；
4. 短序列从长到短放入可行 bin；
5. physical tensor padding 到固定 chunk shape，并用 loss mask 排除 padding token。

当前 ordering 接入了数值仿真器的基本规则：warmup 优先选择预计执行时间接近的 physical microbatch，steady 阶段倾向长任务优先，同时保持同一长序列的 F segment 正序和 B segment 逆序。

现有 packing 和 ordering 已可运行，但它们仍是下一阶段最值得继续研究的算法方向，不能在整体报告中描述为已经全局最优。

### 4.7 热路径优化原则

可写入系统实现部分的有效原则包括：

- 正确性主要编码在 batch plan、TaskId、KV/deferred-W 生命周期和 iteration 边界断言中，不在每个 GPU task 前扫描完整 DAG；
- 使用 CPU metadata 中已有的长度，避免每层执行 GPU tensor `.item()` 造成 host-device 同步；
- no-history pack 直接复用 current K/V 和 dK/dV；
- 单 part 走 view，multi-part 使用预分配 slice，减少额外 materialization；
- 优先减少无意义的小 kernel 和 host launch，而不根据 SVG 中单个方块的视觉长度直接修改调度语义。

已有 profiler 观察中，每个 4-layer B1 的 non-NCCL kernel median 约减少 12 个；同一 profiled iteration 中 `SelectBackward` 从 156 次降为 0。这说明优化确实减少了 launch 工作，但不能据此宣称已经消除全部 eager execution 或 host launch overhead。

## 5. Per-iteration Chunk Size Selection

### 5.1 为什么这是独立的研究问题

chunk size 同时改变：

- F/B/W 的 GPU kernel shape 和执行效率；
- 长序列 continuation chain 深度；
- tail 与短序列 packing 结果；
- physical microbatch、CUDA launch 和 P2P 消息数量；
- pipeline fill/drain、bubble 和 critical path；
- activation、历史 KV、workspace 和通信 buffer 的显存峰值。

因此不能仅凭 token FLOPs、固定经验值或“选显存允许的最大 chunk”得到通用答案。GPU shape efficiency 可能受 tile quantization、wave quantization、FlashAttention work partitioning 和库 kernel heuristic 影响而出现平台、跳变或局部回落。同一 chunk 在 full、middle、tail 和 pack 四种 workload family 上也可能表现不同。

### 5.2 最终选择流程

当前方法分为离线硬件校准和在线 per-iteration 选择。

离线阶段：

```text
profile full / middle / tail / pack 四类 workload 的 F/B/W
  -> 保存完整 measured profile bank
  -> 校准 stage/op compute timeline
  -> profile P2P，或使用 latency + bytes / bandwidth 模型
```

在线阶段：

```text
读取当前 iteration 的真实 sequence lengths
  -> 使用显存模型过滤不可行 chunk
  -> 按真实 slicing/packing 构造结果计算 cheap score
  -> 保留 top-k + slack
  -> 额外保留局部效率优点、最大实测可行点和当前 chunk
  -> 为候选构造 weighted DSPP DAG
  -> 以预测 makespan 排序
  -> pipeline first rank 选择并广播
```

profile bank 应保留 specialist 和结构边界点，不能只保留静态 Pareto 点。某个 chunk 即使在四条独立 efficiency 曲线上都不是第一，也可能因为改变 segment 数、packing 和完整 pipeline DAG 而成为某个 iteration 的最优值。

### 5.3 当前实验结论

当前 dense profile bank 包含 10 个 capacity、150 条 F/B/W shape records：

```text
256, 512, 1024, 2048, 3072,
3328, 3584, 3840, 4096, 4352
```

选择质量使用以下七个端到端实测 chunk 审计：

```text
2048, 3072, 3328, 3584, 3840, 4096, 4352
```

三组采样输入、共 12 个 iteration 的结果为：

```text
候选 menu 覆盖七点 oracle：12/12
DAG 命中逐轮最优：7/12
全部 iteration 累计 ranking regret：1.61%
排除 cold start 后累计 ranking regret：0.87%
```

三组 seed 聚合后，固定 3840 都是当前 L40 配置上的最快固定值；但逐 iteration 最优值为：

```text
3840：8 次
3072：1 次
3328：1 次
4096：1 次
4352：1 次
```

这支持以下有限结论：

- 当前候选构造没有丢失七点搜索空间中的真机 oracle；
- fixed 3840 是当前环境的强 baseline，但不是每轮最优；
- 最大实测 chunk 4352 不是普遍最快；
- 非静态 Pareto 点 3328 和 4352 仍可能因完整 task/DAG 结构成为逐轮最优；
- DAG 即使只命中 7/12，稳态累计 regret 仍低于 1%，说明多数误选发生在性能接近的候选之间。

这不能证明：

- 3840 对其他模型、GPU 或输入分布仍然最优；
- 七个离散点等价于连续 chunk 空间的全局 oracle；
- 当前 memory model 对 256/512 等极小 chunk 已经完全可靠；
- 动态 chunk 已经获得端到端收益。

当前同步 planner 每轮约 79–464 ms，平均约 217 ms。上述 fixed-run 查表比较没有把规划时间计入 iteration time。因此，选择质量已经足够用于继续研究，但正式动态训练需要把下一轮 metadata planning 与当前轮 GPU 计算重叠，或者进一步降低 planner 开销。

## 6. 关键实验与可支持的结论

### 6.1 DSPP 与 InfiniPipe 的初步端到端比较

双方使用固定 `chunk_size=3584` 和相同的 12 个 iteration 输入。排除包含初始化与 warmup 的第一轮后：

| 指标 | DSPP | InfiniPipe | DSPP 相对变化 |
| --- | ---: | ---: | ---: |
| 平均 iteration | 2343.5 ms | 2780.0 ms | -15.7% |
| Median | 2358.0 ms | 2871.7 ms | -17.9% |
| P95 | 3220.7 ms | 3566.9 ms | -9.7% |
| 有效 token/s | 18,781.8 | 15,832.9 | +18.6% |
| 峰值显存 | 32.69 GiB | 36.79 GiB | -11.2% |

这组实验支持：在当前 3×L40、模型配置、输入和 fixed chunk 下，DSPP 原型获得了更低 iteration time 和更高有效吞吐。

它不能证明 DSPP 在任意模型、80GB GPU、更多 pipeline stages 或任意长度分布下都优于 InfiniPipe。两边峰值显存的统计口径并不完全一致，因此显存数字只能作为初步观察，不宜作为最强结论。

### 6.2 Packing × Ordering 消融

最新实验使用三个 Figure-2 采样 seed、固定 `chunk=3584`，每个配置聚合 9 个稳态 iteration 和相同的 340,080 个有效 token：

| 配置 | 平均 iteration | 有效 token/s | Physical MB/iter |
| --- | ---: | ---: | ---: |
| no packing + input order | 2971.8 ms | 12,715.2 | 20.22 |
| no packing + DSPP ordering | 3024.3 ms | 12,494.4 | 20.22 |
| packing + input order | **2014.6 ms** | **18,756.3** | **12.44** |
| packing + DSPP ordering | 2060.1 ms | 18,341.8 | 12.44 |

Packing 将 physical microbatch 数减少 38.5%，平均 iteration 时间降低约 32%，并在 9/9 个逐轮配对中获益。当前 `warmup-short-steady-long` ordering 则没有显示出稳定收益：有无 packing 时都约有 2% 的轻微回落，配对 bootstrap 区间跨过 0。因此只能写“当前 ordering 尚未形成可辨认收益”，不能写成 ordering 已经有效，也不能断言它普遍有害。

### 6.3 双向 Communicator 消融

在保持每条物理 PP edge 独立的条件下，比较两个方向分别使用 communicator 与共享一个 communicator。方向拆分的 3/3 个 seed 全部完成；共享方向的 0/3 个 seed 能在 60 秒内完成第一个训练 iteration。

这个实验不能给出普通的 speedup 数字。它支持的是：对于当前 stage-local 异序 F/B 通信，双向 communicator/stream 解耦是可靠 progress 的必要条件，而不只是微小性能优化。它不能用于声称经典全局对齐 pipeline 也一定需要同样拆分。

### 6.4 大消息通信优化

matched receive lookahead depth 1 相对严格 receiver-at-consume：

```text
平均 iteration：-33.5%
P95：-39.3%
有效 token/s：+50.4%
12/12 iteration 正常完成
```

depth 2 相对 depth 1：

```text
平均 iteration：-2.74%
有效 token/s：+2.82%
P95：-6.12%
rank-0 peak allocated 增加约 4 MiB
rank-0 peak reserved 增加约 30 MiB
两组均完成 12/12 iteration
```

depth 4 的短测已经出现性能回落，因此当前默认 depth 2。这组实验说明有限 lookahead 能恢复大消息下的 overlap，且窗口并非越深越好。它不能证明 depth 2 是其他网络、GPU 或 PP degree 的通用最优值。

### 6.5 Chunk Selection 实验之间的逻辑关系

- packing 消融说明 task 数和组成必须进入 chunk 模型，仅比较单 kernel FLOPs 不够；
- 四类 compute profile 说明不同 physical workload family 的效率曲线不相同；
- 扩展到 8192 的独立 profile 在当前环境观察到约 3072–5120 的高效平台，并在更大值上回落，反驳“更大 chunk 必然更高效”；
- 七个 fixed chunk 的 12 轮真机结果给出离散 oracle；
- menu coverage 12/12 验证候选构造没有在该搜索空间中过早丢弃最优点；
- 稳态 ranking regret 0.87% 验证 weighted DAG 已足以用于候选排序，但不表示 makespan 的绝对预测已经精确。

最新真实在线消融进一步比较 fixed 2048、fixed 4096、最大显存可行和 profiled auto。fixed 4096 的平均 iteration 为 2072.7 ms；最大显存可行虽然把平均 physical microbatch 降至 7.11，却慢 26.8%，直接说明最大可行 chunk 不是性能最优。profiled auto 将同步 planner 计入后慢 16.3%；其 planner 平均耗时 324.2 ms，诊断性扣除后平均为 2085.7 ms，只比 fixed 4096 慢约 0.6%。正式结论是选择质量接近强 fixed baseline，但 planner 尚未异步隐藏，当前端到端动态训练仍未超过 fixed 4096。

随后又增加了五个随机 seed：`567660、56701、297129、816353、394449`。与原 seed 11、19、29 合并后，四种 chunk 策略各有 8 个 seed、24 个稳态 iteration，共 842,368 个有效 token。合并结果为：fixed 2048 平均 2677.3 ms，fixed 4096 平均 1906.8 ms，max-feasible 平均 2383.1 ms，profiled auto 平均 2224.4 ms。这里的 profiled-auto 原始时间包含同步 CPU planner。相对 fixed 4096，三者分别慢 42.1%、22.0% 和 16.9%；profiled auto 的 24/24 个稳态 iteration 都更慢。仅用于诊断地扣除其平均 289.8 ms 同步 planner 后，auto 为 1934.6 ms，仍比 fixed 4096 慢约 1.5%。这组扩展复核进一步支持：当前环境中应把 profile 用于选择全局 fixed chunk，而不能宣称 per-iteration 动态选择已经带来端到端加速。

## 7. 观测与验证工具

当前实现提供：

- 默认指标：有效 token/s、iteration median/P95、packing utilization 和显存；
- 轻量 timeline：每个 stage 一条合并 compute lane，使用 CUDA Event 记录 F/B/W 的 GPU stream 墙钟区间，并标注 physical microbatch id；
- PyTorch Profiler：包在最外层训练循环，可导出 Chrome/Perfetto trace，用于同时观察计算、P2P、CPU launch 和 stream overlap；
- microbatch composition 与排序结果单独写入可读文件，不挤入 SVG。

timeline 中 task duration 表示 compute stream 上从任务起点 event 到终点 event 的墙钟包络，不等于该区间内所有 compute kernels 的纯时间之和。区间中可能包含 host launch gap、stream dependency 或资源竞争。因此不能仅根据某个 B 方块短或长就断言它少算或多算了数学工作，必须结合 profiler kernel 明细和依赖关系判断。

## 8. 当前完成度

已经完成：

- 变长 slicing、非整除 tail 和 residual packing；
- continuation-aware packed attention；
- 跨 segment KV state 与梯度路由；
- PP=3、VPP=2 的 stage-local V-ZB；
- 双向 P2P communicator/stream 解耦；
- 大消息 matched lookahead 和有界 multi-inflight；
- 面向变长 physical task 的 B/W 分解；
- warmup/steady ordering；
- metrics、timeline 和最外层 PyTorch profiler；
- profile-backed per-iteration chunk selection；
- fixed chunk 下与 InfiniPipe 的初步比较。

仍需后续研究和验证：

- 设计更好的 microbatch packing，而不只使用当前 heuristic；
- 优化 warmup/steady ordering 和整体 stage-local schedule；
- 完成普通 pipeline、变长 batch、packing、ordering 等系统消融；
- 使用均匀、双峰、长尾等多类长度分布；
- 将 chunk planner 与前一 iteration 的 GPU 执行异步重叠；
- 利用 bubble 调度 critical-path attention；
- 在 80GB 或更大显存的生产 GPU、更长上下文和更多 PP 配置上验证；
- 支持或验证 TP、CP、DP 组合；
- 对任意合法 input ordering 的通信稳定性做更系统验证。

下一阶段若只能选择一个重点，优先级应放在 microbatch 构造和调度算法，而不是继续针对当前 L40 调节 chunk selector 权重。当前稳态 chunk ranking regret 已低于 1%，而 packing 消融显示 task set 的变化可以带来数量级更大的影响。

## 9. 整体报告的推荐结构

建议正式报告按以下结构写：

1. 背景与问题定义：变长序列为什么破坏等长 pipeline 的静态假设；
2. 现有方法的不足：全局同步节奏、固定 split 数、固定 chunk 或仅 FLOPs 建模为何不够；
3. DSPP 方法总览：logical batch 到 physical tasks，再到 stage-local F/B/W；
4. 关键系统设计：metadata/transport 分离、continuation attention、B/W、双向 P2P 和 matched lookahead；
5. chunk size selection：为什么困难、离线 profile 与在线 weighted-DAG 方法；
6. 实验设置：硬件、模型、输入、baseline、统计口径与 cold-start 处理；
7. 主要结果：DSPP vs InfiniPipe、packing、通信窗口、chunk selector；
8. 结果讨论：每组实验能证明什么、不能证明什么；
9. 当前限制与后续工作。

不要将报告写成：Milestone A 做了什么、B 修了什么、C 又出现了什么问题。这些历史有助于开发审计，但不是论文叙事。

## 10. 权威文档入口

写作时优先参考以下文件：

```text
docs/dspp-method-implementation-progress.md
```

面向论文与阶段汇报的总体方法说明，适合作为整体报告的主要骨架。

```text
docs/dspp-implementation-report.md
```

系统实现手册，包含代码入口、参数、支持边界、通信细节、timeline/profiler 解释和复现信息。需要精确描述实现时查阅。

```text
docs/dspp-chunk-size-selection.md
```

chunk selection 的论文式方法与实验报告，包含问题论证、profile bank、候选构造、weighted DAG、fixed runs、oracle coverage 和 regret。

```text
docs/dspp-ablation-study-20260913.md
```

最新三组核心消融，包含 Packing × Ordering、双向 P2P communicator 和真实在线 chunk selector 对比。整体报告引用最新消融数字时以此文件为准。

```text
docs/dspp-followup-review-report.md
```

KV、通信和 planning 等代码审计细节。它只用于核对实现，不应成为论文叙事主线。

## 11. 新会话可直接使用的任务描述

```text
请基于以下文档撰写 DSPP 整体报告：

1. docs/dspp-overall-report-writing-context.md
2. docs/dspp-method-implementation-progress.md
3. docs/dspp-implementation-report.md
4. docs/dspp-chunk-size-selection.md

报告按论文逻辑组织：问题与现有方法不足、DSPP 方法、关键实现设计、
理论依据、实验设置、实验结果、限制和后续工作。不要按 bug 修复或
milestone 时间线叙述，也不要把工程调试过程写成贡献。

所有实验结论都要指出由哪组结果支持，并说明该结果不能证明什么。
不要把 L40 上测得的 3584、3840、lookahead depth 2 等数值写成跨环境
通用常数。CLI 不渲染 LaTeX，请使用行内代码、代码块和 Markdown 表格。
在开始写正文前，先给出报告提纲和每节的核心论点，等待确认后再展开。
```
