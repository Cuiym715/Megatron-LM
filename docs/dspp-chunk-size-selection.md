# DSPP 面向变长序列的 Per-iteration Chunk Size 选择

更新日期：2026-09-13

本文说明 DSPP 为什么需要动态选择 chunk size、这一问题为什么不能用固定经验值或简单公式解决，以及当前实现采用的离线硬件校准、在线候选筛选和 weighted-DAG 排序方法。文档同时整理现有实验，区分每组实验能够支持的结论与尚不能支持的结论。

## 1. 问题背景

DSPP 用固定容量的 physical microbatch 承载变长训练样本。长度超过 chunk size 的序列被切成 continuation segments；短序列和长序列最后一个未填满的 tail 可以被 packing 到同一个 physical microbatch。

设序列长度为 `L`，chunk size 为 `c`。该序列产生的 segment 数为：

```text
segments(L, c) = ceil(L / c)
```

因此，chunk size 不只是 tensor shape 参数。改变 `c` 会同时改变：

- F、B、W 计算任务的 GPU kernel shape 和执行效率；
- 长序列 continuation chain 的深度；
- tail 的长度以及短序列 packing 的组合；
- physical microbatch、CUDA launch 和 P2P 消息的数量；
- pipeline warmup、steady、cooldown 和 bubble；
- activation、历史 KV、临时 workspace 和通信 buffer 的显存占用；
- 最终 DSPP 依赖图及其 critical path。

目标因而不是寻找一个对所有输入都固定最优的 chunk，而是在显存安全的前提下，为当前 iteration 选择预计 pipeline makespan 最小的 chunk：

```text
choose c from memory-feasible capacities
to minimize predicted DSPP critical path under c
```

## 2. 为什么 Chunk Size 选择是一个困难问题

### 2.1 相同有效计算量不等于相同真机时间

如果只计算有效 token 的理论 FLOPs，把一条长序列切成多段不会明显改变 dense layer 的总有效工作，causal attention 的有效计算也只是被拆分成多个前缀增量。这容易产生“总计算量相同，所以 chunk size 不重要”的直觉。

真机执行的却是 physical task，而不是抽象 FLOPs。不同切分会产生不同数量和形状的 GEMM、FlashAttention、padding、KV 拼接、autograd graph、CUDA launch 和 P2P 操作。即使有效 token 相同，以下两种执行也不等价：

```text
一个较大的高利用率 physical task

多个较小的 physical tasks
  + 更多 launch
  + 更多通信
  + 更深依赖链
  + 更多未填满的 tail
```

因此，token 数或 FLOPs 可以描述工作规模，但不能直接预测不同 chunk 的相对运行时间。

### 2.2 GPU 效率对 Shape 不保证单调

较大的矩阵通常更容易饱和 GPU，但“更大”并不严格推出“更高效”。矩阵 tile quantization、SM wave quantization、维度对齐和库内部 kernel heuristic 都可能产生局部平台、跳变或回落。FlashAttention 的并行度也同时依赖 query length、KV length、batch/item 数量和 work partitioning。

这意味着，即使两个 chunk 都能放入显存，也不能仅按数值大小判断它们的计算效率。目标环境必须实际测量不同 capacity 下的 compute shape。

NVIDIA 的矩阵乘法性能指南给出了 tile 和 wave quantization 导致较大矩阵反而更慢的例子；FlashAttention-2 也说明 attention kernel 的利用率与序列维度上的 work partitioning 密切相关。这些机制支持“必须测量 shape efficiency”，但并不意味着所有硬件都会在相同 chunk 上出现同一个拐点。

### 2.3 最大显存可行 Chunk 不一定最快

直接选择显存允许的最大 chunk 隐含了 iteration time 随 `c` 单调下降的假设。但增大 chunk 同时带来相反作用：

| 较小 chunk | 较大 chunk |
| --- | --- |
| 单任务 activation 和通信 payload 较小 | 通常有更少 segment、launch 和消息 |
| 可能保留更多 pipeline tasks | GEMM/attention 更容易达到高吞吐区 |
| kernel 可能未饱和 | tail padding 和单任务显存更大 |
| continuation chain 更深 | physical microbatch 太少时可能增加 bubble |
| 当前 KV 实现可能重复保存更多历史前缀 | 某些 shape 可能遇到 tile/wave 低效点 |

Megatron 对经典 pipeline 的分析给出 bubble fraction 与 microbatch 数近似成反比。虽然 DSPP 使用的 V schedule 不是经典 flush schedule，但原理仍然成立：过大的 chunk 如果让 physical microbatch 数过少，就可能削弱 pipeline 并行度。反过来，输入中有足够多任务时，大 chunk 又可能通过减少 launch 和通信获益。

### 2.4 相同 Chunk 下也存在不同计算形态

DSPP 的一个 physical microbatch 不能只由 capacity 描述。至少需要区分四类 workload：

| Family | 形态 | 单独测量的原因 |
| --- | --- | --- |
| `full` | 一个无历史 item 填满 capacity | 表示首段和固定 dense shape 的基本效率 |
| `middle` | query 接近 capacity，同时读取历史 KV | attention 的 KV 长度随 segment 位置增长 |
| `tail` | query 未填满 capacity，可带历史 | 暴露低利用率、padding 和末段开销 |
| `pack` | 多个短序列，或 tail 与短序列混合 | item count、cu-seqlens 和组成会改变 kernel 行为 |

某个 capacity 可能在 `full` 上高效，在 `pack` 上却一般；另一个 capacity 可能没有任何单项第一，但更适合某种混合 iteration。因此，离线阶段不能把每个 chunk 压缩成一个脱离输入的固定分数，更不能只测普通等长序列。

### 2.5 显存也不保证随 Chunk 单调变化

较大 chunk 会增加单任务 activation、workspace 和 P2P buffer；但在当前 continuation attention 实现中，较小 chunk 会产生更多 segment，并让多个 autograd context 同时保存不同长度的 K/V 前缀。

对长度 `L`、近似整除的长序列，若 `n = L / c`，各 segment 保存的前缀长度近似为：

```text
c, 2c, 3c, ..., n*c
```

总前缀长度近似为：

```text
L^2 / (2c) + L / 2
```

因此，当前实现中的一部分历史状态会随 chunk 变小而增加。总显存是“大 chunk 的单任务开销”和“小 chunk 的重复历史状态”共同作用的结果，不能只用单调二分假设处理。

### 2.6 在线求解不能成为新的瓶颈

训练前不知道每个 iteration 的长度组合，而在每轮完整遍历所有 aligned chunk、为每个 chunk 构造完整 pipeline DAG，又会产生过高 CPU 开销。可用的方法必须同时满足：

- 离线测量硬件和模型相关的 shape efficiency；
- 在线使用当前 iteration 的真实长度与 packing；
- 保留极端长序列需要的小 chunk fallback；
- 只对一个较小候选集合构造完整 DAG；
- 不把某台机器上人工试出的 3584、4096 写入算法规则。

## 3. 常见方法为什么不够

### 3.1 固定经验 Chunk

固定 chunk 简单、稳定，适合作为 baseline，但它假设输入分布、模型、GPU 和 kernel 长期不变。变长训练中，不同 iteration 的最长序列、tail 余数和短序列 packing 组合都会变化，因此固定值无法对每轮结构变化作出响应。

### 3.2 选择最大可容纳 Chunk

这种方法只解决 OOM，不解决性能。它忽略了 kernel 局部效率、physical task 数和 pipeline bubble 的非单调变化。最新七点真机实验也观察到：虽然 4352 可运行，但多数 iteration 的最优值是 3840；另一方面，4352 在一轮输入上又确实优于所有更小候选。

### 3.3 仅按 FLOPs、Token 数或 Padding 排序

这些指标没有直接表示 kernel 饱和、history-dependent attention、per-task launch、P2P 数量和 stage critical path。它们可以作为 profile 缺失时的 fallback，却不足以单独决定最终 chunk。

### 3.4 训练前穷举所有输入

真实训练通常有大量 iteration，训练前也无法获得全部未来输入。即使数据集可扫描，把所有 iteration 都预先 packing、建图和仿真也会增加启动成本，并把方法绑定到固定数据顺序。因此本文采用“输入无关的离线 profile + 当前 iteration 的在线规划”，而不是预先求完整训练的最优序列。

### 3.5 与相关长序列 Pipeline 方法的区别

已有工作已经证明了长序列切分和变长 pipeline 调度的重要性。TeraPipe 研究单条长序列内部的 token-level pipeline 和切分优化；DynaPipe 研究变长 microbatch 构造与动态 pipeline；InfiniPipe 将长序列切分、packing 和资源约束结合。这些工作为 DSPP 提供了直接基础，但本文处理的是一个更具体的耦合变量：同一个 chunk size 同时决定 continuation attention shape、短序列 packing、V-schedule task 数、双向 P2P 消息和 activation/KV 生命周期。

因此本文不把“序列切分”或“变长 packing”本身作为新贡献。需要解决的缺口是：仅按内存容量决定 token capacity，或者在固定 cost model 下优化切分，都不能直接给出目标 GPU 上不同 physical family 的真实效率，也不能表示 chunk 改变后完整 DSPP DAG 的离散变化。本文方法将硬件 profile 与当前 iteration 的真实 DSPP 依赖图结合，专门解决这一选择问题。

## 4. 最终方法概览

最终方法把问题拆成硬件相关但输入无关的离线阶段，以及输入相关但只处理 metadata 的在线阶段：

```text
目标模型和硬件上的离线阶段
  1. 测量 full/middle/tail/pack 的 F/B/W compute shapes
  2. 建立完整 profiled capacity bank 和分 family efficiency 曲线
  3. 用少量真实 DSPP timeline 校准不同 stage/op 的系统开销
  4. 建立 P2P profile，或使用 latency + bytes / bandwidth 解析模型

当前 iteration 的在线阶段
  1. 根据当前长度、显存预算和通信窗口过滤不可行 chunk
  2. 对可行 bank 点构造真实 slicing/packing metadata
  3. 用 profile-backed cheap score 筛出小候选 menu
  4. 对 menu 构造完整 weighted DSPP DAG
  5. 选择 predicted makespan 最小的 chunk
  6. PP first rank 将选择广播给其他 pipeline ranks
```

该分解的核心是：GPU shape efficiency 由离线实测决定，而 segment、packing、bubble 和依赖结构由当前输入决定。二者只在候选筛选和 weighted DAG 中结合。

## 5. 离线 Compute Profile Bank

### 5.1 Profile 对象

每个离线记录描述一个 physical compute task：

```text
stage
operation = F / B / W
capacity
query_lengths[]
history_lengths[]
item_count
median_ms / p95_ms
```

`capacity` 决定固定 dense shape；`query_lengths` 决定有效 attention query；`history_lengths` 决定每个 item 读取的历史 KV；`item_count` 区分单序列和 packed attention。

### 5.2 为什么 F、B、W 分开测量

DSPP 将 backward 拆成 B 和 W：

- F 执行前向 Transformer subgraph；
- B 计算 activation gradient、attention backward 和 KV gradient routing；
- W 延迟执行 weight-gradient GEMM。

三类任务的 kernel 组成和随 shape 变化的规律不同，不能从 F 时间按固定比例推出 B/W。离线 adapter 因此分别测量 F、B、W，并确保 B 不重复计算 W。

### 5.3 Family 内归一化

不同 family 的原始毫秒不能直接平均，因为它们承担的有效工作不同。profile adapter 为每个 shape 提供与模型结构一致的 useful work，在同一 family 内计算：

```text
normalized_efficiency(shape)
  = measured useful work per millisecond
    / best measured useful work per millisecond in the same family
```

每个 capacity 保存 `full`、`middle`、`tail` 和 `pack` 四个分数。Pareto 和 robust plateau 只作为解释性元数据；运行时保留完整 measured bank，因为一个在单 family 上不占优的点仍可能因 segment 数和 packing 结构在真实 iteration 中获胜。

### 5.4 可迁移性

profile 数值与 GPU、模型宽度、层数、dtype、attention kernel 和软件版本绑定。迁移到其他环境时重新运行 calibration，但不修改候选生成和 DAG 算法。可迁移的是测量接口与决策流程，而不是 L40 上的某个具体 chunk。

## 6. Per-iteration 候选构造

### 6.1 显存可行性过滤

训练运行时为每个 candidate 构造当前 iteration 的实际 batch layout 和 Slice-V schedule，并估计 peak live activation slots。当前显存估计为：

```text
estimated dynamic memory
  = peak_live_slots
    * chunk_size
    * measured bytes per live token
    * memory safety factor
  + bounded communication buffer reserve
```

`bytes per live token` 由 bootstrap iteration 的实际 allocator 峰值形成 high-water estimate，只提高、不因后续较轻 iteration 降低。当前默认 safety factor 为 1.05。

每个 PP rank 本地计算显存可行 mask；PP group 使用 MIN all-reduce，只有所有 stage 都可容纳的 chunk 才进入性能筛选。这保证最终 chunk 在所有 ranks 上一致可行。

该模型对当前测试区间有效，但尚未完整表示小 chunk 下累计历史 K/V 前缀和临时拼接峰值。因此 256/512 等极小 chunk 的通用 OOM 安全性仍需更精确的 history-aware memory model。最新端到端选择评估集中在 2048–4352，并且相关 fixed runs 均正常完成。

### 6.2 Cheap Score

对每个显存可行的 profiled capacity，planner 使用当前 iteration 的真实长度运行 metadata-only slicing 和 packing，从而获得精确的 physical microbatch 组合，而不是预估 microbatch 数量。

每个 physical microbatch 根据实际组成归入 `full`、`middle`、`tail` 或 `pack`，并通过 execution profile 得到 F/B/W 预计时间。cheap score 近似为当前 iteration 所有计算任务的 profile-backed 总时间，同时记录按实际工作量加权的 family efficiency。

该分数只负责筛选，不直接作为最终 pipeline time，因为它还没有表示 stage overlap、通信依赖和 bubble。

### 6.3 候选 Menu

候选集合由以下部分合并而成：

```text
cheap score 的 top-k
union 与最优 cheap score 接近的 slack candidates
union 当前加权曲线中的局部优点
union 最大的直接实测显存可行 chunk
union 当前正在使用且仍安全的 chunk
```

当前 `top-k=4`、`slack=5%`，但补充规则会让最终 menu 超过四个点。候选数量不是算法上的固定常数。

额外保留最大实测可行点非常重要：较大的 chunk 即使单任务 robust efficiency 一般，也可能因显著减少 segment、launch 和通信而在完整 iteration 中获胜。最新实验中的 4352 就是这种情况。

## 7. Weighted DSPP DAG 排序

### 7.1 DAG 结构

只对候选 menu 中的 chunk 构造完整 metadata DAG：

```text
真实 slicing/packing
  -> 当前 microbatch ordering
  -> Slice-V F/B/W schedule
  -> stage-local compute dependencies
  -> continuation dependencies
  -> directed P2P dependencies
```

每个 stage 的 F/B/W compute task 位于该 stage 的串行 compute lane。P2P 使用独立的有向通信 lane；同一 PP edge 上的不同方向不会被合并成一条全局通信序列，因此模型允许 compute/P2P overlap 和双向独立 progress。

### 7.2 Compute 节点代价

离线 local adapter 保留 chunk、query、history 和 packing 的 shape 变化，但无法完整复现真实 PP stage 的 embedding、loss、runtime launch 和 stage-specific 开销。因此使用已有真实 DSPP timeline 对每个 `(stage, op)` 做一次非负仿射校准：

```text
calibrated task time
  = non-negative scale * offline shape prediction
    + non-negative per-task overhead
```

当前校准使用 chunk=4096 的 234 个真实 F/B/W events，形成 `3 stages × 3 operations` 共 9 组参数。shape-dependent 曲线仍来自离线 profile；校准只修正不同 stage/op 的整体比例和每任务固定开销。

### 7.3 P2P 节点代价

P2P 模型支持实测 profile 的分段插值。在缺少直接 P2P profile 时使用：

```text
communication time
  = latency + payload bytes / bandwidth
```

当前 fallback 为 25 GB/s 和 20 us。通信节点保留真实 producer/consumer 依赖，但该模型不试图精确模拟 NCCL 与 attention 对 SM/带宽的竞争。当前实验中通信项对候选预测的影响通常只有约 0.07–0.16 ms，最大约 0.24 ms；它主要用于表示消息数量和依赖，而不是主导排序。

### 7.4 Exhaustive 与 Branch-and-bound

在筛选后的 menu 内：

- audit 模式对所有候选构造完整 DAG，得到 menu 内 exhaustive 结果；
- 默认 branch-and-bound 先使用各 stage compute sum 构造严格 lower bound，只有仍可能优于 incumbent 的候选才构造完整 DAG。

两种模式的目标相同；branch-and-bound 只减少规划工作，不改变候选生成规则。

### 7.5 排序目标

DAG 输出 predicted makespan、predicted P95 和 profile coverage。选择器按 predicted makespan 排序。这里不要求预测的绝对毫秒与真机完全一致；只要候选相对顺序足够准确，就能完成 chunk 选择。

## 8. Pipeline Rank 分工与当前开销

当前 runtime 流程为：

```text
所有 PP ranks：
  获取本轮 logical batches
  -> 查询本地 allocator 状态
  -> 计算各 bank 点的显存可行性
  -> all-reduce 可行 mask

PP first rank：
  候选筛选和 weighted-DAG 规划
  -> 选择 chunk
  -> broadcast 一个 chunk value

所有 PP ranks：
  按相同 chunk rechunk/materialize
  -> 执行本轮训练
```

最新 12 轮离线 selector 的决策时间为 79–464 ms，平均约 217 ms。最新 fixed-run iteration time 没有包含这部分决策时间，因此本报告将“选择质量”和“端到端动态训练收益”分开评价。

planner 只访问下一轮长度 metadata，理论上可以在当前轮 GPU 计算期间由 PP first rank 异步执行；但当前训练代码仍在 iteration 边界同步规划，并且每轮执行显存 mask all-reduce。因而当前结果足以说明 chunk 决策质量，但在声称完整动态策略的端到端加速前，还需要隐藏或降低规划开销。

## 9. 实验设置

主要真机实验配置：

```text
GPU:                         3 × NVIDIA L40
pipeline parallel size:      3
virtual pipeline:            2
layers:                       24
hidden size:                  2048
FFN hidden size:              5504
attention heads / KV groups:  16 / 16
dtype:                        BF16
global batch size:            16 sequences
maximum retained length:      16K
packing:                      baseline residual packing
ordering:                     warmup-short-steady-long
```

长度来自 InfiniPipe Figure-2 风格分布采样，不是手写均匀、双峰或特殊长尾输入。seed 11、19、29 分别生成独立长度序列，超过 16K 的样本在 L40 实验中丢弃。token id 为可复现的合成数据，但每个 seed 下所有 chunk 使用相同 indexed dataset、document 顺序和 token 内容。

每个 seed 运行 4 个 iteration。每个独立进程的第一轮包含 kernel/library cold start，稳态汇总使用 iteration 2–4。

## 10. 实验一：Packing 与任务数量为什么必须进入 Chunk 模型

固定 `chunk=3584` 的消融比较 baseline、关闭 residual packing 和较浅通信窗口：

| 实验 | 平均 iteration | Effective token/s | Physical MB/iter | Packing utilization |
| --- | ---: | ---: | ---: | ---: |
| 完整 DSPP | 2378.0 ms | 18,509.6 | 13.91 | 0.8821 |
| 关闭 packing | 3413.1 ms | 12,896.3 | 22.64 | 0.5259 |
| recv/send window 1/1 | 2518.8 ms | 17,474.8 | 13.91 | 0.8821 |

关闭 packing 后 physical microbatch 总数从 153 增至 249，增加 62.7%；平均 iteration 时间增加 43.5%。该结果说明，性能不能只由有效 token FLOPs 推导：同样的训练样本如果产生更多 physical tasks，就会增加 launch、通信和 pipeline 开销。因此 chunk selector 必须基于该 chunk 下实际构造出的 microbatch 集合，而不能只比较单个 kernel 的吞吐。

通信窗口从 2/2 降到 1/1 后吞吐下降 5.6%，说明 task 数相同的情况下，通信提前匹配和 overlap 仍会影响端到端时间。这支持 weighted DAG 保留 P2P 依赖，而不是只对计算任务求和。

结果位置：

```text
/home/cuiym/src/results/dspp_ablation_fixed3584_baseline_current/
/home/cuiym/src/results/dspp_ablation_fixed3584_no_packing/
/home/cuiym/src/results/dspp_ablation_fixed3584_lookahead1_inflight1/
```

## 11. 实验二：离线四类 Efficiency 曲线

当前 dense profile bank 合并了原有 capacity 和新增的 3328、3584、4352，共 10 个点、150 条 F/B/W shape records：

```text
256, 512, 1024, 2048, 3072,
3328, 3584, 3840, 4096, 4352
```

2K 以上的四类归一化效率如下：

| Chunk | Full | Tail | Middle | Pack | Robust minimum | 四维 Pareto |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2048 | 0.806 | 0.755 | 0.899 | 0.719 | 0.719 | 否 |
| 3072 | 0.943 | 0.994 | 0.965 | 0.971 | 0.943 | 是 |
| 3328 | 0.962 | 0.896 | 0.891 | 0.849 | 0.849 | 否 |
| 3584 | 0.804 | 0.862 | 0.825 | 0.865 | 0.804 | 否 |
| 3840 | 0.938 | 0.957 | 0.932 | 0.975 | 0.932 | 是 |
| 4096 | 0.974 | 0.972 | 0.918 | 0.962 | 0.918 | 是 |
| 4352 | 0.679 | 0.803 | 0.792 | 0.798 | 0.679 | 否 |

这组结果支持三个结论：

1. 不同 family 的效率存在明显差异，单条 full curve 不足以代表真实 DSPP。
2. 3072、3840、4096 构成当前 measured family space 的 Pareto/robust plateau。
3. Pareto 点不能成为唯一 runtime menu。3328 和 4352 虽然不是离线四维 Pareto 点，却分别在后续真实 iteration 中成为最优，因为它们改变了 segment 数和完整 pipeline DAG。

因此，离线 profile 的作用是提供 shape cost 和候选证据，而不是直接宣布一个固定最优 chunk。

结果位置：

```text
/home/cuiym/src/results/dspp_chunk_dense_20260913/compute_profile_dense.json
/home/cuiym/src/results/dspp_chunk_dense_20260913/compute_profile_dense.png
/home/cuiym/src/results/dspp_chunk_dense_20260913/chunk_menu_dense.json
```

为了检查效率是否会随 chunk 一直提高，另一组独立 profile 将容量扩展到 8192。该实验观察到 3072–5120 附近形成高效平台，而 6144 以后 robust efficiency 整体下降；8192 的 robust minimum 为 0.782，低于 4096 的 0.905。由于该扩展实验重复次数较少，它不能证明所有 GPU 在相同位置下降，但足以否定“当前目标环境中更大 chunk 必然更高效”的假设，并说明 profile 范围必须覆盖饱和平台两侧。

扩展结果位置：

```text
/home/cuiym/src/results/dspp_offline_compute_l40_20260910/compute_profile_256_8192.json
/home/cuiym/src/results/dspp_offline_compute_l40_20260910/chunk_efficiency_256_8192.png
```

## 12. 实验三：不同 Chunk 的固定运行性能

下面汇总七个 chunk 在三组采样输入上的 iteration 2–4。每个单元格为三轮总时间；同一 seed 内时间越小越好。

| Seed | 2048 | 3072 | 3328 | 3584 | 3840 | 4096 | 4352 | 最佳固定值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 7584.4 | 6000.0 | 5151.5 | 5153.7 | **4664.7** | 4928.6 | 5491.0 | 3840 |
| 19 | 7925.1 | 6179.5 | 5759.6 | 5636.9 | **5607.3** | 5775.9 | 6282.8 | 3840 |
| 29 | 10711.2 | 7930.0 | 7681.9 | 7552.5 | **7164.6** | 7649.8 | 8402.4 | 3840 |

单位为 ms。三组输入聚合后都以 3840 最快，但逐 iteration 的最优值并不固定：七点 oracle 中，3840 最优 8 次，3072、3328、4096、4352 各最优 1 次。

这说明：

- 3840 是当前三组输入上的强 fixed baseline，但不是每轮都最优；
- 最大可运行 chunk 4352 并不普遍最快；
- 3328 和 4352 这种非 Pareto profile 点仍可能因当前输入的离散 slicing/packing 结构获胜；
- per-iteration selection 的价值主要是避免少数不利 iteration，而不是保证每轮都有大幅收益。

原四点结果：

```text
/home/cuiym/src/results/dspp_multisample_chunk_compare_20260912/
```

新增 3328、3584、4352：

```text
/home/cuiym/src/results/dspp_chunk_dense_20260913/fixed_runs/
```

## 13. 实验四：Per-iteration Menu 与 DAG 选择质量

七点真机 oracle 使用：

```text
{2048, 3072, 3328, 3584, 3840, 4096, 4352}
```

下表中的 menu 是 cheap screening 后实际进入 DAG 比较的集合；“DAG 选择时间”来自相同输入的 fixed-chunk 真机运行，不包含 CPU 决策时间；“七点 oracle”是在上述七个实测点中事后选择的最快值。

| Seed | Iter | Screened menu | DAG 选择及真机时间 | 七点 oracle |
| ---: | ---: | --- | --- | --- |
| 11 | 1 | 3328, 3584, 3840, 4096, 4352 | 4096: 3444.0 ms | 4096: 3444.0 ms |
| 11 | 2 | 3584, 3840, 4096, 4352 | 3584: 1040.3 ms | 3840: 971.5 ms |
| 11 | 3 | 3328, 3584, 3840, 4096, 4352 | 3840: 2314.3 ms | 3840: 2314.3 ms |
| 11 | 4 | 2048, 3328, 3584, 3840, 4096, 4352 | 3328: 1408.4 ms | 3840: 1378.8 ms |
| 19 | 1 | 2048, 3328, 3584, 3840, 4096, 4352 | 3840: 2963.6 ms | 3840: 2963.6 ms |
| 19 | 2 | 3328, 3584, 3840, 4096, 4352 | 4096: 1913.3 ms | 3328: 1863.9 ms |
| 19 | 3 | 3328, 3584, 3840, 4096, 4352 | 3840: 1711.3 ms | 3840: 1711.3 ms |
| 19 | 4 | 3072, 3584, 3840, 4096, 4352 | 3840: 1910.6 ms | 3072: 1908.2 ms |
| 29 | 1 | 3072, 3584, 3840, 4096, 4352 | 3840: 3574.9 ms | 4352: 3289.3 ms |
| 29 | 2 | 3072, 3328, 3840, 4096, 4352 | 3840: 1257.9 ms | 3840: 1257.9 ms |
| 29 | 3 | 3072, 3328, 3840, 4096, 4352 | 3840: 2058.6 ms | 3840: 2058.6 ms |
| 29 | 4 | 3328, 3584, 3840, 4096, 4352 | 3840: 3848.1 ms | 3840: 3848.1 ms |

汇总结果：

```text
七点 oracle 被 screened menu 覆盖： 12 / 12
menu coverage regret：              0%

DAG 精确命中七点 oracle：           7 / 12
DAG 选择总时间：                    27445.4 ms
七点 oracle 总时间：                27009.6 ms
DAG ranking regret：                1.61%
```

排除三个独立进程的第一轮 cold start：

```text
DAG 精确命中：                      5 / 9
DAG 选择总时间：                    17462.8 ms
七点 oracle 总时间：                17312.7 ms
DAG ranking regret：                0.87%
```

该实验最重要的结论不是 7/12 的离散命中率，而是：

1. 候选构造 12/12 保留了七点 oracle，说明当前 menu 足以覆盖这组实测搜索空间。
2. 多数误选发生在运行时间接近的候选之间。即使命中率不高，稳定阶段累计 regret 仍低于 1%。
3. 新增 profile 点找到了原四点集合遗漏的实际最优：seed 19 iteration 2 的 3328，以及 seed 29 iteration 1 的 4352。
4. 当前剩余误差主要来自 DAG 节点代价近似，而不是候选筛选遗漏。

Selector 输出：

```text
/home/cuiym/src/results/dspp_chunk_dense_20260913/seed11_selector.json
/home/cuiym/src/results/dspp_chunk_dense_20260913/seed19_selector.json
/home/cuiym/src/results/dspp_chunk_dense_20260913/seed29_selector.json
```

## 14. 每组实验分别证明了什么

| 实验 | 能支持的结论 | 不能支持的结论 |
| --- | --- | --- |
| fixed-3584 packing 消融 | physical task 数和 packing 对端到端性能影响显著，chunk 模型必须重建真实 microbatch | 不能说明某个动态 chunk policy 一定更快 |
| 四类离线 compute profile | kernel efficiency 依赖 capacity 和 workload family，不能只测 full 或使用固定 FLOPs 比例 | compute-only 最优不能直接等价为 pipeline 最优 |
| 七点 fixed-chunk 对照 | 最优 chunk 随 iteration 改变，最大可行 chunk 和固定单点都没有逐轮最优保证 | 七个离散点不能证明连续 aligned 空间中的全局最优 |
| menu coverage 审计 | 当前筛选规则在 12 轮中没有丢失七点 oracle | 不能证明所有模型、硬件和长度分布上都零 regret |
| DAG ranking 审计 | profile-backed DAG 已能把稳定阶段累计选择损失控制在 0.87% | predicted makespan 不是端到端时间的精确预测 |
| 决策时间记录 | 当前同步 CPU planner 的成本不可忽略，应与选择质量分开报告 | fixed-run iteration time 不能用来证明在线决策开销已隐藏 |

## 15. 当前性能是否足够

从 chunk 决策质量看，当前实现已经足够作为下一阶段调度研究的基础：

- 候选 menu 在当前七点实测空间中达到 12/12 oracle coverage；
- 排除 cold start 后的累计 ranking regret 为 0.87%；
- 误选没有表现出稳定的“总是偏大”或“总是偏小”，继续针对这 12 轮调整权重容易对 L40 和当前输入过拟合；
- 新增点能够被动态纳入 menu，并确实发现了固定三点/四点集合之外的最优值。

因此，没有必要继续为了提高离散命中率而设计 L40 特定规则。后续更高价值的工作是优化 microbatch 组合和调度，因为它们会直接改变 task set、bubble 和 critical path，潜在收益大于继续压缩不足 1% 的稳定阶段 chunk ranking regret。

但从完整在线系统开销看，还不能把 0.87% 直接解释为动态训练相对 fixed 的端到端收益：当前 planner 平均约 217 ms，并且同步执行；最新比较使用 fixed runs 查表获得被选 chunk 的真机时间，没有把决策成本计入 iteration。正式端到端实验应先把下一轮 metadata planning 与当前轮 GPU 执行重叠，或者显著降低同步规划开销。

## 16. 方法的理论与系统意义

当前方法的贡献不在于给出某个固定 chunk 数值，而在于建立以下分解：

1. **硬件效率与输入结构分离。** GPU kernel 的 shape efficiency 离线测量；当前 iteration 的 slicing、packing 和依赖在线构造。
2. **保留完整 profile bank。** Pareto/平台点用于解释，不能删除 specialist 或结构边界点；3328 和 4352 的真实胜出为这一设计提供了直接证据。
3. **先筛选、再建图。** cheap score 处理完整 measured bank，weighted DAG 只处理少量候选，从而避免在训练热路径穷举所有 aligned capacity。
4. **以 critical path 而不是总 FLOPs 为目标。** chunk 改变的不只是节点权重，也改变节点数量、packing 和 pipeline 依赖，因此必须在真实 DSPP DAG 上比较。
5. **方法不绑定 L40。** 换环境时重新采集 profile 和少量 stage calibration；候选规则、显存过滤和 DAG 结构保持不变。
6. **允许得出“动态收益很小”的结论。** 如果生产环境中多个 chunk 都进入饱和平台，且输入结构变化不足以拉开 critical path，最佳策略可能是固定一个稳健 chunk。方法不会预设 dynamic 必然更快。

## 17. 当前限制

1. 最新七点 oracle 只覆盖 2048–4352 的七个离散实测值，不是所有 aligned capacity 的连续全局 oracle。
2. 初始 profile 点和新增三个 dense 点使用的 warmup/repeat 数不同。最终论文实验应统一测量协议并报告方差。
3. stage/op 校准目前只使用 chunk=4096 timeline；换模型或显著扩大 chunk 区间时，适合由 profile 曲线自动选择少量低、中、高锚点重新校准。
4. 解析 P2P 模型没有表示 NCCL 与 compute 的 SM/带宽竞争，但当前配置下通信代价差异较小。
5. 当前显存模型对极小 chunk 的累计历史 KV 状态表示不足；生产环境的 OOM 安全结论需要 history-aware memory profile。
6. 当前 planner 在 iteration 边界同步执行，尚未实现设计中的跨 iteration CPU/GPU overlap。
7. 当前真机输入来自三个采样 seed 和 16K 截断范围，仍需在生产 GPU、更长上下文和更多分布上复核。

## 18. 实现入口

主要代码：

```text
megatron/core/datasets/dspp_compute_profile.py
  离线 physical-shape profile 和 family efficiency

megatron/core/datasets/dspp_execution_cost.py
  exact/interpolated/fitted execution cost 与 stage/op 校准

megatron/core/datasets/dspp_chunk_selector.py
  per-iteration efficiency screening、branch-and-bound 和选择结果

megatron/core/datasets/dspp_chunk_tuner.py
  layout、peak-live slot 和运行时显存估计

megatron/core/datasets/dspp_dag.py
  Slice-V compute/P2P weighted DAG 和 critical path

megatron/core/datasets/dspp_p2p.py
  directed P2P profile/解析代价

megatron/core/pipeline_parallel/schedules.py
  训练侧显存协商、PP first-rank 选择、broadcast 和 rechunk
```

工具和脚本：

```text
examples/calibrate_dspp_compute.sh
tools/profile_dspp_compute_shapes.py
tools/generate_dspp_compute_menu.py
tools/plot_dspp_compute_profile.py
tools/fit_dspp_execution_cost.py
tools/evaluate_dspp_per_iteration.py
examples/run_dspp_multisample_chunk_compare.sh
```

训练侧主要参数：

```text
--dspp-auto-chunk-size
--dspp-chunk-selection per-iteration-menu
--dspp-chunk-menu <chunk_menu.json>
--dspp-execution-profile <execution_profile.json>
--dspp-efficiency-top-k 4
--dspp-efficiency-slack 0.05
--dspp-memory-safety-factor 1.05
--dspp-dag-communication-cost profile
```

## 19. 参考依据

- NVIDIA, Matrix Multiplication Background User's Guide：tile quantization、wave quantization 和矩阵 shape 非单调效率。
- NVIDIA, Optimizing Linear/Fully-Connected Layers User's Guide：Transformer GEMM shape、SM wave 和不同 gradient GEMM 的效率差异。
- Dao et al., FlashAttention-2：attention work partitioning、occupancy 和 sequence dimension 并行。
- Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM：pipeline microbatch 数与 bubble 的关系。
- TeraPipe、DynaPipe、InfiniPipe：长序列切分、变长 microbatch 和 pipeline 调度的相关设计基础。
