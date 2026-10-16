# DSPP 方法实现与当前进展

更新日期：2026-09-06  
用途：面向论文写作和阶段汇报，说明 DSPP 如何从调度算法落实为真实训练系统。具体参数、代码入口和调试工具见 `docs/dspp-implementation-report.md`。

## 1. 当前实现概览

我们已经在 Megatron 训练框架中实现了可运行的 DSPP 原型。当前系统能够把一个包含变长序列的 iteration 转换为固定 shape 的 physical microbatch，并在 3 个 pipeline stage 上执行 V 型 Zero-Bubble 调度。每个 stage 按本地 F/B/W 任务依赖独立推进，真实执行时间由 CUDA 计算和 P2P 通信决定，不使用仿真器中的 FLOPs 时间模型。

当前数据路径为：

```text
变长 logical batch
  -> 长序列切分、短序列与 tail packing
  -> physical microbatch 排序
  -> 每个 stage 的本地 F/B/W 序列
  -> packed attention、跨 segment KV/梯度传递
  -> 双向异步 P2P 和参数更新
```

## 2. 从算法到真实训练的主要实现问题

### 2.1 固定 physical transport 与变长逻辑语义解耦

固定 `chunk_size` 已经保证了 P2P tensor 的 physical shape 相同，因此这里并不是要解决“不同 microbatch 的通信 tensor shape 不一致”。真正的问题是：形状相同的 physical tensor，内部可能代表完全不同的逻辑任务。例如，`chunk_size=3584` 时，一个 tensor 可能是长序列 A 的第 2 个 segment，另一个 tensor 可能是多个短序列的 residual pack；它们需要不同的 sequence boundary、KV history、position ids 和 loss mask。若只传递 tensor 而不保留这些逻辑信息，stage 无法正确执行 attention 或反向梯度路由。

我们的实现把两层语义明确分开：

1. **transport 层**：所有 activation 和 gradient 都 padding 到固定 `chunk_size`，因此 P2P 只处理统一的 `[chunk_size, hidden]` payload，不需要动态 shape handshake；
2. **logical 层**：只广播每个 logical sequence 的紧凑长度，各 stage 根据同一长度确定性地重建 `DsppBatchPlan`，并在本地保存 `sequence_id`、`segment_id`、offset、valid-token 数量、`cu_seqlens` 和 loss mask。

这样，固定 shape 只负责让通信协议简单稳定，metadata 才负责恢复变长训练语义。packed attention 根据独立的 Q/K `cu_seqlens` 隔离不同序列，KV state 按 `(layer, sequence_id)` 保存历史，padding token 通过 loss mask 排除。非整除长度自然产生 tail：`length=16384, chunk=3584` 表示为四个 3584-token segment 和一个 2048-token tail，后者仍可与其他短序列 packing。

### 2.2 将全局调度转化为 stage-local 推进

普通 pipeline 实现通常隐含各 stage 按同一通信节奏推进，这会把本应自然错位的 stage 重新对齐。DSPP 则要求每个 stage 只等待自身任务的真实数据依赖。

实现中先根据同一 batch plan 为每个 stage 生成固定的本地 F/B/W 任务序列，再以 `(logical batch, physical microbatch, model chunk)` 标识 activation、gradient 和 deferred-W 状态。运行时没有全局 action index，也不在 iteration 内使用 barrier 强制 stage 对齐；某个 stage 完成当前任务后即可推进到下一个本地可执行任务。这样仿真器中的 stage-based schedule 被落实为真实的异步 pipeline，而不是由中央调度器逐步发令。

同一长序列的 segment 在 F 入口顺序中保持连续和正序，B 按逆序执行。它们不需要连续占用 GPU wall-clock：两个 segment 之间仍可插入已经就绪的 B、W 或其他 stage-local 任务。

### 2.3 双向 P2P 解耦，以及大消息的有界提前通信

仅把计算和通信放到不同 CUDA stream 上还不够。同一 PP edge 上的 activation 与 gradient 方向相反；如果共享 communicator/stream，一个方向中排队的通信可能阻塞另一个方向。因此实现为每条物理 edge 的两个方向分别创建 NCCL communicator 和通信 stream，使两个方向能够独立 progress。

进一步实验发现，单纯提前提交大量 unmatched NCCL send/recv 在大消息下仍可能长期占用通信资源并阻碍计算推进。最终采用以下 matched 协议：

```text
sender ready
  -> receiver 提前提交匹配的 irecv
  -> receiver 返回 credit
  -> sender 提交 isend
```

receiver 可以在真正消费 tensor 之前提前接收，但 sender 不会产生没有匹配端的长期在途 NCCL 操作。每条 lane 的 receive lookahead 和 send in-flight 深度均有上限；这既保留了大消息的 compute/P2P overlap，又限制了通信 buffer 和在途工作对显存、SM 与 NCCL progress 的压力。这个实现也说明：双向 stream 分离解决的是 head-of-line blocking，而 matched lookahead 解决的是大消息 rendezvous 和资源有界性，两者不可互相替代。

这里的 credit 是发送顺序的控制条件，不等于把整个训练变成 CPU 同步通信。receiver 执行 `post_receive` 时会立即在对应方向的 NCCL stream 上提交 `irecv`，并通过 CPU/Gloo 控制组发送 1-byte credit；它不会等待 payload 到达。credit/ready/ack 分别由独立的 CPU 辅助线程阻塞在控制组的 `recv` 上，主训练线程不会为了等待 credit 而停住，也不由这些线程调用 NCCL。

sender 完成计算后先把 tensor 放入本地 ready queue，并发送 ready index。训练线程在任务边界以及等待其它依赖时检查已经到达的 credit；一旦 credit 可用，就在相应 NCCL stream 上提交 `isend`，随后发送 launch-ack。这个过程是“控制面按顺序匹配、数据面异步传输”：主线程可能暂时只把 tensor 留在 ready queue，但不会阻塞当前无关的计算；而 receiver 只有在即将消费某个依赖时，才等待对应的 launch-ack 和 `irecv` completion。lookahead 使未来的 `irecv` 可以更早提交，因此通信能够与当前 stage 的其它计算重叠。这里的等待是单个计算任务的数据依赖，不是跨 stage 的全局 barrier。

### 2.4 continuation 与短序列混合 packing 的 Attention 语义

一个 pack 可以同时包含长序列当前 segment 和若干独立短序列。长序列需要看到自身历史 KV，短序列之间又必须完全隔离。实现为 Q 和 K 分别构造 `cu_seqlens`：

```text
Q = [A_current, B, C]
K = [A_history + A_current, B, C]
```

并使用 bottom-right causal alignment 保持每条序列的因果边界。KV state 按 `(layer, sequence)` 管理，而不是绑定到某个固定编号的 microbatch。反向过程中，后续 segment 对历史 KV 的梯度被保存并传回前一 segment，使计算满足：

```text
F(A, i-1) -> F(A, i)
B(A, i)   -> B(A, i-1)
```

这种表示使一个 physical task 的内部组成可以变化，而调度器仍只处理统一的 TaskId，不需要知道它是长序列 segment 还是短序列 pack。

### 2.5 面向变长 physical task 的 B/W 分解

DSPP 沿用 Zero-Bubble 的思想，把 backward 拆成 activation-gradient 计算 B 和 weight-gradient 计算 W。但等长实现通常可以用固定的 microbatch/slice 下标组织 deferred W；DSPP 的 segment 数和 pack 组成随输入变化，不能直接复用该索引结构。

当前实现让每个 physical task 在 B 完成后，把对应 Linear wgrad 工作写入所属 model chunk 的 FIFO，W 再按本地调度顺序弹出。队列项直接携带该 physical task 所需的 activation 和 grad-output，因此不依赖其内部包含几条序列或属于长链的第几段。iteration 结束时要求所有 W 和 KV 状态完全排空，避免变长布局导致梯度遗漏。

## 3. 有意义的性能实现细节

实现过程中保留了几项不会改变算法、但对真实执行较重要的热路径优化：

- release 路径依赖构造期生成的合法顺序，不在每个任务前扫描或验证完整 DAG；
- packed attention 使用 CPU metadata 中已有的长度，避免每层通过 GPU tensor `.item()` 触发 host-device 同步；
- 无历史 KV 的 pack 直接复用当前 K/V 及其梯度，不做等价的 `cat`、scatter 和 copy；
- 单 part 使用零拷贝 view，多 part 的梯度回填使用预分配 slice，避免为了接口统一增加 materialization kernel。

这些设计的共同原则是：把正确性约束放在 plan、TaskId 和状态生命周期中表达，而不是在 GPU 热路径反复检查。

## 4. 与 InfiniPipe 的初步端到端实验

当前只报告 DSPP 使用固定 `chunk_size=3584` 的结果；`3580` 是口头表述误差，实际实验值为 3584。双方使用相同的 12 个 iteration 长度输入，后 11 个测量 iteration 的有效 token 总数均为 484,176。第一个 iteration 含初始化和 warmup 效应，因此从比较中排除。

主要配置如下：

| 项目 | 配置 |
| --- | --- |
| GPU | 3 × NVIDIA L40 |
| 模型 | 24 layers，hidden 2048，FFN 5504，16 heads |
| 精度 | BF16 |
| DSPP 并行 | PP=3，VPP=2，每个逻辑 stage 4 layers |
| Global batch | 16 sequences |
| 输入 | 同一份采样自真实数据集长度分布 max-16K 变长序列 trace |
| DSPP chunk | 固定 3584 tokens |
| DSPP P2P payload | activation/gradient 各 14 MiB |

测量结果：

| 指标 | DSPP fixed-3584 | InfiniPipe | DSPP 相对变化 |
| --- | ---: | ---: | ---: |
| 平均 iteration 时间 | 2343.5 ms | 2780.0 ms | -15.7% |
| iteration median | 2358.0 ms | 2871.7 ms | -17.9% |
| iteration P95 | 3220.7 ms | 3566.9 ms | -9.7% |
| 累计有效 token/s | 18,781.8 | 15,832.9 | +18.6% |
| 观测峰值显存 | 32.69 GiB（DSPP rank 0） | 36.79 GiB（InfiniPipe 全局峰值） | -11.2% |

这组结果说明，在当前硬件、模型和输入 trace 上，DSPP 即使使用固定 chunk，也能够稳定处理 tail、短序列 packing 和 14 MiB 双向 P2P，并取得高于 InfiniPipe 的端到端有效吞吐。不过它目前只能作为一个初步端到端结果：双方显存统计口径并不完全相同，而且尚未完成多随机种子、不同长度分布、chunk sweep 和组件消融，因此不能据此概括所有长上下文训练场景。

结果来源：

- DSPP：`src/results/dspp_fixed3584_seq16500/metrics.json`
- InfiniPipe：`src/results/infinipipe_compare_latest/metrics.json`

## 5. 当前完成度与后续工作

已经完成：

- 变长切分、tail 处理和短序列 packing；
- continuation-aware packed attention 和跨 segment KV/梯度生命周期；
- PP=3、VPP=2 的 stage-local V-ZB F/B/W 执行；
- 双向 communicator/stream 分离和有界 matched lookahead；
- microbatch ordering、有效 token/s、iteration 时间、显存和 timeline 观测；
- 固定 chunk 下与 InfiniPipe 的初步端到端对比。

尚未形成论文最终结论：

- chunk size 的吞吐最优选择策略仍需确定；当前自动选择器主要保证显存可行，本文实验使用固定 3584；
- 尚未完成普通 pipeline、packing、ordering 和通信机制的完整消融；
- 尚未覆盖均匀、双峰和长尾等多种长度分布及多次重复实验；
- 利用 pipeline bubble 执行 critical-path attention 的调度尚未实现；
- 当前原型固定为 3-stage、TP=CP=DP=1，通用并行配置仍需扩展。

因此，当前阶段已经证明 DSPP 的核心调度和变长训练语义可以在真实 GPU pipeline 中实现，并给出了一个有竞争力的端到端结果；下一阶段的重点应从“能否正确运行”转向“哪些机制产生收益，以及收益在不同输入与集群配置下是否稳定”。
