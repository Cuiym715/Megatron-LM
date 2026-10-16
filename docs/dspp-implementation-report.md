# DSPP 当前实现与实验报告

更新日期：2026-09-13  
状态：Milestone A、B1、B2+B3、C1+C2 已实现；launch starvation、大消息 matched receive lookahead 和有界 multi-inflight 已完成针对性优化；no-history packed KV 冗余、input-mode 观测不一致、运行时自适应 chunk size 和大消息 P2P 卡死已修复/实现

## 1. 当前结论

本仓库已经形成一条可在 3 张 NVIDIA L40 上运行的 DSPP MVP 训练路径。它不是把仿真器的 FLOPs 时间模型搬进真实训练，而是将仿真器中的变长 microbatch 构造、入口排序和基于 stage 的 V-ZB 顺序落实到 Megatron 运行时，由真实 CUDA 计算、P2P request 和本地依赖推动执行。

当前实现包括：

- 变长 document 到固定 `chunk_size` physical microbatch 的切分与 residual packing；
- continuation 感知的 packed FlashAttention 和跨 segment KV/梯度生命周期；
- 单卡语义基线，以及 PP=3、VPP=2 的三卡 stage-local V-ZB 执行器；
- 每条物理 PP edge 的两个方向独立 NCCL communicator 和通信 stream；
- 分离的 F、B、W task，及严格的 deferred-W FIFO；
- 仿真器规则对应的 warmup/steady physical-microbatch ordering；
- 基于当前 iteration 长度、实际 CUDA 显存和在线 activation 采样的自适应 chunk size；
- 有效 token/s、iteration 时间、packing、显存指标；
- 可关闭的 CUDA-event timeline，以及包在最外层训练循环的 PyTorch Profiler Chrome trace；
- packed-attention 热路径的 slice/select、copy/fill 和 GPU `.item()` 优化。

此前 L40 基线自动化测试为 `34 passed`。最新三卡复现覆盖 max16k 分布下的 8--28 MiB payload，matched lookahead depth 1/2 的完整 12-iteration 实验均正常结束，0 skipped、0 NaN。当前仍不是完整通用实现：通信窗口的 profiler 复核、正式消融、三类长度分布和 attention bubble filling 尚未完成。

## 2. 实现边界与数据流

### 2.1 当前支持配置

- 单卡 B1 correctness 路径，或固定 PP=3、VPP=2 的三卡路径；
- TP=CP=DP=1；
- local Transformer、BF16、FlashAttention、gradient accumulation fusion；
- dropout=0；
- 当前不支持 activation recomputation、MoE、activation offload、vocab pipeline、overlap grad reduce、attention balancing、fast RoPE 和 CUDA Graph；
- packed-attention wrapper 当前只支持模型 micro-batch size 1；一个 dataloader logical batch 内可包含多条 logical sequence；
- attention bubble filling 尚未实现。

不支持的组合会在参数解析或执行器入口 fail fast，不会静默回退到其它 schedule。

### 2.2 训练数据流

```text
DocumentGPTDataset
  -> padded text + 精确 raw lengths
  -> pipeline group 广播 compact lengths
  -> 每个 stage 确定性构造相同 DsppBatchPlan
  -> 固定 [1, chunk_size] tokens/labels/position/loss-mask
  -> iteration 级 physical-microbatch ordering
  -> stage-local V-ZB F/B/W + 双向 P2P
  -> token-weighted loss / deferred W / optimizer step
```

长度显式传递，不能通过 pad token 值反推，因为该 token 也可能真实出现在 document 内。所有 pipeline activation/gradient payload 保持固定 shape，变长 metadata 不参与逐消息动态 shape handshake。

## 3. 变长 microbatch 构造与 attention 语义

### 3.1 切分与 packing

`DsppBatchPlan` 将每条 logical sequence 表示为带有以下 metadata 的 segment：

```text
sequence_id, segment_id, token_offset, token_length,
sequence_length, segment_count, estimated_flops
```

构造规则为：

1. 每个完整 `chunk_size` segment 单独成为 physical microbatch；
2. 每个长序列 tail 建立独立 residual bin；
3. 短序列按从长到短放入可行 bin；
4. 优先选择累计 proxy cost 最低、剩余 token 最匹配的 bin；
5. physical tensor 始终补齐到固定 chunk shape，padding 的 loss mask 为 0。

`estimated_flops` 只用于 CPU 侧构造和无真机 cost 时的排序 proxy。真实训练没有虚拟时钟、sleep 或 FLOPs 驱动的 ready queue。

### 3.2 Packed FlashAttention 与 KV state

一个 residual pack 可以同时包含长序列 continuation 和若干独立短序列。例如：

```text
Q = [A_current, B_current, C_current]
K = [A_history + A_current, B_current, C_current]
```

独立的 `cu_seqlens_q`、`cu_seqlens_k` 和 bottom-right causal alignment 使 A 看到自己的历史，同时隔离 B、C。iteration-scoped `DsppSequenceKVState` 按 `(layer_id, sequence_id)` 路由历史 K/V 和 pending gradient。

正确性依赖显式的数据结构和执行顺序：

```text
F(sequence, segment i-1) -> F(sequence, segment i)
B(sequence, segment i)   -> B(sequence, segment i-1)
```

这不意味着 chain 必须连续占用 GPU wall-clock：入口 F 顺序中同一 chain 连续，但实际 stage-local schedule 可以在相邻 segment 之间插入 B/W 或其它已经就绪的工作。release 热路径不做通用 DAG 扫描；构造期 validation 和 iteration 结束的 KV/W empty-state assertion 负责发现结构性错误。

### 3.3 Loss 语义

全局排序后，一个 physical task 的 B 可能早于其原 logical batch 的其它 F，因此 loss stage 对每个 physical output 立即计算：

```text
loss_i = sum(token_loss_i * loss_mask_i) / iteration_valid_tokens
```

iteration 日志 loss 为所有 `loss_i` 之和。这样 packing、切分和排序不改变有效 token 分母，不同 logical accumulation group 也按有效 token 加权。

### 3.4 运行时自适应 chunk size

本节说明显存安全和动态 rechunk 的底层机制。当前最终的性能选择策略已经从“选择最大显存可行值”扩展为“离线四类 compute profile + per-iteration 候选筛选 + weighted DSPP DAG”；完整方法、理论依据和最新实验统一见 `docs/dspp-chunk-size-selection.md`。下面保留的最大可行选择是 `memory` policy 和显存过滤基础，不代表当前推荐的性能目标。

固定模式下，最长切分段数原本没有单独求解，它直接由以下式子得到：

```text
max_segment_count = max(ceil(training_sequence_length / micro_seq_length))
```

这里 `training_sequence_length = raw_document_length - 1`，因为最后一个 raw token 只作为前一个 token 的 label。最终饱和 case 中 `micro_seq_length=256`，最长 document 约 768，因此最长切分为 3 段。

新增 `--dspp-auto-chunk-size` 后，每个 pipeline rank 在当前 accumulation iteration 已经取到全部 logical batch、但尚未开始 F/B 前执行以下低频 CPU 规划：

1. 从当前 batch plan 读取真实 sequence lengths，而不是使用数据分布的静态上界；
2. 用 `torch.cuda.mem_get_info()`、`memory_allocated()` 和 `memory_reserved()` 得到当前设备容量、外部空闲显存和 PyTorch 可复用 reserve；
3. 按 memory utilization、显式 reserve 和已分配 model/optimizer/input memory 算本 rank 可给新增 activation 使用的预算；
4. 对待检查的 chunk 重新构造 metadata-only packing/order/Slice-V schedule，统计真实 F→B activation 和 B→W deferred-weight 状态的峰值存活 slot；
5. 用 `peak_live_slots * chunk_size * measured_bytes_per_live_token * safety_factor`，再加每条 incoming/outgoing lane 各一个 payload 的 window-1 通信 reserve，判断显存可行；
6. 先检查当前最长序列的对齐上界；若不可行，对 alignment grid 做二分，选择显存可容纳的最大值；
7. pipeline group 对各 rank 的本地选择做 `MIN`，保证所有 stage 使用同一个固定 P2P shape，然后用保留的原 sequence view 重新 materialize token/label/position/loss-mask。

第一轮没有可靠 activation sample，因此严格使用 `--micro-seq-length` 作为安全 bootstrap。该轮只在已有 PyTorch allocator 记账上调用 `reset_peak_memory_stats()`/`max_memory_allocated()`，不加入逐 task 检查和同步；测得的动态峰值除以调度估算的 peak-live tokens，形成每个 rank 自己的 bytes-per-live-token；后续只提高、不降低该 high-water envelope，避免便宜 shape 连续出现后低估未来的高显存 layout。若已有离线 profile，也可用 `--dspp-activation-bytes-per-token` 跳过 bootstrap 学习。

当前临时策略按要求直接使用“显存可容纳的最大 chunk”，不再尝试在运行时优化 padding、task 数或预测吞吐。需要注意，这不等价于理论吞吐最优：较大 chunk 会减少 P2P 次数、Python/CUDA launch 和 KV 拼接，但可能增加 padding/MLP 工作、降低 pipeline task 数并增大峰值 activation。例如同一 iteration 包含两组 `768,512,192,128` 训练长度时：

| chunk | 最长段数 | physical tasks | 固定 shape token capacity |
| ---: | ---: | ---: | ---: |
| 256 | 3 | 14 | 3584 |
| 512 | 2 | 8 | 4096 |
| 768 | 1 | 6 | 4608 |

临时策略在三者都满足显存时会选 768。chunk 默认以 256 tokens 对齐，但不会限制为 2 的幂；当前最长长度只向上对齐到下一个候选，例如 700 对齐到 768，而不是跳到 1024；若显式 hard maximum 截断了对齐值，则保留该 exact boundary。二分搜索假设估算的 peak activation 随 chunk 单调不减；128K、alignment=256 时最多约 9 次 layout 检查，而不是逐项检查 512 个值。

原先的最小 chunk=256 来自 FlashAttention/矩阵形状和 InfiniPipe 的经验粒度，并不是测量得到的饱和阈值。随后对当前 L40/hidden-2048/FFN-5504/BF16/每逻辑 stage 4 层配置执行了 32--8192 的离线 CUDA-event profile；在相同输入的受控复核中，无 history 满 chunk 的归一化 F+B+W 计算率在 2048、4096、8192 时分别为 28.85、47.09、50.87 TFLOP/s。以“达到最高可行测量点 90% 的最小候选”为规则，4096 达到 8192 的 92.6%，因此 4096 被记录为该配置的最小高效 chunk，但不作为内存可行性的硬下限。L40 saturated preset 继续使用 `DSPP_MIN_CHUNK_SIZE=256`，使太长的序列在显存无法容纳 4096 时仍可回退到更小 chunk；4096 后续只应用于性能偏好或代价模型。这个效率拐点与 GPU 和模型 shape 绑定，其他环境应重新离线测量。完整 profile 位于 `/home/cuiym/dspp_artifacts/dspp_chunk_saturation_l40_20260906/`。

这个设计借鉴而不照搬 InfiniPipe：其 `CompProfiler` 用真实 F/B 前后的显存差拟合 `act_per_token`，`CostModel.token_capacity_with_ckpt()` 从显存容量减 model states 后计算 token capacity，`Splitter` 再按 256/512 对齐；它并不是每个 iteration 直接询问 CUDA 得到一个最优值。DSPP 当前实现改为在线读取本机状态并按当前 iteration 重建真实 Slice-V liveness，因此不绑定 L40 容量或固定模型尺寸。

当前默认不再限制 P2P payload，`--dspp-max-p2p-bytes 0` 让显存模型独立决定 chunk 上限。该参数仍可显式设置为正数，作为用户选择的 payload/chunk hard bound，而不是 correctness guard。

## 4. 三卡 V-ZB 与通信

### 4.1 V 型路径和均衡层布局

当前固定路径为：

```text
F: rank0/chunk0 -> rank1/chunk0 -> rank2/chunk0
   -> rank2/chunk1 -> rank1/chunk1 -> rank0/chunk1 -> loss

B: rank0/chunk1 -> rank1/chunk1 -> rank2/chunk1
   -> rank2/chunk0 -> rank1/chunk0 -> rank0/chunk0
```

默认 `--dspp-v-layer-layout balanced` 将 Transformer layers 均匀分给 6 个逻辑 stage。最终 24-layer case 每个逻辑 stage 为 4 层：

```text
rank0: chunk0 layers 1-4,   chunk1 layers 21-24
rank1: chunk0 layers 5-8,   chunk1 layers 17-20
rank2: chunk0 layers 9-12,  chunk1 layers 13-16
```

`legacy-output-slot` 仍保留旧 `(num_layers + 1)` 分配方式，仅用于兼容对照。

### 4.2 Stage-local schedule

每个 rank 根据自己的固定 F/B/W 列表独立推进，不使用全局 action index，不在 iteration 内加入用于重新对齐 stage 的全局 barrier 或 `cuda.synchronize()`。activation、gradient、KV、loss 和 W queue 以 physical TaskId 管理。

V 型首尾 embedding 位于 rank 0 的两个 vchunk。初始化时本地复制 tied weight；backward 后本地合并两份 gradient，使它们持续执行相同 optimizer update。

### 4.3 双方向 P2P 解耦

PP=3 的每条 edge 创建 `next`、`prev` 两个 NCCL group；方向按物理传输方向绑定，而不是按 activation/gradient 类型绑定。当前所有 payload 都采用 matched receive 协议，不存在按消息大小绕过 credit 的运行时分支：sender 计算出 tensor 后先通过 CPU/Gloo 发送 ready index；receiver 只有看到该 index 后才分配对应 buffer、提交 NCCL `irecv` 并返回 1-byte credit；sender 收到 credit 后才在训练主线程提交匹配的 NCCL `isend`。因此通信可以早于 receiver 的实际消费 task 启动，但不会产生长期驻留的 unmatched NCCL send/recv。

默认 `--dspp-recv-lookahead=2`、`--dspp-send-inflight-depth=2`。每条 incoming lane 最多预发布两个已经收到 sender-ready 通知的 future recv，每条 outgoing lane 最多保留两个已经获得 matched credit 的 in-flight send，lane 内仍严格按 receiver schedule 顺序提交。sender 提交 send 后返回 1-byte launch acknowledgement；receiver 只等到匹配 send 已提交，便恢复普通 `Work.wait()` 的 CUDA-stream 异步依赖，不在 host 等 payload 完成。等待控制依赖时仍协作推进其他 ready/credit/send，避免重新形成跨 edge 的运行时等待环。

CPU 辅助线程只处理 ready/credit/ack，不调用 NCCL/CUDA。auto chunk 显存模型按每个 neighbor 保守预留 `send_inflight_depth + 1 current recv + recv_lookahead` 个 payload；默认深度 2 对应每个 neighbor 五个 payload。深度可通过启动参数降回 1，用于消融或显存极紧张的环境。

此前 Nsight 审计确认中心 stage 出现四条 P2P stream，并且每条 edge 两个方向均有实际 send/recv。但小通信实验只证明了解耦和异步提交机制，未证明可观的 compute/P2P overlap 收益。原始 Nsight 中间产物已按本次整理要求删除，结论保留在本文。

## 5. Ordering 与 warmup 修复后的定义

默认 `--dspp-microbatch-order warmup-short-steady-long`：

- warmup 先选 workload 接近的 short-only pack；数量不足时按短到长加入完整 long chain；
- steady 将剩余 long chain 按 sequence length 从长到短放入；
- 剩余 short-only task 按 workload 从大到小；
- 每条 long chain 构成一个 schedule group，short-only task 构成长度 1 的 group；
- warmup 边界可以落在 chain 内，它只是入口位置标签，不是运行时 fence，也不会扩大 warmup。

当前最终 case 的最大切分段数为 3，13 个 physical microbatch 的 schedule group sizes 为：

```text
[1, 1, 1, 2, 2, 3, 3]
```

ordering warmup 为前 7 个 physical task；三个 stage 在开始第一个 B1 前的 `F0 + F1` 数量均为 10。此前按 dataloader logical batch 聚组导致 warmup 过大的实现已经修复，其结果已删除。

排序 cost 来源依次为：

1. `--dspp-order-profile` 提供的同 shape 真机 CUDA-event cost；
2. 未匹配 shape 使用 metadata proxy。

这两个值都只影响 CPU 入口排序，不代表运行时执行时间。

## 6. B/W 分解审计

当前 B/W 参考了 `zero-bubble-pipeline-parallelism` 的核心边界：

- B 通过 PyTorch autograd 立即计算 activation dgrad；FlashAttention backward、Linear dgrad、normalization/activation backward 和 KV history gradient routing 都留在 B；
- tensor-parallel Linear 的 wgrad GEMM 被捕获并按 physical task flush 入 queue；
- W 从对应 vchunk FIFO 弹出工作并累加到 `weight.main_grad`。

DSPP 没有复制参考实现按固定 `chunk x seq_split_idx` 的队列，因为这里 split count 可变，physical task 既可能是长 sequence segment，也可能是短 sequence pack。当前 queue item 直接持有该 physical task 的 activation 和 grad-output，因此与内部组合方式无关。

保留的 correctness 修复是：W queue underflow 不再用 `min()` 静默少执行，而是立即报告 rank/chunk/请求数/可用数；iteration 结束继续 `assert_empty()`。

曾尝试移植 deferred sequence-parallel/layout preprocessing。在当前 TP=1、SP=false case 中，它没有减少 kernel，反而增加 Python/dispatcher 开销；相关代码已经完全回退。也没有引入跨 physical task 的 wgrad 合并。

## 7. Launch starvation 优化

最终保留的优化位于 `megatron/core/kv_cache/dspp_packed_attention.py`：

1. custom autograd 直接接收完整 `[chunk, 1, heads, head_dim]` physical Q/K/V，将 valid-prefix view 和必要的 pad/unpad 收进 Function 内，避免外部 Slice/Select backward chain；
2. 满 chunk 的 output 和 gradient 使用 view，不做无意义补零和 slice copy；
3. 无 history 的 K/V 直接复用 current tensor（包括 multi-short pack）；对应的 all-final backward 直接复用 FlashAttention dK/dV，不分配或 scatter；带 history 的 multi-part forward 仍拼接，continuation+short gradient 保留预分配 slice 写入；
4. Q heads 与 KV heads 分别保存 physical shape，支持 GQA/MQA；
5. 用 CPU metadata `meta.cu_seqlens_k[-1]` 替代每层 GPU `cu_k[-1].item()`，去掉热路径同步。

相同 PyTorch profiler 条件下，每个 4-layer B1 的 non-NCCL kernel median：

| Stage | 优化前 | 优化后 | 变化 |
| ---: | ---: | ---: | ---: |
| 0 | 274 | 262 | -12 |
| 1 | 260 | 248 | -12 |
| 2 | 260 | 248 | -12 |

同一 profiled iteration 中，每个 stage 的 `SelectBackward` 从 156 次降为 0，`SliceBackward` 从 232 降为 208；`copy_` 约从 838--864 降到 482--508，`fill_` 约从 388--401 降到 232--245。attention 热路径 scalar read 在 stage 1/2 从 104 次降为 0，stage 0 从 143 降为 39；stage 0 剩余项来自 loss/log 等路径。

该优化只是缓解 starvation。B 仍有约 250 个 non-NCCL kernels/4-layer chunk，来源主要是模型正常 backward 以及有 history 时必要的 K/V 拼接和梯度累积。

### 7.1 为什么某些 B 方块很短：host launch overlap 与 launch starvation

此前 timeline 中出现过某些 stage 的 B1 只有约 1.7--2.3 ms，而同类 B 在其它位置达到 20--30 ms。这个现象不能直接解释为“该 B 少算了”或“反向只执行了一部分”。当前至少有三个不同的时间量：

1. **host interval**：PyTorch autograd 在 CPU 上遍历反向图、创建临时对象并逐个发起 CUDA kernel 的时间；
2. **compute-stream CUDA-event interval**：timeline 在 `task_batch.backward()` 前后于 compute stream 上记录 begin/end event 的时间；
3. **kernel interval**：每个具体 CUDA kernel 真正占用 GPU 的时间。

SVG 使用第 2 种时间，PyTorch Profiler 中的 `DSPP/B0`、`DSPP/B1` annotation 主要反映第 1 种时间。二者不相等是正常的，因为 CUDA kernel launch 通常是异步的。

一个简化例子如下：

```text
情况 A：launch 开销被前一项 GPU 工作隐藏

CPU: 进入 B ──(遍历 autograd、发起很多 kernel，18 ms)── 记录 end
GPU:       前一项工作还在运行 ───── B 的 kernel 实际运行 2 ms ───
          begin event                         end event

结果：CPU annotation 约 20 ms，CUDA-event B 方块约 2 ms。

情况 B：GPU queue 已经空了，launch 开销暴露

CPU: 进入 B ──(18 ms 后才发起第一个 kernel)──────────────
GPU: begin event ──空等 18 ms── B 的 kernel 运行 2 ms ── end event

结果：CPU annotation 仍约 20 ms，但 CUDA-event B 方块约 20 ms。
```

这就是“CPU launch time overlap”：CPU 正在为当前 B 发起后续 kernel 时，GPU 可能还在执行之前已经排队的工作，因此 host 开销被隐藏在 GPU 工作下面。反过来，如果 GPU 队列被 CPU launch 速度耗尽，就会出现真正的 launch starvation，表现为 kernel 之间有空洞，CUDA-event 区间反而会变长。

因此“短 B 方块”与“launch starvation”不是同义词：短方块通常表示 GPU stream 上可见的 device work 很短，可能是 host 开销被隐藏，也可能叠加了不同 attention history、padding/valid-token 量、通信资源竞争或跨 stream 排队差异。只有把 CPU annotation、CUDA kernel 区间和 stream idle gap 对齐后，才能判断其中哪一种占主导。

现有 trace 已经显示出这种 host/device 时间差：部分轻量 timeline 中的 B1 CUDA-event 约为 1.7--2.2 ms，而对应 profiler 的 `DSPP/B1` CPU annotation 为数十毫秒。与此同时，一个 4-layer B 仍包含约 250 个 non-NCCL kernel，因此 CPU launch/dispatcher 开销确实有足够机会成为瓶颈；但这组证据不能单独证明每个短 B 都存在 GPU starvation。

这也是为什么当前优化优先减少无意义的 kernel 和 host dispatch，而不是仅凭 SVG 方块长度改变调度语义：去除外部 `SelectBackward`、部分 `Slice/Copy/Fill` 和重复的 GPU `.item()` 后，B 的实际工作保持不变，但 CPU 需要发起的操作减少，launch gap 更不容易暴露。

## 8. 观测方式

### 8.1 默认指标

默认路径仅记录 iteration 级 CUDA Event，并在已有同步点后读取，输出：

- effective tokens/s；
- iteration median/P95；
- packing utilization 和 schedule padding ratio；
- rank-0 peak allocated/reserved memory。

Megatron 原有固定 `seq_length` tokens/s 仍打印，但不作为 DSPP 有效吞吐。

### 8.2 轻量 timeline

指定 `DSPP_TIMELINE_DIR` 和 one-based `DSPP_TIMELINE_ITERATION` 后，仅选中一个 iteration：

- 在默认 compute stream 上用 CUDA Event 包围每个 F0/F1/B0/B1/W0/W1；
- 每个物理 stage 生成一条合并 compute lane 的 SVG，矩形标注入口 microbatch id；
- 生成每 rank JSON、critical span/bubble summary、cost JSON 和可读 microbatch 构造/排序报告。

该 duration 是 task 在 compute stream 上的墙钟包络，不是内部 kernel duration 纯和；并发 NCCL 可能造成排队或资源竞争。P2P API range 不画进 compute 方块。

### 8.3 PyTorch Profiler

PyTorch profiler 包在 `training.py` 的最外层训练循环，使用 schedule 只激活指定 iteration。DSPP task/P2P 仅用 `record_function` 添加可读标签。设置 `with_stack=False`，避免此前 `with_stack=True` 对 Python/autograd launch 的明显放大。生成的每 rank `.pt.trace.json` 可直接用 Chrome tracing 或 Perfetto 打开，并能同时看到计算与 NCCL 活动。

profiler run 用于定位 kernel、CPU launch、依赖和通信，不作为端到端性能数据。

## 9. 最终 L40 实验

### 9.1 配置和数据

```text
GPU:                     3 x NVIDIA L40 (46068 MiB)
PP / VPP / TP / CP / DP: 3 / 2 / 1 / 1 / 1
Transformer layers:      24 (每个逻辑 stage 4 层)
hidden / FFN / heads:    2048 / 5504 / 16
dtype:                   BF16
chunk size:              256
training length pattern: 768, 512, 192, 128
logical batch:           2 x 4 sequences / optimizer iteration
activation P2P payload:  1 MiB
```

数据不是公开语料，而是 `tools/build_dspp_synthetic_dataset.py` 生成的可复现 Megatron mmap dataset：32 个 document，训练长度循环为 `768,512,192,128`，token 为合成 id。它用于调度、correctness、性能形态和 timeline 验证，不用于评价语言模型收敛。

### 9.2 无 PyTorch profiler 的最终结果

3 个 iteration 的 GPU train-step 时间为：

```text
3401.304 ms, 1279.343 ms, 1569.872 ms
median:                    1569.872 ms
P95:                       3218.161 ms
aggregate effective tok/s: 1750.895
packing utilization:       0.9500
peak allocated rank 0:     10503.1 MiB
skipped / NaN:             0 / 0
```

首个 iteration 包含初始化效应；只有 3 个样本，因此 P95 不应作为稳定性能结论。iteration 2 含 13 个 physical microbatch，其 timeline：

```text
critical span: 1186.607 ms
stage 0/1/2 compute utilization: 98.34% / 77.49% / 84.92%
B0 median: 25.960 ms
B1 median: 25.253 ms
F0 median:  9.340 ms
F1 median: 14.587 ms
W0/W1 median: 2.566 / 2.572 ms
```

与 warmup 修复后、launch 优化前的正确 baseline 相比：

- iteration median 降低 5.85%；
- timeline critical span 降低 4.35%；
- aggregate effective token/s 提高 4.16%；
- B0/B1 median 分别降低 4.83% / 7.02%；
- rank-0 peak allocated memory 减少约 68 MiB。

旧 baseline 目录已经删除，以上紧凑对比保留用于解释最终优化来源。

### 9.3 GPU 饱和、大消息修复与两轮通信优化

该 24-layer/hidden-2048/chunk-256 preset 在活跃计算区间的 1 秒采样曾观察到 99--100% SM busy，说明 task 已脱离 hidden=64/chunk=32 的纯小 kernel 微基准；这不等于整个 pipeline 或 Tensor Core roofline 达到 100%。

旧实现中 `hidden=4096, chunk=1024` 的 8 MiB activation payload，以及 `hidden=2048, chunk=512` 的 2 MiB payload，曾出现未匹配 NCCL send 长期驻留并阻塞后续 compute progress。最初加入的 receiver-at-consume credit/sender-launch-ack 修复了大消息卡死，但把大消息的通信启动推迟到 receiver 实际依赖点，明显缩小了 compute/P2P overlap。

#### 9.3.1 已删除的小消息 eager fast path

2026-09-06 曾实现每条有向 lane 最多 1 MiB 的未匹配 eager-send 字节窗口。固定 1 MiB payload 的纯 A/B 中，iteration median 从 2870.1 ms 降至 1758.0 ms（-38.7%），P95 从 5289.5 ms 降至 3644.9 ms（-31.1%），aggregate effective token/s 从 1014.0 提高到 1570.3（+54.9%），峰值显存不变。该实验说明 DSPP 的性能高度依赖提前通信，而不是说明 1 MiB 是通用阈值。

max16k 对比中的稳态 chunk 为 2048--7168；在 hidden=2048、BF16 下，每条 activation/gradient P2P 为 8--28 MiB，因此 1 MiB eager fast path 几乎不覆盖目标训练场景。当前代码已经删除 `--dspp-eager-p2p-bytes`、`DSPP_EAGER_P2P_BYTES` 和对应分支，所有消息统一使用 matched receive。若未来在其他模型、较小 hidden size 或较小 chunk 下重新出现大量不超过 1 MiB 的 P2P，可以从 `/home/cuiym/dspp_artifacts/dspp_eager_window_20260906/` 的 A/B 结果回忆并重新评估“有界累计未匹配字节窗口”，但不能无条件恢复 unmatched NCCL。

#### 9.3.2 第一轮：matched receive lookahead

直接提前提交 unmatched `irecv` 的尝试在第一轮 1 MiB payload 就发生 GPU progress starvation，说明未匹配 recv 与未匹配 send 一样不能长期驻留；失败日志保存在 `/home/cuiym/dspp_artifacts/dspp_large_recv_lookahead_20260906/failed_unmatched_recv_lookahead1.log`。最终方案改为 `sender ready notification → receiver 提交 matched irecv → receiver credit → sender isend`，使任意大小的消息都能在 receiver 消费前开始传输，同时保证 NCCL 两端已经配对。

完整 max16k 12-iteration 实验覆盖 8--28 MiB payload。排除 bootstrap 后，严格 receiver-at-consume credit 的平均 iteration 为 4623.6 ms、median 4847.5 ms、P95 6808.3 ms、有效吞吐 9519.8 token/s；matched lookahead depth 1 分别为 3074.7 ms、3017.0 ms、4130.9 ms、14315.6 token/s。平均 iteration 降低 33.5%，P95 降低 39.3%，有效吞吐提高 50.4%，12/12 iteration 正常结束且没有 OOM、NaN 或通信卡死。产物位于 `/home/cuiym/dspp_artifacts/dspp_large_recv_lookahead_20260906/`。

#### 9.3.3 第二轮：有界 multi-inflight depth 2

在 matched 协议不变的前提下，将每条 lane 的 receive lookahead 和 send in-flight 深度分别从 1 扩展到 2，并对深度 1/2/4 做了相同 8 MiB 短测：3864.7、3733.1、3893.6 ms。depth 4 已出现过度排队或通信资源竞争，未继续扩大。

同一份当前代码下，depth 1 与 depth 2 又分别完成完整 max16k 12-iteration。排除 bootstrap 后，平均 iteration 从 3195.3 降至 3107.7 ms（-2.74%），有效吞吐从 13775.2 提高到 14163.7 token/s（+2.82%），P95 从 4642.2 降至 4358.2 ms（-6.12%）；median 从 2948.9 轻微增加到 2965.2 ms（+0.55%）。rank-0 peak allocated 只增加约 4 MiB，peak reserved 增加约 30 MiB，auto chunk 决策完全相同。两组均完成 12/12 iteration，无 OOM、NaN 或卡死，因此当前默认采用 depth 2，并保留 depth 1 作为消融和低显存配置。结果位于 `/home/cuiym/dspp_artifacts/dspp_inflight_depth_20260906/`。

删除小消息 eager 分支后，默认 depth 2 又执行了一次两轮 smoke：1 MiB bootstrap 为 9291.6 ms，8 MiB 第二轮为 3719.8 ms，chunk 依次为 256、2048，均正常完成且没有 OOM、NaN、skip 或通信卡死；相关通信与 chunk 显存模型单测为 9 passed。该结果保存在 `/home/cuiym/dspp_artifacts/dspp_no_small_eager_20260906/`，确认当前代码不依赖任何小消息特例。

### 9.4 Figure-2 分布采样的显存检查

使用 `src/numerical-simulator/scripts/sample_infinipipe_seq_lens.py` 按 InfiniPipe Figure-2 分布以 seed 19 采样；L40 测试丢弃超过 8K 的极端项后取前 32 条，而不是手工指定长度。四轮输入使 auto chunk 依次选择 256、1024、3584、6656，对应 1、4、14、26 MiB P2P payload，4/4 iteration 均完成且无 OOM、skip 或 NaN。

早期 depth-1 机制完成过同一 4/4 run，最后一轮 rank-0 peak allocated/reserved 为 24.59/30.75 GiB；选择时 high-water activation 加显式通信 reserve 的动态显存估计为 29.66 GiB，budget 为 30.83 GiB，实测动态 allocator peak 为 15.54 GiB。当前 tuner 已进一步按每个 neighbor 预留 `send_inflight_depth + 1 current recv + recv_lookahead` 个 payload，默认 depth 2 时为五个，并继续使用只升不降的 bytes/live-token high-water envelope。若未来继续扩大任一深度，必须同步扩大该 reserve，不能只修改通信队列。

随后把采样上限提高到 32K，并从 seed 19 的 512 个候选中确定性打乱后选择 32 条，使输入实际覆盖 8K--32K；所选最长序列为 24681。相同 24-layer/hidden-2048/global-batch-8 L40 配置完成前两轮，第三轮因同时包含 24681 和 13600 两条长序列，在计算开始前被显存 guard 拒绝：最小 chunk=256 时各 rank 估计需要约 70.17--73.59 GiB，而预算只有约 30.82--33.04 GiB。该结果没有发生 CUDA OOM 或通信卡死，但 4 轮训练没有完整结束，说明单序列 32K 上限不能保证任意 iteration composition 在 48 GB L40 上可行；完整记录位于 `/home/cuiym/dspp_artifacts/dspp_sampled_memory_cap32k_20260906/`。

## 10. 测试结果

环境：Python 3.12.3、PyTorch 2.10.0a0 NVIDIA build、FlashAttention 2.7.4.post1、3 x NVIDIA L40。

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && \
   /workspace/src/venvs/megatron/bin/python -m pytest -q tests/unit_tests/dspp'
```

此前 L40 基线结果（2026-09-04）：

```text
34 passed, 0 failed
```

2026-09-05 KV fast path 与 input-mode canonical order 修改后，在恢复 CUDA 可见性的 3 x L40 容器中实际结果为：

```text
45 passed, 0 skipped, 0 failed
```

新增并通过两条 KV-state 回归：all-independent pack 的 K/V 与 dK/dV 对象复用，以及长序列首段仍等待后继 historical-KV gradient。新增的 all-independent packed FlashAttention output/Q/K/V gradient oracle 也已在 L40 上通过。

input-mode 新增长尾回归，验证 `entrance_order == flatten(schedule_groups)`，并固定检查 `P0,P1,P3,P2,P4,P5,P6` 这一 stable-input-chain 顺序。该修复只统一 timeline/report 的任务编号来源，不改变 Slice-V schedule、F/B/W 顺序或运行时性能。

自适应 chunk 新增 8 条回归，覆盖 Slice-V peak-live 估算、首轮安全 bootstrap、非 2 次幂候选、显存不可行报告、high-water activation 模型、最大可容纳选择，以及动态 rechunk 后 token/label/loss-mask 不变。小模型三卡真机测试在相邻 iteration 中实际选择 32、16、64 三种 shape，3/3 iteration 完成且无 skip/NaN；这只用于验证动态 shape 正确性，不作为性能对比。

饱和 L40 配置的旧三卡验证曾在 1 MiB guard 下完成 3/3 iteration。大消息修复后，关键两轮复现改为 chunk 256（1 MiB）和 chunk 768（3 MiB），分别完成于 5217.2 ms 和 861.7 ms；rank 0 peak allocated 分别为 9.27 GiB 和 11.13 GiB。该复现保存在容器 `/tmp/dspp_credit_ack_fix_metrics.json`，未向 `results/` 增加临时实验目录。

隔离 profiler 使用 `(512, 16, 128)` BF16 K/V 和 4 个 independent short item。旧 routing 每次包含 2 次 `aten::cat`、2 次 `aten::empty` 和 8 次 `aten::copy_`；新 direct path 对应操作均为 0。200 次调用的单次平均墙钟约从 0.174 ms 降至 0.008 ms。该结果只衡量被删除的 KV routing 冗余，未包含 FlashAttention，不能外推为 iteration 级加速比例；完整训练性能尚未重新测量。

覆盖范围包括构造/packing、token/label/position/loss mask、跨序列 attention 隔离、Q/K/V 和模型参数梯度 oracle、GQA、KV drain、token-weighted accumulation、ordering/group/warmup、timeline/metrics、不同 physical shape 的 W FIFO 与 underflow。

最终 3 x L40 run 完成 3/3 optimizer iteration；第一步 loss 为 `10.74023`，所有 iteration 均无 skip、NaN 或 deadlock。

## 11. 最终保留结果与复现

`results` 只保留两组：

1. `src/results/dspp_final_no_profiler/`：最终无 PyTorch profiler 运行；包含 `train.log`、`metrics.json`，以及 iteration 2 的轻量 CUDA-event timeline、SVG、microbatch report 和 summary。轻量 timeline 只用于任务时序，端到端指标仍未启用 PyTorch profiler。
2. `src/results/dspp_final_profiler/`：相同实现、数据和 ordering 的 PyTorch profiler 运行；包含三个 rank 的 Chrome trace、训练日志及配套 timeline metadata。

普通最终运行：

```bash
TRAIN_ITERS=3 \
DSPP_METRICS_PATH=/workspace/src/results/dspp_run/metrics.json \
DSPP_TIMELINE_DIR=/workspace/src/results/dspp_run/timeline \
DSPP_TIMELINE_ITERATION=2 \
bash examples/run_dspp_b23_l40_saturated.sh
```

PyTorch profiler：

```bash
TRAIN_ITERS=3 \
DSPP_TORCH_PROFILER_DIR=/workspace/src/results/dspp_profile/traces \
DSPP_TORCH_PROFILER_ITERATION=2 \
bash examples/run_dspp_b23_l40_saturated.sh
```

## 12. 代码索引

- `megatron/core/datasets/dspp_batch_plan.py`：切分、packing、metadata 和 materialization；
- `megatron/core/datasets/dspp_training.py`：训练 batch adapter；
- `megatron/core/datasets/dspp_chunk_tuner.py`：候选 layout、调度存活量、显存模型和 chunk 选择；
- `megatron/core/datasets/dspp_ordering.py`：chain grouping、warmup/steady ordering 和 profile cost；
- `megatron/core/kv_cache/dspp_packed_attention.py`：packed attention、KV state、跨 segment backward；
- `megatron/core/pipeline_parallel/schedules.py`：单卡与 PP=3/VPP=2 stage-local schedule；
- `megatron/core/parallel_state.py`：方向化 P2P process groups；
- `megatron/core/zbpp_utils.py`：deferred W queue；
- `megatron/core/pipeline_parallel/dspp_metrics.py`：低开销统计；
- `megatron/core/pipeline_parallel/dspp_timeline.py`：task/P2P 标记、CUDA-event timeline 和 SVG；
- `megatron/training.py`：iteration metrics 与外层 PyTorch profiler；
- `examples/run_dspp_b23_l40_saturated.sh`：最终 L40 preset；
- `tools/build_dspp_synthetic_dataset.py`：可复现合成数据；
- `tests/unit_tests/dspp/`：DSPP 自动化测试。

## 13. 后续优先级

1. 用 PyTorch profiler 复核新 rendezvous 的 compute/P2P overlap，并减少 launch-ack 等待带来的性能代价；
2. 用更长的无 profiler run 报告 steady-state median/P95 和有效 token/s；
3. 再执行普通 pipeline、变长普通 schedule、DSPP 原顺序、packing、packing+ordering 的完整消融；
4. 补充均匀、双峰、长尾长度分布；
5. correctness 和通信稳定后，再独立设计利用 bubble 执行 critical-path attention 的调度；
6. 若继续减少 launch starvation，优先评估真正 padded pack 的 fused pad/unpad，以及按稳定 `(valid_tokens, history layout, item lengths)` signature 分桶的 CUDA Graph。
