# DSPP 后续问题代码审计与修改计划

日期：2026-09-04  
实现状态更新：2026-09-05 已完成 P2 的 no-history multi-item KV fast path，并通过 L40 回归和隔离 profile
范围：逐项回答 `docs/dspp_followup_questions.md`；原始审计结论保留，已完成项在对应章节标注

## 1. 总结

当前 DSPP 的核心数学语义没有发现新的错误：fixed-chunk tensor、packed causal attention、
long-chain 正序 F/逆序 B、token-weighted loss 和 B/W 边界仍成立。但代码级复查发现三个应当
进入下一轮修改的实际问题：

1. **通信进度**：recv 不是提前 post，而是在 consumer task 到达时才 allocate、post，随后
   立即 `wait()`。配合 send 的 lane-local 顺序门控，这很可能是大消息 rendezvous 卡住的原因。
2. **KV 明确冗余（已修复）**：包含多个独立 short sequence、且完全没有 history 的 pack，
   forward 现直接复用 physical current K/V，backward 直接返回 FlashAttention 的 dK/dV；不再
   逐 item `torch.cat`、预分配或 slice copy。
3. **`input` 模式观测不一致（已修复）**：`DsppOrderingPlan.entrance_order` 现统一取
   `flatten(schedule_groups)`，timeline 的 `mN` 和 final entrance report 与实际 F0 入口顺序一致。
   默认 ordered 模式和实际 scheduler 均未改变。

另外，当前每个 stage 都在 compute 开始前同步构造完整 BatchPlan，并 materialize
tokens/labels/positions/loss mask；后两个非首尾 stage 实际只需要 metadata 和 position ids。
这确实可以精简，但现有 trace 尚未把 planning 各部分单独标记，因此应先测量再决定是否引入
next-iteration pipeline。

| 问题 | 核心答案 | 处理分类 |
| --- | --- | --- |
| 1. Historical KV forward | O(1) dict lookup；主要成本是多 part K/V concat | 先做明确 fast path；预分配方案先 microbenchmark |
| 2. Historical KV backward | 不重建 K/V，但保存整段 prefix；主要是 full-prefix dK/dV、scatter/add | 做无-history pack fast path；其余先测量 |
| 3. CPU planning | 当前整轮 planning/materialization 都在首个 F 前；仅 DataLoader CPU 部分预取 | 先增加 opt-in 分段计时，再决定改造 |
| 4. `batch_isend_irecv` | 每次只有一个 op，不是真正 batching；send 有 lane-local 门控 | 需要修改，和第 5 项合并解决 |
| 5. recv launch | consumer 到达时才 post，紧接着 wait | 高优先级修改 |
| 6. CUDA stream | idle stream 不占 SM；活跃/等待中的 NCCL kernel 才竞争资源 | 不需修改，仅 profile |
| 7. schedule padding | 真实传输零 tensor，但不运行模型；不能简单缩短现有 warmup | 核心不改；增加 benchmark fail-fast |
| 8. ordering cost | 可选 lookup table；max-stage 的 F+B+W scalar mean；无拟合 | 暂不复杂化，先改善采样可信度 |
| 9. microbatch 长度 | 报告显示有效/padding composition；模型 tensor 始终固定 chunk | 不需修改 |
| 10. 每 stage BatchPlan | metadata 必需，完整 token/label materialization 非必需 | 测量后做 stage-aware lightweight batch |
| 11. chain 连续性 | 默认模式实际 F0 入口连续；wall-clock 可插 B/W/F1 | 保持约束；input-mode metadata 已修复 |

## 2. Historical KV forward

### 2.1 当前实现

入口是：

```text
ParallelAttention.forward
  -> dspp_packed_flash_attention
  -> _DsppPackedFlashAttention.forward
  -> DsppSequenceKVState.build_key_value
  -> DsppSequenceKVState._get_history
```

关键文件：`megatron/core/kv_cache/dspp_packed_attention.py`。

`DsppSequenceKVState._entries` 是普通 Python dict：

```python
Dict[(layer_idx, sequence_id), _SequenceLayerEntry]
```

查找使用 `dict.get()`，不是扫描所有 sequence，也没有 search algorithm。找到 entry 后，history
是 `entry.segments` 中按 forward 顺序 append 的 `_CachedSegment` list。`_get_history()` 会：

- 检查 `next_segment_id`；
- 对这一条 chain 的 segment list 求 history token 总长；
- 返回一次浅层 `list(entry.segments)`。

因此 lookup 是均摊 O(1)，额外 Python 工作是 O(当前 segment id)，但只遍历当前 sequence 的
history，不遍历其它 sequence。

物理表示不是 contiguous prefix buffer，而是每个历史 segment 一个 tensor view：

```text
entry.segments = [segment0 K/V, segment1 K/V, ...]
```

执行 segment `i` 时，每层都会构造：

```text
[history segment 0, ..., history segment i-1, current segment]
```

只要 part 数大于 1，就分别执行一次 K `torch.cat` 和一次 V `torch.cat`。所以 long chain 的
后续 segment 会反复 materialize 整个 prefix。这才是“找到历史 KV”路径里的主要 GPU 成本；
dict lookup 本身不是当前优先优化对象。

截至 2026-09-05 已有 fast path：

- 无 history（单 item 或多个独立 item）：直接返回 current K/V，不执行 cat；
- full physical chunk：custom Function 外部不再产生 Slice/Select autograd chain；
- `cu_seqlens_q/k`：每个 state、每个 physical microbatch 只创建一次 CUDA tensor，供该
  vchunk 的多层复用。

多个独立 short sequence 组成的 pack 虽然逻辑上有多个 part，但其拼接结果正好就是原
`current_key/current_value` 的有效前缀，因此直接复用。带 history 的 continuation 路径仍按
`[history, current, short...]` 拼接，没有改变。

### 2.2 与 InfiniPipe 对比

InfiniPipe 的 `FlashSelfAttention` 每个 layer instance 自己持有 `kv_cache` 和 `kv_grad` dict，
dict 再按 `sequence_id` 索引。因此它在语义上也相当于 `(layer, sequence_id)`，只是 layer
由 Python module 实例隐式表达。

`src/InfiniPipe-code/infiniPipe/model/flash_attn_kvcache.py` 中：

- history 同样是 `k_cache/v_cache` tensor list；
- forward 每个 chunk 都使用 `torch.concat` 组成 prefix；
- 最后一个 chunk 使用 `history + [current]` 再 concat；
- 没有 DSPP 当前的 single-part direct-return fast path。

所以两者的 dict/list 查询复杂度相近。当前 DSPP 在 single-part 和外部 pad/unpad autograd
边界上结构更精简；但这只能说明少了一些显式操作，不能在没有同 shape benchmark 时宣称
端到端更快。

### 2.3 与 SlimPipe 对比

SlimPipe 当前本地分支为 `sc25slimpipe`。其 `KVCache` 属于一个原始 `MicroBatch`，只按
`layer_idx` 索引，因为这个 MicroBatch 内的 slices 属于同一个上下文。它不能直接表达 DSPP
这种同一 physical pack 内同时存在 continuation 和多个独立 short sequence 的 cache routing。

SlimPipe 提供两种比普通 `torch.cat` 更专门的表示：

- `extended`：通过可增长 storage 和 `ConcatInplace` 追加 current KV；
- `chunked`：保留 `ChunkedTensor`，可用 `fast_cat_cuda` materialize，或用多个 attention
  调用处理各 chunk 并合并 output/LSE。

这些实现说明“预分配/可增长 prefix storage”和“chunked attention”值得测试，但不能原样移植：
DSPP 还有 varlen residual pack、不同 item 的 causal boundary，以及手工 historical-gradient
routing。SlimPipe 的 chunked attention 还会把一次 prefix attention 改为多次 attention 加
LSE merge，是否更快取决于 segment 数和 shape。

### 2.4 结论和验证

当前 forward hot path 的预期优先级为：

1. FlashAttention 本身；
2. long continuation 的 prefix K/V concat 和显存带宽；
3. residual pad/unpad；
4. Python list/dict/metadata。

后续建议：

1. 用四类同 shape microbenchmark 分别测 `no-history single`、`no-history packed`、
   `2-segment prefix`、`3-segment prefix`；
2. 只有 prefix concat 占比明确后，再实现 per-sequence/per-layer contiguous storage 原型，与
   `torch.cat` 比较 CUDA time、peak memory 和 backward version/lifetime correctness。

不建议为了减少一次 dict lookup 改成 dense array。它会增加实现复杂度，但没有证据表明这是
可见热点。

## 3. Historical KV backward

### 3.1 当前实现

forward 的 custom Function 会保存 assembled Q/K/V、output、LSE 和 cu-seqlens。backward 先让
FlashAttention 为完整 assembled prefix 生成 dQ/dK/dV，再由
`DsppSequenceKVState.distribute_key_value_grads()` 路由。

对每个 history segment：

- 从 assembled dK/dV 取对应 slice；
- 存入 `pending_key_grads/pending_value_grads[segment_id]`；
- 如果已有更后 successor 的贡献，用 `previous + grad` 累加。

对当前 segment：

- 取 assembled dK/dV 中 current 区间；
- 如果它还有 successor，加上 pending gradient；
- pop retained KV segment；
- segment 0 完成后删除整条 `(layer, sequence)` entry。

无 history 且所有 item 都是 final segment 时，直接返回 assembled dK/dV。带 history 的
single-item 路径直接返回 current gradient view；带 history 的 multi-item pack 使用一个
fixed-shape 预分配 tensor，再逐 item slice 写回。这一选择是此前实测后保留的：把所有
multi-item 路径改成 `torch.cat` 会增加 materialization kernel。

### 3.2 与 InfiniPipe/SlimPipe 对比

InfiniPipe backward 会重新 `torch.concat` historical KV，再调用 FlashAttention backward；
然后把 dK/dV 按历史 chunk 切片，使用 `kv_grad` list 的 append、`+=` 和 pop 反向传递。
DSPP 的 pending dict 与它的 list 在数学上相同。

当前 DSPP **没有在 backward 重建 K/V**，因为 forward assembled K/V 被保存在 ctx 中。这在
显式 copy/kernel 数上少于 InfiniPipe，但代价是每个未 backward 的 physical task都保留自己的
assembled prefix。也就是说：

- DSPP：少一次 backward concat，可能更快，但 prefix saved-tensor 显存更大；
- InfiniPipe：forward 后主要保留分段 cache，backward 再 concat，copy 更多但可能节省在途
  assembled-prefix 显存。

不能只根据操作数判定哪一个端到端更优，需要同时测 task time 和 peak memory。

SlimPipe 让 concat/chunk wrapper 进入 autograd，并由 `MicroBatch.input_stack/output_stack` 把
KV gradient 作为跨 slice backward 的额外 input/output gradient 传递。它对单一等长上下文
很自然，但不能直接处理 DSPP residual pack 中多个 sequence 的独立梯度路由。

Zero-Bubble 的 B/W 分解只延迟 Linear wgrad，不处理 historical KV，因此这里没有可直接复用
的 KV-gradient 优化。

### 3.3 剩余冗余和优化空间

no-history multi-short pack 的明确冗余已删除：assembled gradient 的顺序等于 current physical
prefix，现直接返回 assembled gradient。判定还要求所有 item 都是 final segment；长序列首段
虽然没有 history，但 backward 仍必须等待 successor 的 historical-KV gradient，不会走该路径。

有 history 的路径中：

- FlashAttention API 本身要求产生完整 prefix dK/dV；这块大 tensor 不能靠 Python 数据结构
  优化消失；
- 每个 history segment 的 `previous + grad` 会产生 add kernel；
- continuation+short pack 必须把 current gradient 从 `[history,current,short...]` scatter 回
  `[current,short...]`，不能简单整体返回；
- 首次 pending gradient 当前可能是 assembled grad 的 slice view，会保留其底层大 storage，
  但所有 history slice 总量本来就接近该 prefix，是否 clone 更省内存并不显然。

候选优化是 per-sequence contiguous pending-gradient buffer，按 offset `add_`，或者一个 fused
K/V scatter-add kernel。但它们应在 profiler 证明 `slice/copy/add` 仍是主要剩余热点后再做。

验证必须同时报告：

- 每个 B 的 `aten::cat/copy_/add_/fill_` 次数；
- FlashAttention backward kernel time 与 routing kernel time；
- 2/3/更多 segment 的 B CUDA-event duration；
- peak allocated memory；
- 与 unsliced causal SDPA 的 Q/K/V 和参数梯度 oracle。

## 4. CPU planning 是否在 critical path

### 4.1 当前调用位置

当前 distributed DSPP 在 `pipelining_with_dspp_stage_local_v()` 开始时：

1. 连续调用 `get_batch_func()` 读完本 optimizer iteration 的所有 logical batch；
2. `pretrain_llama.get_batch()` 将 text/lengths 搬到 CUDA，并广播 lengths；
3. `build_dspp_training_batch()` 把 CUDA lengths 拷回 CPU，构造 BatchPlan；
4. 在当前 stage 的 CUDA device 上 materialize tokens、labels、position ids、loss mask；
5. 所有 logical batch 完成后构造 ordering、schedule groups 和 Slice-V schedule；
6. 然后才执行第一个 F。

所以 BatchPlan、packing、materialization 和 ordering 都位于 iteration 当前 critical path。当前
没有 next-iteration DSPP plan，也没有 background planning worker。

DataLoader 本身使用 `num_workers=2`、`pin_memory=True`、`prefetch_factor=2`，因此 mmap 读取和
CPU collate 可以与训练重叠。但 main process 中的 H2D、Megatron `broadcast_data`、lengths
broadcast、GPU-to-CPU lengths copy、BatchPlan 和 CUDA materialization 不在 DataLoader worker
预取范围内。

`cu_seqlens_q/k` 还有另一层延迟：它们不是在 planning 阶段统一准备，而是在每个 vchunk 的
physical F 第一次进入 attention 时，由 `prepared_cu_seqlens()` 创建并缓存。这避免了每层
重复创建，但仍把两个小 CUDA tensor 的创建放在 F 热路径；两个 vchunk 的 KV state 会各自
准备一次。

### 4.2 当前证据

轻量 timeline 的 anchor 在 planning 完成之后创建，因此它完全看不到上述开销。最终外层
PyTorch trace 中，rank 0 从 `ProfilerStep#1` 开始到第一个 F0 CPU range 约为 16.6 ms；整个
profiled step 约 2059 ms。这个 16.6 ms 是“batch 获取、广播、planning、schedule 构造和其它
step 前置工作”的上界，不是隔离后的 planning 数字；它也不能准确归属此前排到 CUDA stream
上的 materialization kernel。

因此目前最多只能说 CPU 前置区间看起来不像百毫秒级主瓶颈，不能据此证明 planning 可以忽略。

### 4.3 InfiniPipe

InfiniPipe 确实实现了 one-batch look-ahead：

- `collate_fn()` 第一次拿到 batch 时启动 `multiprocessing.Process(solve_target, ...)`；
- 下一次 collate 取出上一个 batch 的 solved schedule；
- 同时为当前新 batch 启动下一 solver process；
- 训练 batch N 时，rank 0 的独立进程求 batch N+1 的 plan；
- plan 经 pickle 和 `broadcast_object_list` 发送到其它 rank，训练数据随后在主进程 attach 和
  materialize。

所以它重叠的是主要 solver，而不是所有数据准备。它还付出 process startup、pickle、queue、
object broadcast 和 first-iteration priming 的成本。DSPP 当前只是 O(n log n) packing/order，
远轻于 InfiniPipe 的全局 solver，不能仅因为对方有 worker 就直接照搬。

### 4.4 DistCA

本地 DistCA 可见的 e2e path 在每个 loop 中先取得两份 ping/pong batch，随后同步执行
`batch_to_items_general()` 和两次 `planner.plan()`，再进入模型执行。没有找到 next-iteration
background planner/thread。

DistCA 的 ping-pong 主要发生在 Transformer 内部：用 compute stream 和显式 communication
stream 交错两半 workload，并将 layout transfer 与 all-to-all 分开。它证明的是
communication-computation overlap，不等于 CPU planner 与 GPU execution overlap。

### 4.5 建议

先增加只在显式 profile iteration 启用的范围：

```text
DSPP/prepare/dataloader
DSPP/prepare/broadcast
DSPP/prepare/plan
DSPP/prepare/materialize
DSPP/prepare/order
DSPP/prepare/cu_seqlens
```

分别用 CPU wall time 和必要的 CUDA Event 测 20 个 steady iterations。如果准备时间低于稳定
iteration 的 1%，保持同步实现更简单；如果达到 3%--5% 或造成 GPU 可见空闲，再做：

1. stage-aware lightweight materialization；
2. cu-seqlens 提前准备并在两个 vchunk 共享；
3. 最后才考虑一批 look-ahead background planning/double buffer。

## 5. `batch_isend_irecv` 与 HOL blocking

当前 `batch_isend_irecv` 名称容易造成误解：DSPP 每次调用都只传一个 `P2POp`。

send 路径：

```text
task tensor ready
  -> 放入 ready_sends[peer][message_key]
  -> 只有它等于 receiver schedule 的下一 expected key 才提交
  -> batch_isend_irecv([single isend])
  -> request 保存到 iteration 末尾再 wait
```

recv 路径：

```text
consumer node 到达
  -> torch.empty(recv buffer)
  -> batch_isend_irecv([single irecv])
  -> 立即 request.wait()
  -> 返回 tensor 给 consumer
```

所以目前没有把 send 和 recv 放在同一个 batch 中，也没有 multi-op coalescing。不存在“同一次
batch 中 send 等 recv”的 HOL；但存在另外两类串行化：

- 同一 peer/物理方向的 send 必须按 receiver schedule 顺序提交，晚 key 即使 ready 也不能越过
  尚未 ready 的早 key；
- consumer-time recv 会阻塞 Python stage-local executor，直到对应 send 完成。

每条 physical edge 的 next/prev group 已经阻断了两个物理方向共享同一个 NCCL communicator
造成的直接 HOL。它解决的是 cross-direction communicator contention，不会自动解决上述
lane-local ordering 和 late recv。

### 5.1 大消息问题判断

大于 1 MiB 时卡住很可能与当前 launch 顺序有关：

1. 小消息可能走 eager/buffered protocol，sender 在 receiver 尚未 post 时也能前进；
2. 大消息更可能依赖 rendezvous，sender 需要对应 recv 已发布；
3. 当前 receiver 只有执行到 consumer node 才 post，并立刻阻塞；
4. sender 可能因为 `expected_sends` 的更早 key 尚未 ready，而没有提交 receiver 正在等待的 key；
5. 更早 key 的产生又可能依赖已阻塞 stage 的后续本地 task，形成循环等待。

这与现有“1 MiB 可运行、2/8 MiB spinning”的阈值现象一致，但还不是最终证明。需要 payload
sweep 和 NCCL trace 确定具体等待环。

## 6. recv 什么时候 launch

答案明确：**当前不是提前 post recv，而是 consumer 到达时才 launch recv，然后立即 wait。**

`receive(sender, node)` 内 allocate、`irecv`、`request.wait()` 连续发生；调用点就在对应 F/B
node 执行前。最终报告中的“recv 只在对应本地任务成为消费者时等待”不够准确，应该改成：

> recv 在本地 consumer task 到达时才 allocate 和 post，并在同一函数内立即等待完成。

提前 post 的收益是：

- sender 可以更早匹配大消息 rendezvous；
- 通信可在 consumer 前与其它 compute overlap；
- 降低 late-recv 导致的循环等待风险。

成本是：

- 每个 outstanding recv 持有一个固定 activation/gradient buffer；
- request、buffer 和 message-key 生命周期更复杂；
- 无 tag 的同 communicator P2P 必须严格保持相同匹配顺序。

不建议一开始 prepost 整个 iteration。先实现每个 incoming physical direction 1--2 个 message 的
有界窗口：按该 stage 的 consumer schedule 顺序 allocate/post；consumer 只 wait 已存在的 handle，
完成后补发窗口。最终 case 每 buffer 为 1 MiB，即使中心 stage 四条 incoming/outgoing lane 做
少量窗口，显存增量也远小于当前 8--11 GiB 模型峰值。

对于小 payload，收益可能很小；但解决 2/8 MiB liveness 的价值高于小消息吞吐收益。

## 7. 多 CUDA stream 是否持续占 SM

CUDA stream 是命令队列/依赖域。仅创建一个 stream 不会持续预留固定比例的 SM，也不会因为
stream 数量多就自动降低 compute kernel 可用 SM。

真正需要关注的是：

- NCCL kernel 正在运行时会使用 SM、copy engine、L2、HBM 和互联资源；
- NCCL kernel 如果已经 launch 但在等待 peer/progress，可能以 active/spinning 状态停留，仍
  可能占用资源；
- 两个独立 communicator 允许两个方向同时 progress，也可能让多个 NCCL kernel 同时和
  compute 竞争；
- stream priority 只影响调度优先级，不是硬 SM 配额。

因此这里不需要因为“stream 数量”修改实现。应在修复消息匹配后，再用 trace 检查活跃 NCCL
kernel 与 compute kernel 的 overlap、SM active、duration 和 power state。CUDA Green Context、
SM margin 或 NCCL CTA tuning 属于资源竞争优化，不能修复未 post recv 或循环依赖。

DistCA 也显式创建高优先级 communication stream，并通过 event/tick 同步 compute；其代码同样
把问题当作活跃通信和计算的资源/依赖管理，而不是 idle stream 本身的成本。

## 8. Short-iteration padding

padding 是 schedule-only identity task，但不是“完全没有 GPU 工作”：

- `_pad_dspp_v_split_counts()` 增加某个 schedule group 的 split count；
- 超出真实 group 长度的 `(group, split)` 在 `slot_to_task` 中映射不到 TaskId；
- F 在 stage 0 建立固定 shape zero tensor，并沿 V 路径 receive/forward；
- B 在 loss 端建立 zero gradient，并沿反向 V 路径 receive/forward；
- 不调用 Transformer、attention、KV state、loss 或 parameter backward；
- 对应 W node直接 no-op。

所以它有 zero/memset、buffer 和 P2P 成本，但没有模型计算或参数梯度。最终保留实验的
`schedule_padding_ratio` 为 0。

当前 Slice-V 模板要求：

```text
total_segments >= max(split_counts) + 2 * PP - 2
```

阶段计数和跨 stage dependency 由这一模板共同决定。简单减小 warmup 数会使 phase count 为负，
或者破坏 F/B/P2P 依赖；它不是一个可以独立调小的性能 knob。只有重新设计并验证另一套 schedule
才可能减少该要求。

正式 benchmark 应增加 opt-in fail-fast，例如 `--dspp-require-zero-schedule-padding`。它只在
CPU 构造期比较两个整数，性能成本可忽略；普通训练继续允许 schedule padding。

## 9. Ordering cost model

当前 profile cost 是可选 lookup table，不是拟合模型，也不会自动读取“最近一次”结果。只有
显式指定 `--dspp-order-profile PATH` 才会加载。

signature 为：

```text
c{chunk_size}|{item.token_length}:{item.token_offset + item.token_length};...
```

它表达当前 physical pack 的 query length 和每个 item 的 prefix end，但不包含 hidden/FFN、
layers、dtype、heads、GPU 或 PP/VPP 配置。因此它应被称为“同配置下的 token-layout signature”，
不能把一个 cost 文件跨模型配置复用。

一次 profile 的聚合为：

1. 对每个 `(logical_id, physical_id, physical_stage)`，将该 task 的
   F0/F1/B0/B1/W0/W1 CUDA-event duration 相加；
2. 对同一 physical task 取三个 stage 中的最大和；
3. 对同一 signature 的多个 physical task 做算术平均；
4. 保存 `signature -> scalar cost`。

task duration 是 compute-stream 墙钟包络，不包含 P2P API range，但可能被并发 NCCL 的资源竞争
拉长。当前没有 median/outlier removal、跨 iteration aggregation、regression 或 interpolation。
unseen signature 唯一 fallback 是该 task 所有 item 的 `estimated_flops` 之和。

是否需要 stage-specific/F-B-W-specific：

- 对当前简单 ordering，scalar lookup 已足够表达“粗略接近”，无需立即增加复杂模型；
- warmup 实际主要关心 forward 注入是否接近，F+B+W scalar 未必是最匹配的目标；
- 若未来用 cost 预测 bubble/critical path，必须至少保留 `stage x {F0,F1,B0,B1,W0,W1}` 向量；
- 在这之前，先用多个非 profiler timeline iteration 对 signature 取 median/P95，并给 cost 文件
  写入模型配置 fingerprint，防止误用。

最终保留的无 profiler run 实际使用 `metadata-proxy`，不是 profile-driven ordering。

## 10. `microbatches.md` 中的长度

问题中引用的 `dspp_saturated_chrome` 已在结果整理时删除；对应最终文件是：

```text
src/results/dspp_final_no_profiler/timeline/iteration_000002_microbatches.md
```

表中的：

```text
Valid/padding tokens = 192/64
```

表示固定 `chunk_size=256` tensor 中有 192 个有效 token、64 个 padding。Composition 只列有效
item，所以看起来长度没有统一；它不是 tensor shape 列表。

真实边界始终为：

```text
tokens/labels/position_ids/loss_mask: [1, chunk_size]
pipeline activation/gradient:         [chunk_size, 1, hidden_size]
attention Q/K/V physical input:       [chunk_size, 1, heads, head_dim]
```

attention 内部会把 padding 去掉后调用 varlen FlashAttention，返回 pipeline 前再恢复 fixed shape。
无需修改代码；现有报告的 `Valid/padding` 列已经表达了这一点。

## 11. 为什么每个 stage 构造相同 BatchPlan

后续 stage 确实需要完整的**逻辑 metadata**，因为每层 attention 都要知道：

- 当前 physical task 包含哪些 sequence/item；
- segment id、offset、prefix length；
- `cu_seqlens_q/k`、max q/k length；
- 哪些 K/V 要保留、查找和在 backward 中回传；
- schedule group、entrance order 和固定 P2P message identity。

如果只有 rank 0 知道这些信息，后续 stage 无法独立推进 sequence-aware attention。各 stage 根据
同一 lengths 确定性重建 CPU BatchPlan，避免广播复杂 Python object；这个选择本身合理。

但当前代码不仅重建 metadata，还在每个 stage 调用 `build_dspp_training_batch()`，完整 materialize：

- tokens；
- labels；
- position ids；
- loss mask。

在 V 型路径中：

- rank 0/chunk 0 需要 tokens；
- 所有 stage/chunk 的 RoPE 需要 position ids；
- rank 0/chunk 1 的 output/loss 需要 labels/loss mask；
- rank 1/2 不需要 tokens、labels 或 loss mask。

因此后续 stage 的完整 materialization 是实现简化，不是 correctness 必需条件。它的 tensor
字节数相对模型显存很小，但会产生许多 `full/zeros/arange/slice copy` CUDA launch，并且全部在
首个 F 前发生。

建议保留“各 stage 本地构造相同 CPU metadata”，改成 stage-aware materialization：

- rank 0 构造 token/label/loss 数据；
- rank 1/2 只构造 metadata、position ids 和可共享的 device cu-seqlens；
- `_DsppVTaskBatch` 不再假设所有 stage 都有完整 `DsppPackedBatch` 数据字段。

是否进一步只在 rank 0 构造 plan、广播序列化 metadata，要看 CPU plan 测量；当前 plan 很简单，
广播 Python object 可能比本地重建更慢。

## 12. Long-chain 连续性与未来空间

### 12.1 当前实际保证

默认 `warmup-short-steady-long` 模式满足目标：

1. `_build_long_chains()` 按 `(logical_batch_id, sequence_id)` 收集并按 `segment_id` 排序；
2. ordering 将 chain tuple 作为不可拆分 unit；
3. `schedule_groups` 直接生成 Slice-V `split_counts`；
4. Slice-V 的 F 队列按 group、segment 正序，B 队列按 group、segment 逆序；
5. `slot_to_task` 将这些位置映射到 physical task。

最精确的说法是：

> 在 rank0/chunk0 的 pipeline 入口 F0 子序列中，同一 long chain 连续；每个 vchunk 自己的 F
> 子序列也保持 chain segment 正序。完整 stage-local wall-clock 时间线上，两个相邻 F0 之间
> 可以出现 B、W、F1 或通信等待，但不能出现另一个 chain 的 F0。

“atomic schedule group”不是 GPU 原子执行，也不是 chain-wide fence。

### 12.2 `input` 模式不一致

`mode=input` 时，实际 scheduler 使用 `_input_atomic_units()` 产生的 chain groups，因此执行
满足 chain adjacency 和 reverse B correctness。修复前 `DsppOrderingPlan.entrance_order` 被设置
为原始 BatchPlan order；修复后它等于 `flatten(schedule_groups)`。

BatchPlan 会先放所有 full chunks，再放 residual bins，所以一条带 tail 的 sequence 在原始 order
中可能被其它 sequence 分隔。修复前这会造成：

- 实际 F0 顺序由 atomic `schedule_groups` 决定，仍然正确；
- timeline `mN`、microbatch report 和 `entrance_order` 字段可能描述另一顺序；
- `input` 实际语义不是严格 raw input order，而是按首次出现稳定地把 chain 聚合起来。

当前 canonical order 是 stable-input-chain order：group 仍按其在原始 input 中第一次出现的
顺序排列，但同一 chain 的 segments 聚合在一起。带 long tail 的 input-mode 测试固定检查
`entrance_order == flatten(schedule_groups)`。此次修改只修复观测元数据，没有改变 scheduler。

### 12.3 是否放松 adjacency

未来允许 `A0,B0,A1,B1,A2...` 会增加 cost/slack matching 自由度，但也会：

- 延长多条 sequence KV state 同时存活的区间；
- 可能增加 activation 和 assembled-prefix saved tensor 峰值；
- 需要显式 per-chain forward/backward dependency；
- 不能再只依赖“一个 chain 等于一个 Slice-V group、组内 B 逆序”；
- 要让 scheduler 支持跨 group dependency 和 memory-aware ready selection。

当前不应放松。需要先修复大消息通信、建立稳定 benchmark 和显存模型，再把它作为独立算法
变体做实验。还要注意：现有 adjacency 只能缩短多 chain 重叠窗口，不能保证任意时刻只有一条
chain 存活，因为 pipeline 中本来就会有多个任务在途。

## 13. 按是否需要代码修改分类

### 13.1 需要修改

1. **有界提前 post recv**：将 receive 拆成 `post_receive()` 和 consumer-time `wait_receive()`，
   每个 incoming direction 保持 1--2 个有界窗口；同时审计 `expected_sends` 是否形成循环等待。
2. **大消息 liveness 验证**：增加 1/2/4/8 MiB payload sweep 脚本和超时诊断，不把 1 MiB
   preset 当成通信完成条件。
3. **no-history multi-item KV fast path（已完成）**：forward 直接复用 current K/V，backward
   直接复用 assembled current gradient，避免 cat、allocation 和 slice copy。
4. **修复 input-mode canonical order（已完成）**：`entrance_order == flatten(schedule_groups)`；
   timeline id/report 自动使用 canonical order，并已增加 long-tail 测试。
5. **benchmark padding guard**：增加默认关闭的 `require-zero-schedule-padding` 参数或饱和脚本
   guard，正式性能 run 打开。

### 13.2 测量后再决定是否修改

1. **Planning overlap**：先分段测量；超过 3%--5% 再实现 look-ahead/double buffer。
2. **Stage-aware lightweight batch**：结构上可删冗余，优先于完整 background solver；实际收益
   需测前置 CUDA launch 和 CPU time。
3. **contiguous KV/pending-grad storage**：可能减少重复 prefix copy，但会影响 autograd saved
   tensor、version counter 和显存 lifetime，必须先做独立原型。
4. **stage/F-B-W-specific cost**：只有当前 scalar ordering 被证明选错任务并造成 bubble 时才做。
5. **跨 chain interleaving**：属于新算法，不是当前实现修复。

### 13.3 不需要修改

- `(layer_idx, sequence_id)` dict lookup；目前不是值得优化的热点；
- 模型/pipeline fixed `chunk_size` tensor contract；
- short-iteration padding 的模型 no-op 语义；
- 当前 Slice-V warmup 数学模板；
- idle CUDA stream 的数量；
- long-chain adjacency 的默认约束；
- 将 Zero-Bubble wgrad queue 用于 historical KV；两者职责不同。

## 14. 最终修改计划

### P0：建立不可回退的基线

- 保留并通过当前 38 个 DSPP tests；
- input-mode long-tail order test、no-history packed gradient oracle 均已增加；
- 固定相同 synthetic data、seed、ordering 和 1 MiB baseline；
- 记录无 profiler 20 个 steady iterations 的 median/P95/token/s。

验收：loss/grad 不变，0 skip/NaN/deadlock，结果可与现有 final artifact 对照。

### P1：先解决 P2P 大消息进度

- 建立 per-direction receive descriptor/handle queue；
- 按 consumer schedule 有界提前 post 1--2 个 recv；
- consumer 只 wait 已 post request，完成后补充窗口；
- 保持每个 communicator 内严格 message order和 buffer lifetime；
- 检查 send 的 `expected_sends` 门控，避免等待环；
- 跑 1/2/4/8 MiB payload sweep，并保存一组 PyTorch/Nsight trace。

验收：至少 2 MiB 和 8 MiB case 连续多 iteration 完成；两个物理方向都 progress；无新增
global barrier；1 MiB case 性能不显著退化。

### P2：删除确定的 KV 冗余

- 状态：代码、L40 FlashAttention 回归和隔离 profiler 均已完成；
- no-history multi-short pack forward 不 cat；
- 对应 backward 不分配/scatter；
- profile `cat/copy/fill/add` kernel 数；
- 重跑完整 Q/K/V、GQA、packed isolation 和参数梯度测试。

验收：全量 `tests/unit_tests/dspp` 通过；目标 shape kernel 数下降；无 profiler median 不退化。

2026-09-05 验证结果：当前全量 `38 passed`。对 `(512, 16, 128)` BF16 K/V、4 个 independent
short item 的隔离 profile，旧 routing 每次包含 2 次 `aten::cat`、2 次 `aten::empty` 和 8 次
`aten::copy_`；新路径对应操作均为 0，并验证 K/V、dK/dV 对象直接复用。200 次调用的单次平均
墙钟约从 0.174 ms 降至 0.008 ms。该数字只表示每层 KV routing 隔离成本，不代表完整
FlashAttention 或 iteration 的端到端加速；完整无 profiler median 尚未重跑。

### P3：修复 observability 和 benchmark guard

- input-mode `entrance_order` 已修复；
- timeline/report 已使用实际 schedule flatten order；
- 增加 `require-zero-schedule-padding` benchmark guard；
- 更新 implementation report 中 recv launch 和 optional profile-cost 的准确描述。

验收：报告顺序与每个 stage 的 F0 子序列一致；正式 benchmark padding ratio 必须为 0。

### P4：测量并精简 planning

- 仅在选中 profile iteration 增加 prepare 子阶段 ranges；
- 测 CPU、CUDA preparation、D2H sync 和首个 F 前空闲；
- 若超过阈值，先实现 stage-aware materialization 和共享 cu-seqlens；
- 仍显著时再借鉴 InfiniPipe 做 one-batch look-ahead。

验收：报告各 preparation 子项 absolute ms 和 iteration 占比；优化后 GPU timeline 不新增空闲，
数值和 physical order 不变。

### P5：再评估 KV storage 与 cost model

- microbenchmark contiguous prefix KV、chunked attention 和当前 cat；
- 同时报 CUDA time、kernel 数、copy bytes、peak memory；
- 收集多个 timeline iteration 的 signature median/P95；
- 只有 scalar cost 造成可复现错误排序时，才扩为 stage/F-B-W vector。

### 暂不进入

- 跨 chain segment interleaving；
- attention bubble filling；
- Green Context/SM margin；
- CUDA Graph。

这些都应在 P1 通信 liveness 和 P0--P4 基线稳定后独立评估，避免把新的调度变量混入当前
correctness 与通信问题。
