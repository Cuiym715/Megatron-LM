# DSPP Milestone C1+C2：物理 microbatch ordering 与实机观测

状态：已实现并在 3 张 NVIDIA L40 上完成正确性、timeline 和 Nsight 验证

日期：2026-09-03

## 1. 完成范围

C1 完成了 optimizer iteration 级的 DSPP 物理 microbatch 排序：

- warmup 优先选择 workload 最接近的 short-only residual pack；
- short-only 不足时，按短到长加入完整 long-sequence chain；
- steady 阶段把剩余 long-sequence chain 按序列长度从长到短排列；
- 剩余 short-only physical microbatch 按 workload 从大到小排列；
- long-sequence segment 在入口 F 顺序中连续；warmup 边界可以落在 chain 内，不会为容纳
  整条 chain 扩大 warmup；
- 为适配当前 Slice-V“组内逆序 B”的实现，每条 long-sequence chain 自身构成一个
  schedule group，short-only packed task 构成长度为 1 的 group；dataloader logical batch
  只负责装载数据，不参与 Slice-V 分组；
- 支持 `input` 原始顺序作为轻量对照。

C2 完成了两层观测：

- 默认训练路径：有效 token/s、iteration median/P95、packing utilization、schedule
  padding ratio 和 rank-0 peak allocated/reserved memory；
- 显式 profiling iteration：每个 stage 上每个 F/B/W CUDA task 的开始/结束 event、方向化
  P2P NVTX range、每 rank JSON、合并 SVG、bubble/critical-span 摘要和可回灌的真机 cost
  JSON；
- 一个 Nsight SQLite 摘要工具，用于计算 GPU compute/P2P overlap、P2P stream 数和方向化
  NVTX 实例数。

按讨论，本次没有执行完整五组消融，也没有生成均匀、双峰、长尾三套分布；它们保留为
C3。Attention bubble filling 仍不在 C1/C2 范围内。

## 2. C1 设计

### 2.1 iteration 级入口顺序

执行器先读取本次 optimizer step 的全部 logical batch，再建立：

```text
entrance_order = [(logical_batch_id, physical_microbatch_id), ...]
```

所有 stage 使用相同入口次序，但仍只按自己的本地 Slice-V F/B/W 列表推进。排序只是 CPU
侧每 iteration 一次的 metadata 操作，不向 GPU 热路径加入 ready-set 扫描或逐 task
validator。

### 2.2 chain 连续性与 Slice-V 分组

现有 Slice-V 的 B 顺序是“调度组内逆序、调度组间正序”。如果只把全局排序结果依次覆盖
原 logical-batch slot，一条 long sequence 可能跨组，导致 B 顺序错误。

实现分开处理两个概念：

1. ordering 的 warmup/steady 只是入口序列的位置标签。warmup task 数严格等于目标值，边界
   可以落在一条 chain 内；
2. scheduler 把每条 chain 本身作为一个 schedule group，short-only packed task 作为
   split-count 为 1 的 group，以适配现有 B 的组内逆序实现。不能把多个无关 chain 再按
   dataloader logical-batch 数量合并，否则会把最大 split count 和 warmup 错误放大。

对于 `PP=3`、`chunk=256`、最大 sequence split count 为 3 的饱和复测，正确分组为
`[1, 1, 1, 2, 2, 3, 3]`，ordering warmup 为 7；三个 stage 在首个 B1 前的
`F0+F1` 数均为 10。

因此入口 F 顺序和 KV 梯度顺序满足：

```text
F(seq, 0) -> F(seq, 1) -> ...
B(seq, n) -> B(seq, n-1) -> ...
```

由入口排序与 schedule group 结构直接保证。它不要求连续 wall-clock 执行；F segment 之间
可以按 stage-local 固定调度插入 B/W。release 路径不执行额外 dependency 检查。

### 2.3 逐 physical microbatch loss readiness

全局重排后，某个 physical task 的 B 可能早于它原 logical batch 的其它 F。旧 B3 实现要
等待一个 logical batch 的全部输出拼接后才构造 loss gradient，因此不能支持该顺序。

新实现对 loss-stage 的每个 physical output 立即计算：

```text
loss_i = sum(token_loss_i * loss_mask_i) / iteration_valid_tokens
```

并立即用 `autograd.grad` 得到该 task 的 output gradient。iteration 的日志 loss 是所有
`loss_i` 之和。这样 packing/ordering 不改变有效 token 分母或 optimizer 总梯度，也不再
要求同一个 logical batch 的全部 F 先完成。

### 2.4 排序 cost

排序从不把 FLOPs 当成训练耗时。cost 选择顺序为：

1. 若指定 `--dspp-order-profile`，按 physical microbatch 的 Q/K shape signature 读取前次
   CUDA Event profile 的 max-stage F+B+W 时间；
2. profile 未覆盖的 shape 才使用 `estimated_flops` metadata proxy；
3. proxy 只参与 CPU 排序，不 sleep、不控制 task 执行时长。

每次 timeline 会生成 `iteration_XXXXXX_costs.json`，可直接供后一次运行使用。

## 3. C2 观测实现

### 3.1 默认低开销指标

每个 iteration 只增加 rank 0 上两个 CUDA Event；读取发生在已有的
`optimizer.get_loss_scale().item()` 同步之后，没有新增逐 iteration synchronize。日志和可选
metrics JSON 包含：

- `effective_tokens_per_second = valid LM targets / train-step GPU time`；
- iteration median/P95；
- `packing_utilization = valid tokens / physical fixed-shape capacity`；
- `schedule_padding_ratio = communication-only slots / all schedule slots`；
- rank 0 peak allocated/reserved memory。

逐 stage 显存峰值保存在 profiling JSON/summary 中。原 Megatron 的固定 `seq_length`
tokens/s 仍保留，避免改变既有日志，但不作为 DSPP 有效吞吐。

### 3.2 显式 timeline 模式

只有同时提供 timeline 目录和一个 one-based iteration 时才启用：

```bash
DSPP_TIMELINE_DIR=/workspace/src/results/run/timeline \
DSPP_TIMELINE_ITERATION=2 \
./examples/run_dspp_b23_l40.sh
```

选中的 iteration 会执行一次开始 barrier、每 task CUDA Event、结束 synchronize 和产物写入；
这些操作明确属于 profiling 开销，普通训练完全跳过。合并 SVG 每个 stage 一行，颜色区分
F0/F1/B0/B1/W0/W1，鼠标提示包含原始 logical/physical TaskId 和时间。

可读性增强同时生成 `iteration_XXXXXX_microbatches.md`，分别列出排序前 physical
microbatch 的 sequence/segment/offset/valid-padding 组成，以及最终 entrance order、
warmup/steady phase、schedule group、排序 workload 和实际 stage/vchunk 模型布局。SVG 将
每个物理 stage 的 F/B/W 合并在同一条 compute lane，矩形标注入口 microbatch id `mN`；不足
3 px 的 W 用可见宽度 marker 表示，精确时长仍以 tooltip/JSON 为准。summary 另外按 stage
和 F0/F1/B0/B1/W0/W1 给出 median/P95。

CUDA Event 包围的是 task 在默认 compute stream 上的 GPU 墙钟包络，P2P submit/wait 在
range 外。它不是该 task 内所有 compute kernel duration 的纯求和：若 NCCL 与计算并发并
争用 GPU 资源，计算包络本身会被拉长。需要纯 kernel attribution 时，以 Nsight/CUPTI 将
kernel 投影到对应 NVTX task range 为准。

### 3.3 B/W 的实际边界与计时解释

橙色 `B0`、红色 `B1` 都是纯 B，紫色 `W0`、黄色 `W1` 才是 W。当前 B/W 已解耦：B 在
`WeightGradStore.set_split_bw(True)` 下执行 autograd activation-gradient graph，并把
tensor-parallel linear 捕获到的 weight-gradient GEMM 入队；W 只弹出这些延迟 GEMM。B 仍含
FlashAttention backward、linear dgrad、RMSNorm/activation backward、KV gradient routing
等大量工作，W 并不是“整个 backward 中除 B 外的一切”。

小模型尤其是 hidden=64、chunk=32 时，W 的几个 GEMM 很小，而 B 会通过 autograd 发射大量
小 kernel。CUDA Event 给出 begin/end 之间的 compute-stream 墙钟包络，它包含 kernel 之间的
CPU/autograd 发射空隙以及并发 NCCL 引起的排队/争用，所以不能直接拿该包络与模拟器 FLOPs
比例比较。

保存的 11-layer/hidden-512 Nsight report 提供了 kernel-sum 对照：stage1/2 每个 B 稳定发射
约 138--146 个非 NCCL kernel，kernel 纯执行时间 median 约 0.260 ms；每个 W 只有 8 个 kernel，
median 约 0.027 ms。同期 CUDA Event B 包络 median 为约 12.8 ms。尾部短 B 因而不等于少算
了一层；需要区分“调度所感受到的 task 墙钟包络”和“kernel duration 纯和”。

### 3.4 Nsight 摘要

`tools/analyze_dspp_nsys.py` 读取由 `.nsys-rep` 导出的 SQLite，输出：

- 每 GPU 的 `ncclDevKernel_SendRecv` 数量和 stream id；
- compute kernel 与 P2P kernel 的时间交集；
- 两条及以上 P2P stream 同时 active 的时长；
- 每个物理方向的 send/recv NVTX 实例数。
- 每个 GPU、每种 F/B/W 的非 NCCL kernel count 与 kernel-duration-sum median/P95。

它不修改训练执行，也不在普通运行中导入 SQLite。

## 4. 参数与复现

新增参数：

| 参数 | 含义 |
| --- | --- |
| `--dspp-microbatch-order` | `input` 或默认 `warmup-short-steady-long` |
| `--dspp-v-layer-layout` | 默认 `balanced`；`legacy-output-slot` 保留旧 `(num_layers+1)` 布局 |
| `--dspp-order-profile` | 上次 timeline 生成的真机 cost JSON |
| `--dspp-timeline-dir` | profiling 产物目录 |
| `--dspp-timeline-iteration` | one-based 单次 trace iteration，0 关闭 |
| `--dspp-metrics-path` | 可选低开销 metrics JSON 路径 |

L40 脚本默认 6 层，使 PP=3/VPP=2 的六个逻辑 stage 各含一个 Transformer layer；同步支持
环境变量 `DSPP_V_LAYER_LAYOUT`、`DSPP_ORDER`、`DSPP_ORDER_PROFILE`、
`DSPP_TIMELINE_DIR`、`DSPP_TIMELINE_ITERATION`、`DSPP_METRICS_PATH`，并允许覆盖模型层数、
hidden/FFN size 和 head 数。

典型的 profile 回灌流程：

```bash
# 1. input-order 采样
TRAIN_ITERS=2 DSPP_ORDER=input \
DSPP_TIMELINE_DIR=/workspace/src/results/input/timeline \
DSPP_TIMELINE_ITERATION=2 ./examples/run_dspp_b23_l40.sh

# 2. 使用真机 cost 排序
TRAIN_ITERS=2 DSPP_ORDER=warmup-short-steady-long \
DSPP_ORDER_PROFILE=/workspace/src/results/input/timeline/iteration_000002_costs.json \
./examples/run_dspp_b23_l40.sh
```

## 5. 测试结果

### 5.1 自动化测试

```text
DSPP tests:    32 passed, 0 failed
utility:       6 passed, 0 failed
py_compile:    passed
bash -n:       passed
git diff --check: passed
```

测试新增覆盖 simulator ordering 规则、long chain 不跨 schedule group、profile cost 读取、
timeline/bubble/cost 聚合、SVG 生成和有效吞吐/利用率统计。首次把两组测试合并运行时，仓库
既有 `test_global_memory_buffer` 因比较两个 `torch.empty` 的未初始化内容偶发失败；单独复跑
为 6/6 通过，与本次代码无关。

### 5.2 三卡训练与数值对照

5-layer/hidden-64/BF16、PP=3/VPP=2 的 ordered 短训练完成 3/3 iteration：

```text
iteration 1: loss 10.39755, grad norm 1.605
iteration 2: loss 10.16533, grad norm 1.413
iteration 3: loss 10.04412, grad norm 1.229
skipped: 0, NaN: 0, deadlock: 0
```

最后将 disabled-timeline 热路径改为直接调用后，又独立完成 2/2 iteration；第二步为
318.4 ms CUDA train-step、1231.3 effective token/s，loss/grad norm 仍为 10.16533/1.413，
无 skip、NaN 或 deadlock。

同一 seed、同一数据分别运行 input order 和 profile-driven order：

```text
                 input order       profile-driven order
iteration 1 loss 10.39755          10.39755
iteration 1 grad 1.605             1.605
iteration 2 loss 10.16533          10.16533
iteration 2 grad 1.413             1.413
```

ordered timeline 明确记录 `cost_source: profile`。这证明本测试中 ordering 没有改变 loss 或
总梯度；不是完整的全参数逐元素 distributed oracle，但与 A/B 的全参数单卡 oracle 共同覆盖
了两层语义。

### 5.3 stage skew、bubble 和 critical span

profile-driven ordered 的 iteration 2 中，三个 stage 的首个 compute 分别在：

```text
stage 0: 0.132 ms
stage 1: 3.072 ms
stage 2: 4.711 ms
```

末个 compute 分别在 321.724、321.619、328.968 ms。各 stage 没有被全局 action index
强制成同一时刻推进。自动摘要给出的 critical span 为 328.836 ms；stage 0/1/2 的 bubble
分别为 293.494/213.721/22.715 ms。各 rank CUDA Event 的本地时间通过同机 monotonic CPU
anchor 校正后再合并，避免把三个 GPU 的相对零点直接视为同一零点。

作为最小正确性对照，同一 iteration 的 input-order critical span 为 320.056 ms，短于
ordered 的 328.836 ms；对应 CUDA train-step 为 348.9 与 358.5 ms。该 run 同时受到小模型、单次 trace 和 profiler overhead 影响，不能
作为性能结论，但清楚表明当前规则在这个 workload 上没有收益。正式结论必须等待 C3。

### 5.4 Nsight P2P/compute 结果

保留了一份 11-layer、hidden-512、3×L40、单 iteration 的 Nsight Systems 报告。摘要为：

```text
GPU 0: 66  P2P kernels, streams [18,22]
GPU 1: 132 P2P kernels, streams [18,22,26,30]
GPU 2: 66  P2P kernels, streams [18,22]

每个 edge 的两个方向各有 32 send + 32 recv NVTX instances
```

中心 stage 的四条 P2P stream 对应两个 edge 的两个物理方向，证明方向化 communicator 已
落到不同 CUDA stream，所有方向都能实际提交和完成。不过，本次 profile 测到的
compute/P2P overlap 仅 GPU 2 上 0.0138 ms，其余为 0；多 P2P stream 同时 active 也只有
0.015--0.052 ms。结论是“解耦和异步提交机制存在”，但这个单机小通信 workload **没有证明
有实质 overlap**。不能仅凭代码结构宣称 overlap 已带来性能收益；C3 应在正式模型/序列规模
上重新 profile，并检查 recv wait、launch ordering 和 NCCL stream 资源占用。

### 5.5 均衡层布局与单泳道 timeline 复测

新增 `balanced` 布局后，以 6-layer/hidden-64/BF16、PP=3/VPP=2 在 3 张 L40 上完成 2/2
iteration：

```text
iteration 1: loss 10.39565, grad norm 1.591
iteration 2: loss 10.14858, grad norm 1.373
skipped: 0, NaN: 0, deadlock: 0
```

报告确认布局为 stage0 `[1]/[6]`、stage1 `[2]/[5]`、stage2 `[3]/[4]`，不再存在零层
chunk。各 stage/chunk 的 F median 为 2.275--2.588 ms。第 2 iteration 的有效吞吐为
1130.6 token/s，packing utilization 0.9423；该 run 开启逐 task timeline，只用于正确性与
可读性复测，不作为关闭 profiler 后的性能结论。另以
`DSPP_V_LAYER_LAYOUT=legacy-output-slot NUM_LAYERS=5` 完成 1 iteration 兼容性复测，loss
10.39755、grad norm 1.605、无 skip/NaN。

## 6. 产物

- `results/dspp_c1c2_input/`：input-order 两 iteration 日志、metrics、JSON/SVG 和 cost；
- `results/dspp_c1c2_profile_ordered/`：回灌真机 cost 的 ordered 对照；
- `results/dspp_c1c2_ordered/`：3-iteration ordered smoke test；
- `results/dspp_c1c2_final/`：关闭 timeline 后的最终 2-iteration smoke test；
- `results/dspp_c1c2_nsys_medium/dspp_medium_one_iteration.nsys-rep`：3.1 MB Nsight 原始报告；
- `results/dspp_c1c2_nsys_medium/overlap_summary.json`：930-byte overlap 摘要；
- `results/dspp_c1c2_nsys_medium/timeline/`：每 stage JSON、合并 SVG、bubble summary 和 cost。
- `results/dspp_c1c2_readable/timeline/`：增强后的 SVG、分 stage duration summary 和可读
  microbatch construction/order Markdown 示例。
- `results/dspp_c1c2_balanced_timeline/`：6 层均衡布局、单 compute lane、`mN` 标注后的最新
  3×L40 复测产物。
- `results/dspp_c1c2_nsys_medium/overlap_and_task_kernel_summary.json`：P2P overlap 与
  F/B/W 非 NCCL kernel-sum 的紧凑摘要。

为控制空间，已删除重复的小模型 Nsight 目录和 medium report 导出的 7.7 MB SQLite；SQLite
可以随时从保留的 `.nsys-rep` 重建。

## 7. 结论与剩余工作

C1+C2 的实现目标已经完成：真实变长 physical microbatch 能跨 logical batch 排序，long
chain 的正反向次序由 schedule group 保证，loss/grad 语义保持不变；同时已有低开销常规
指标和可关闭的 task-level timeline/Nsight 证据链。

当前没有性能收益结论，且 Nsight 暴露了 material compute/P2P overlap 尚未出现。下一步
C3 才执行完整消融和均匀/双峰/长尾分布；结果稳定后，再决定是否进入 attention bubble
filling，而不是现在把新的 attention 调度变量混入正确性阶段。

后续用于避免小模型 launch-bound 失真的 L40 饱和配置、合成数据来源、大消息 P2P 限制和
最新 timeline 见 `docs/dspp-saturated-l40-profile.md`。
