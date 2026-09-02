# DSPP Milestone B2+B3：三卡双向通信与 stage-local V-ZB

状态：已实现并在 3 张 NVIDIA L40 上通过测试

日期：2026-09-02

## 1. 完成范围

本里程碑将原 B2、B3 合并交付，在 B1 的变长 BatchPlan、residual packing、
packed FlashAttention 和 KV backward 基础上，实现了可真实训练的 PP=3、VPP=2
DSPP 路径：

- 每条物理 PP edge 建立 `next`、`prev` 两套独立 NCCL communicator；
- activation/gradient P2P payload 固定为
  `[micro_seq_length, 1, hidden_size]`；
- iteration 开始时只广播长度向量，各 stage 独立构造相同 BatchPlan；
- 每个 stage 只执行自己的 Slice-V F/B/W 任务流，不执行全局 action plan；
- send 异步提交，recv 只在对应本地任务成为消费者时等待；
- 支持不同 logical microbatch 产生不同数量的 physical microbatch；
- loss 位于 V 路径终点 `rank 0 / vchunk 1`，反向沿 V 路径返回；
- B/W 分离接入 `WeightGradStore`，optimizer step 前清空所有 deferred W；
- tied input/output embedding 在同一 rank 的两个 vchunk 间正确初始化并合并梯度；
- 默认训练路径不输出逐任务 trace，也不执行逐任务 debug validator。

本实现没有把模拟器中的 FLOPs 当作真实执行时间。FLOPs 估计仅保留在 B1 的
residual packing 启发式中；运行时推进完全由真实 CUDA 计算、P2P request 和本地
任务依赖决定。

## 2. B2：双向独立 P2P

### 2.1 communicator 拓扑

PP=3 有两条物理 edge：`0--1`、`1--2`。每条 edge 分别创建两个 NCCL group：

```text
next: rank 0 -> rank 1，rank 1 -> rank 2
prev: rank 2 -> rank 1，rank 1 -> rank 0
```

group 按物理方向命名，而不是按 activation/gradient 命名。V 型流水线中，F 和 B
都会根据 vchunk 改变物理方向，方向绑定可以避免相反方向共享 communicator 所导致的
head-of-line blocking。初始化时每条 lane 做一次单元素 P2P warmup，避免 NCCL lazy
connection 初始化落到不对齐的 stage-local 热路径中。

### 2.2 固定 payload 与 metadata

所有真实 P2P 数据均使用固定 shape。变长信息不在 P2P 前做动态 shape handshake，
而是在每个 logical microbatch 读取时广播一个很小的 `lengths` tensor。各 stage 使用
同一确定性构造函数得到相同的：

- physical microbatch 数量；
- sequence/segment id；
- token offset 和有效长度；
- packed `cu_seqlens_q/k`。

发送端可能比接收端更早产生消息。执行器按照接收端的本地任务顺序维护每条 lane 的
ready-send queue，并用 `batch_isend_irecv` 异步提交已经 ready 的 tensor。发送 buffer
一直保留到 request 完成；recv 只在对应 F/B 消费前等待。

## 3. B3：stage-local V-ZB 执行器

V 路径为：

```text
F: rank0/chunk0 -> rank1/chunk0 -> rank2/chunk0
   -> rank2/chunk1 -> rank1/chunk1 -> rank0/chunk1 -> loss

B: rank0/chunk1 -> rank1/chunk1 -> rank2/chunk1
   -> rank2/chunk0 -> rank1/chunk0 -> rank0/chunk0
```

每个 rank 本地调用 `build_slice_v_schedule` 获取自己的 F/B/W 列表，然后直接顺序执行。
新路径不调用 `build_slice_v_execution_plan`，没有全局 action index、iteration 内全局
barrier 或为了对齐 stage 而添加的 `cuda.synchronize()`。

状态以 `(logical_microbatch, physical_microbatch)` TaskId 保存。每个 logical batch、
每个 vchunk 分别持有：

- pipeline input/output activation；
- output gradient；
- sequence-aware KV state；
- loss 和 token weight；
- deferred W 所属的 vchunk queue。

长序列 segment 的 F 顺序和 B 逆序由固定的 stage-local schedule 保证。根据项目的性能
取向，release 热路径没有再对每个任务执行通用 DAG ready-set 扫描或重复 dependency
检查；只在 iteration 结束保留低频的 unsent-message、KV-state 和 W-queue 清空断言。

### 3.1 短 iteration 的 warmup padding

原 Slice-V 骨架要求 stage 0 至少具有 `max(split_counts) + 2*PP - 2` 个任务。变长
batch 可能偶尔少一个或多个任务。DSPP 会给较短 logical microbatch 增加最少数量的
schedule-only padding slot。这些 slot 只转发固定 shape 的零 tensor：

- 不进入 Transformer 或 attention；
- 不创建 KV state；
- 不进入 loss；
- B 不产生参数梯度，W 为 no-op。

因此它只补足 V-ZB warmup 的通信拓扑，不改变有效 token、loss 分母或模型梯度，也避免
为了补齐 schedule 执行伪 token attention。

### 3.2 V 型 tied embedding

普通 Megatron 假设输入 embedding 和 tied output head 位于不同物理 rank，通过 embedding
group all-reduce 同步。V 型拓扑把两端放在 rank 0 的两个 vchunk，原逻辑会使 output
embedding 保持全零。现在：

1. model 构造结束时从 chunk 0 本地复制初始化权重到 chunk 1；
2. backward 后本地求和两个 embedding gradient，并写回两份 gradient；
3. 两份参数执行相同 optimizer update，持续保持一致。

该修正使梯度能够从 output head 向所有 pipeline stage 正常传播。

## 4. 主要代码改动

- `megatron/core/parallel_state.py`
  - 每条 edge 的 next/prev group、getter、warmup 和 teardown；
  - DSPP V 型首尾 stage 与 embedding group 语义。
- `megatron/core/pipeline_parallel/schedules.py`
  - PP=3/VPP=2 DSPP selector；
  - stage-local V-ZB 执行器、lane-local send ordering、固定 payload；
  - TaskId activation/KV/loss 状态、B/W 分离和 schedule-only padding。
- `megatron/model/transformer.py`
  - DSPP V 型 layer allocation；
  - 零 Transformer 层逻辑 stage 的 KV 参数兼容。
- `megatron/optimizer/optimizer.py`、`megatron/training.py`
  - rank 0 两个 vchunk 的 tied embedding 初始化与梯度合并；
  - DSPP loss stage 和日志输出位置。
- `pretrain_llama.py`
  - pipeline group 内广播 compact length metadata。
- `megatron/arguments.py`
  - PP=3/VPP=2 约束与不支持组合的 fail-fast 检查。
- `examples/run_dspp_b23_l40.sh`
  - 三卡 L40 可复现实机脚本，支持 `TRAIN_ITERS`、`LOG_INTERVAL` 等环境变量。

## 5. 测试结果

环境：

```text
container: nvidia_pytorch
GPU: 3 x NVIDIA L40, 46068 MiB each
Python: 3.12.3
PyTorch: 2.10.0a0+a36e1d39eb.nv26.01.42222806
FlashAttention: 2.7.4.post1
```

### 5.1 自动化测试

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && \
   /workspace/src/venvs/megatron/bin/python -m pytest -q \
   tests/unit_tests/dspp tests/unit_tests/test_basic.py'
```

结果：

```text
29 passed, 0 failed
```

其中新增测试覆盖 V schedule 的最小 communication-only padding 计算和单 logical
microbatch 的明确拒绝。既有 utility smoke tests 单独运行结果：

```text
6 passed, 0 failed
```

`py_compile` 和 `bash -n examples/run_dspp_b23_l40.sh` 均通过。测试警告来自仓库既有的
PyTorch/FlashAttention/SWIG deprecation warning。

### 5.2 三卡单步诊断

小模型配置：5 个 Transformer layer、hidden 64、FFN 128、4 heads、BF16、PP=3、
VPP=2、sequence cap 96、chunk size 32、logical microbatch 为 4 个变长文档、每次
optimizer step 累积 2 个 logical microbatch。

在移除临时诊断前观测到：

```text
lm loss: 10.39755
grad norm: 1.605
stage 0/1/2 max gradient: 0.0666 / 0.0380 / 0.0424
skipped iterations: 0
NaN iterations: 0
```

三个 stage 均有非零梯度，证明 loss 已沿完整 V 路径反向传播。正式代码已删除这些逐任务
和逐 stage 打印。

### 5.3 三卡 100 iteration 验收

命令：

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && \
   TRAIN_ITERS=100 LOG_INTERVAL=10 DSPP_DEBUG_BATCHES=0 \
   ./examples/run_dspp_b23_l40.sh'
```

结果：

```text
100/100 optimizer iterations completed
NCCL/Python deadlock: 0
skipped iterations: 0
NaN iterations: 0

iteration  10: lm loss 9.697800, grad norm 1.000, 568.7 ms/iter
iteration  50: lm loss 5.705286, grad norm 1.081, 477.5 ms/iter
iteration 100: lm loss 4.031033, grad norm 1.072, 449.7 ms/iter
```

中间各 10-iteration 窗口均保持有限 loss 和非零 gradient。Megatron 当前打印的
`tokens per sec` 使用固定 `seq_length` 计算，不是 DSPP 的有效 token throughput，
因此本里程碑不把它作为性能结论。性能对照与有效 token/s 统计属于下一里程碑。

## 6. 当前支持边界

本次完成的是三卡 MVP，而不是通用 Megatron 并行组合：

- 固定 PP=3、VPP=2、TP=CP=DP=1；
- 至少两个 logical microbatch/optimizer iteration；
- local Transformer、gradient accumulation fusion；
- dropout=0，无 activation recompute、MoE、offload、vocab PP、overlap grad reduce；
- 当前 B/W queue 仍按 vchunk FIFO，由固定 local schedule 保证归属；
- 没有实现 attention bubble filling；
- 尚未完成与普通 pipeline 的性能消融或 Nsight overlap 报告。

这些限制均在参数解析或执行器入口 fail fast，不会静默回退到其它 schedule。

## 7. 结论与下一步

B2+B3 的核心验收已完成：变长 residual-packed batch 能在 3 张 L40 上经由两个独立物理
方向 communicator，按 stage-local V-ZB 顺序完成 F/B/W 和 optimizer update，并连续运行
100 iterations。下一步进入 ordering、有效 token/s、普通 schedule 对照和 profiler overlap
验证；attention bubble filling 继续保留为后续独立设计项。
