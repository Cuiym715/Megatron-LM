# DSPP 核心组件消融实验

更新日期：2026-09-13

本文报告三组 DSPP 消融实验：Packing × Ordering、双向 P2P communicator 是否拆分，以及固定/最大显存可行/自动 chunk size 选择。实验目标是区分任务构造、调度入口顺序、通信 progress 和 chunk 策略各自带来的影响。

## 1. 实验设置

### 1.1 模型和硬件

```text
GPU: 3 × NVIDIA L40
PP = 3, VPP = 2, TP = CP = DP = 1
Transformer layers = 24
每个逻辑 stage = 4 layers
hidden size = 2048
FFN hidden size = 5504
attention heads = 16
BF16 + FlashAttention
micro batch size = 4
global batch size = 16
```

Packing、Ordering 和通信消融固定 `chunk_size=3584`。Chunk selector 消融分别使用 fixed 2048、fixed 4096、最大显存可行策略和当前 profiled auto 策略。

### 1.2 输入和统计口径

输入不是手工构造的特殊分布，而是根据 InfiniPipe Figure-2 长度分布采样得到的 seed 11、19 和 29 三组数据，并丢弃当前 L40 环境无法承受的 16K 以上极端项。每个配置对每个 seed 运行 4 个 iteration。

每次独立进程的第一个 iteration 包含 CUDA、allocator 和运行时 bootstrap 效应，聚合时将其排除。因此每个成功配置包含：

```text
3 个独立 seed
9 个稳态 iteration
340,080 个相同的有效 token
```

表中的有效 token/s 使用全部 9 个稳态 iteration 的总有效 token 除以总时间，而不是简单平均各轮吞吐。所有成功配置均为 0 skipped、0 NaN、无 OOM。

### 1.3 结果位置

原始日志、逐轮 metrics、退出状态和机器可读汇总位于：

```text
/home/cuiym/src/results/dspp_ablation_suite_20260913/
```

其中：

```text
run_manifest.tsv       每个运行的配置、seed 和退出状态
summary.json           排除 cold start 后的统一汇总
packing_ordering/      Packing × Ordering 原始结果
communication/         P2P communicator 消融结果
chunk/                 Chunk selector 原始结果
```

复现脚本和汇总工具为：

```text
examples/run_dspp_ablation_suite.sh
tools/summarize_dspp_ablation.py
```

## 2. Packing × Ordering 的 2×2 消融

### 2.1 对比配置

四个配置只改变 residual packing 和 pipeline 入口 ordering：

| 配置 | Residual packing | 入口顺序 |
| --- | --- | --- |
| `no_pack_input` | 关闭 | 原始 input order |
| `no_pack_ordered` | 关闭 | warmup-short + steady-long |
| `pack_input` | 开启 | 原始 input order |
| `pack_ordered` | 开启 | warmup-short + steady-long |

四组均使用相同的 stage-local V-ZB、大消息 matched lookahead 和双向独立 communicator。关闭 packing 时，每个短序列或 tail 单独占一个 padding 后的 physical microbatch。

### 2.2 结果

| 配置 | 平均 iteration | Median | P95 | 有效 token/s | 平均 Physical MB/iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_pack_input` | 2971.8 ms | 2817.9 ms | 4123.4 ms | 12,715.2 | 20.22 |
| `no_pack_ordered` | 3024.3 ms | 2826.5 ms | 4131.5 ms | 12,494.4 | 20.22 |
| `pack_input` | **2014.6 ms** | **1859.3 ms** | **3457.8 ms** | **18,756.3** | **12.44** |
| `pack_ordered` | 2060.1 ms | 1981.8 ms | 3501.5 ms | 18,341.8 | 12.44 |

固定 input order 时，开启 packing 带来：

```text
平均 iteration：-32.2%
有效 token/s：+47.5%
physical microbatch：-38.5%
```

固定 DSPP ordering 时，开启 packing 带来：

```text
平均 iteration：-31.9%
有效 token/s：+46.8%
physical microbatch：-38.5%
```

Packing 在 9/9 个逐轮配对中都更快。按逐轮比例平均，packing 的时间下降约为 34%，95% bootstrap 区间不跨过 0。因此，本实验能够稳定支持“packing 显著减少 physical task、launch 和 P2P 数量，并提高端到端吞吐”的结论。

当前 ordering 没有显示出稳定收益：

```text
有 packing：ordered 相对 input 平均时间 +2.3%
无 packing：ordered 相对 input 平均时间 +1.8%
```

逐轮观察中，两种情况下 ordering 都是 3/9 轮更快、6/9 轮更慢。配对 bootstrap 95% 区间均跨过 0；此外，同一完整配置的独立重复本身存在约 1% 的波动。因此合理结论不是“ordering 一定有害”，而是：

> 当前 `warmup-short-steady-long` 规则在这三组采样输入和固定 chunk=3584 下没有产生可辨认的稳定收益。现阶段 DSPP 的主要收益来自 packing，而当前 ordering 仍需要重新设计或在更强长度偏斜下验证。

这组消融还说明，ordering 没有改变 physical microbatch 数，性能差异来自 stage 间任务排列、bubble 和运行时波动，而不是任务总量变化。

## 3. 双向 P2P Communicator 消融

### 3.1 对比定义

为了只比较“同一 PP edge 的两个方向是否独立”，两个配置都保留每条物理 edge 的隔离：

- `split_directional`：每条 edge 的两个通信方向各使用一个 NCCL communicator；
- `shared_directional`：每条 edge 只使用一个 NCCL communicator，两个方向共享该 communicator 的 NCCL 执行顺序。

两组都使用同一个 stage-local V-ZB schedule、matched receive/credit 协议、`recv lookahead=2` 和 `send in-flight depth=2`。因此该实验不会把 edge 隔离、窗口深度或任务顺序混入比较。

### 3.2 结果

| 配置 | 成功运行 | 已完成稳态 iteration | 平均 iteration | 有效 token/s |
| --- | ---: | ---: | ---: | ---: |
| `split_directional` | 3/3 | 9 | 2032.7 ms | 18,589.0 |
| `shared_directional` | 0/3 | 0 | 不可定义 | 不可定义 |

三个 shared-directional 运行均在 60 秒限制内无法完成第一个训练 iteration，最终由外部 timeout 终止；三个 split-directional 运行均完成 4/4 iteration。日志中没有 Python exception、OOM 或 NaN，失败表现为通信 progress 停止。

因此，这组实验不能给出“拆分后快了多少”的百分比，因为共享模式没有形成可测量的稳态吞吐。它支持的是更强但范围更精确的结论：

> 在当前 stage-local 异序 F/B 通信模型中，同一 edge 的两个方向共享一个 NCCL communicator 无法提供可靠 progress。双向 communicator/stream 解耦不是普通的微小性能优化，而是让该调度可执行的必要条件。

这里不应进一步声称所有普通 pipeline 都必须拆分双向 communicator。经典全局对齐 schedule 的通信顺序更规则，可能在共享 communicator 上正常运行；本实验验证的是 DSPP stage-local schedule，而不是任意 PP 实现。

## 4. Chunk Size 策略消融

### 4.1 对比配置

| 配置 | 策略 |
| --- | --- |
| `fixed_2048` | 所有 iteration 固定 chunk=2048 |
| `fixed_4096` | 所有 iteration 固定 chunk=4096 |
| `max_feasible` | 根据当前显存模型选择可容纳的最大 chunk |
| `profiled_auto` | 显存过滤、profile menu、cheap screening 和 weighted DSPP DAG |

两个动态策略的第一个 iteration 都以 3584 bootstrap，并学习 activation memory rate；聚合时该 bootstrap iteration已随 cold start 一起排除。后续每轮使用相同 sequence lengths 独立决策。

与之前离线 fixed-run 查表评估不同，本次 `profiled_auto` 是真正运行自动选择训练路径，训练 iteration timer包含同步 planner 开销。

### 4.2 端到端结果

| 策略 | 平均 iteration | Median | P95 | 有效 token/s | 平均 Physical MB/iter | 峰值显存 rank 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_2048` | 2869.7 ms | 2809.0 ms | 4616.6 ms | 13,167.6 | 21.00 | 30.02 GiB |
| `fixed_4096` | **2072.7 ms** | **1877.0 ms** | **3563.9 ms** | **18,230.7** | 11.33 | 31.23 GiB |
| `max_feasible` | 2628.0 ms | 2239.9 ms | 4598.0 ms | 14,378.4 | **7.11** | 37.89 GiB |
| `profiled_auto` | 2409.9 ms | 2224.5 ms | 4147.3 ms | 15,679.9 | 11.33 | **30.25 GiB** |

相对 fixed 4096：

```text
fixed 2048：平均时间 +38.5%，吞吐 -27.8%
max feasible：平均时间 +26.8%，吞吐 -21.1%
profiled auto（包含同步规划）：平均时间 +16.3%，吞吐 -14.0%
```

### 4.3 最大显存可行策略

9 个稳态 iteration 中，最大显存可行策略选择：

```text
seed 11: 5632, 8704, 7168
seed 19: 6400, 8960, 6656
seed 29: 3584, 8960, 8960
```

这些较大 chunk 将平均 physical microbatch 数从 fixed 4096 的 11.33 降到 7.11，但 9/9 轮都没有快于 fixed 4096，平均反而慢 26.8%。这提供了直接证据：

- 较少的 segment 和消息不等于更短的 iteration；
- 超过高效 kernel 平台的 shape、padding 和 pipeline 并行度损失可能抵消 task 数下降；
- 最大显存可行策略还把峰值显存提高到 37.89 GiB，却没有获得性能收益。

因此，“选择显存能承受的最大 chunk”只解决可行性边界，不是合理的性能目标。

### 4.4 当前 Profiled Auto 策略

9 个稳态 iteration 中，当前策略选择：

```text
seed 11: 3584, 3840, 3328
seed 19: 4096, 3840, 3840
seed 29: 3840, 3840, 3840
```

即 3840 选择 6 次，3584、3328 和 4096 各 1 次。每轮完整 measured bank 均先经过显存过滤和 efficiency screening，再由 weighted DAG 在保留的候选中排序。

本次同步 planner 的逐轮决策时间为 80.7–649.5 ms，稳态平均 324.2 ms。端到端表中已经包含这部分时间。为了区分“选择质量”和“当前执行方式”，可以进行一个诊断性扣除：

```text
profiled auto 原始平均 iteration：2409.9 ms
平均同步 planner：               324.2 ms
诊断性扣除后：                   2085.7 ms
fixed 4096：                     2072.7 ms
```

扣除 planner 后，profiled auto 只比 fixed 4096 慢约 0.6%，逐轮有 5/9 优于 fixed 4096；对应有效吞吐约 18,117 token/s，接近 fixed 4096 的 18,231 token/s。

该扣除不是正式端到端结果，因为它假设记录的决策时间完全处于 iteration critical path。但当前 planner 确实在 iteration 边界同步执行，所以它能够说明主要瓶颈：

> 当前自动策略的候选决策质量已经接近强 fixed baseline，但同步 CPU planner 开销尚未隐藏，因而当前端到端动态训练仍慢于 fixed 4096。

下一步若继续推进动态 chunk，不应再针对 L40 调整选择权重；更直接的工作是将下一轮 metadata planning 与当前轮 GPU 训练重叠，或降低 weighted-DAG 的 CPU 开销。

## 5. 三组实验的总体结论

1. **Packing 是当前最明确的性能来源。** 它将 physical microbatch 数减少 38.5%，并使平均 iteration 时间下降约 32%，9/9 个配对 iteration 均获益。
2. **当前 ordering 尚未形成稳定收益。** `warmup-short-steady-long` 在这三组采样输入上与 input order 接近，并有约 2% 的非显著回落。后续调度优化应优先重新设计 ordering 的目标，而不是假设当前规则已经有效。
3. **双向 communicator 解耦是 stage-local schedule 的 progress 条件。** 共享双向 communicator 的三个 seed 均无法完成第一个 iteration，不能定义稳态性能。
4. **最大显存可行 chunk 不是性能最优。** 它虽然进一步减少任务数，却使平均时间相对 fixed 4096 增加 26.8%，同时使用更多显存。
5. **当前 profiled selector 的主要未完成项是规划开销隐藏。** 包含同步决策时比 fixed 4096 慢 16.3%；诊断性扣除平均 324.2 ms planner 后，两者只差约 0.6%。

从下一阶段研究优先级看，本轮结果支持先优化 microbatch ordering 和 stage-local schedule，并并行考虑 planner overlap。Packing 与通信方向拆分已经显示出清晰价值；继续压缩 chunk ranking regret 的边际价值相对较低。

## 6. 结论边界

- 实验覆盖 3×L40、PP=3/VPP=2 和 16K 截断后的三组 Figure-2 采样输入；不能直接外推到 80GB GPU、更多 PP stages 或更长上下文。
- fixed 4096 是本次指定对照中最优，不代表当前环境的全局最佳 fixed chunk；此前更密集的七点实验中，3840 是更强的固定 baseline。
- ordering 的差异较小，目前只能说“没有观察到稳定收益”，不能说所有输入上都无效。
- shared communicator 的失败说明它不适合当前 stage-local 异序执行；不能据此否定经典全局对齐 pipeline 使用共享 communicator。
- profiled auto 的“扣除 planner”只用于定位开销，不是可直接报告为端到端训练性能的正式结果。

## 7. 五个新增随机 seed 的扩展复核

为检查前三个 seed 的结论是否偶然，又随机选择了五个未重复 seed：

```text
567660, 56701, 297129, 816353, 394449
```

每个 seed 使用同样的 Figure-2 采样器、256-token 对齐、16K 上限和超限样本丢弃规则。新数据保留 92–94 条序列，最长序列为 13,056–14,848 tokens。对应长度文件和采样元数据位于：

```text
/home/cuiym/src/InfiniPipe-code/datasets/dspp_multisample_seed{seed}_max16k_drop.txt
/home/cuiym/src/InfiniPipe-code/datasets/dspp_multisample_seed{seed}_max16k_drop.txt.json
```

新实验结果位于：

```text
/home/cuiym/src/results/dspp_chunk_new5_20260913/
```

每个策略增加 5 个 seed、15 个稳态 iteration。与原三 seed 合并后，每个策略共有 8 个 seed、24 个稳态 iteration，并处理完全相同的 842,368 个有效 token。

### 7.1 八个 seed 的合并结果

| 策略 | 平均 iteration | Median | P95 | 有效 token/s | 平均 Physical MB/iter | 峰值显存 rank 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_2048` | 2677.3 ms | 2519.8 ms | 4413.8 ms | 13,109.9 | 19.46 | 30.02 GiB |
| `fixed_4096` | **1906.8 ms** | **1816.6 ms** | **3289.6 ms** | **18,406.8** | 10.50 | 31.23 GiB |
| `max_feasible` | 2383.1 ms | 2138.3 ms | 4281.1 ms | 14,728.3 | 7.13 | 37.89 GiB |
| `profiled_auto` | 2224.4 ms | 2042.8 ms | 3808.8 ms | 15,778.8 | 10.42 | 30.85 GiB |

相对 fixed 4096 的配对结果为：

```text
fixed 2048：平均慢 42.1%，24/24 轮都更慢
max feasible：平均慢 22.0%，仅 1/24 轮更快
profiled auto：平均慢 16.9%，24/24 轮都更慢
```

profiled auto 在 24 个稳态 iteration 中选择：

```text
3840：15 次
4096：5 次
3328：2 次
3584：1 次
4352：1 次
```

表中的 `profiled_auto` 原始 iteration time 和有效 token/s **包含**同步 CPU planner；它不是删去 planner 后的结果。其记录的同步 planner 平均耗时为 289.8 ms。只有下面这个诊断性数字才扣除了 planner：profiled auto 平均为 1934.6 ms，对应约 18,142.7 token/s，与 fixed 4096 的 1906.8 ms 和 18,406.8 token/s 接近，但仍慢约 1.5%。这不是正式端到端收益，因为 planner 当前确实位于 iteration critical path；它只说明 selector 本身没有产生很大的 chunk 选择损失。

### 7.2 对结论的影响

新增五个随机 seed 后，原先的判断得到加强而不是改变：

1. fixed 4096 在当前 8 个 seed 上仍是最强的简单 baseline；
2. 最大显存可行策略持续选择 3584–8960 等较大值，但任务数下降没有转化为吞吐提升；
3. profiled auto 的动态选择值集中在 3328–4352 的高效平台，但其潜在收益不足以覆盖同步 planner；
4. 当前实验尚不能支持“per-iteration chunk selection 提高端到端性能”的论文结论。

因此，如果论文需要一个稳健、可复现的 chunk 配置，当前应报告：离线 profile 用于避开明显低效 shape，并选择一个全局 fixed chunk；per-iteration selector 作为可行但尚未端到端获益的扩展，而不是主性能贡献。
