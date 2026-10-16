# DSPP 编译、融合与 CUDA Graph 优化计划

更新日期：2026-09-16  
状态：设计计划，尚未执行；不得作为当前实现能力或已测性能写入 implementation report

## 1. 目标和预期收益

当前 DSPP 的主要矩阵乘和 attention 已分别由 cuBLAS 与 FlashAttention 执行，编译优化不会减少必要的训练 FLOPs，也不能直接消除 pipeline bubble 或通信等待。它主要解决的是大量短 kernel 带来的 CPU dispatch、autograd 遍历和 kernel launch 开销。已有 profiler 中，一个 4-layer B task 仍包含约 250 个 non-NCCL kernel，因此在小 chunk、physical microbatch 较多或 GPU 队列较浅时，CPU 可能无法持续向 GPU 供给工作。

现阶段不把编译包装为 DSPP 的核心算法贡献。合理目标是增强 runtime，使调度算法的效果不被 eager execution 开销掩盖，并使与采用融合 kernel 的系统比较更加公平。根据当前 kernel 组成，先采用以下工程预期，而不是把它当作已经测得的结论：

| 场景 | 合理的端到端收益预期 |
| --- | ---: |
| `chunk=3584/4096` 等较饱和配置 | 3%--8% |
| 小 chunk、physical task 较多、launch starvation 明显 | 10%--20% |
| 跨输入分布的综合现实目标 | 5%--12% |

收益不会由各项优化线性相加。若受控原型在固定 chunk 下的端到端收益低于 2%，则不继续投入复杂的多 chunk CUDA Graph；若达到 5% 以上，再扩展 graph cache 和 B/W 支持。

## 2. 当前实现是什么

当前正式 DSPP 训练不是整图编译，而是 eager Python 调度加局部编译/融合：

```text
Python stage-local scheduler
  -> eager Transformer forward/autograd
  -> 局部 torch.compile / TorchScript
  -> FlashAttention、GEMM、RMSNorm 和 fused CUDA kernels
```

已经生效的机制包括：

1. `megatron/model/utils.py` 使用 `torch.compile(dynamic=True)` 编译 SwiGLU，并在训练前分别预热 forward 和 backward；
2. bias/dropout/residual 等小子图使用 TorchScript；
3. packed attention 使用 custom autograd 和 FlashAttention varlen CUDA kernel；
4. RMSNorm、gradient accumulation 和 Linear wgrad accumulation 已使用融合实现；
5. DSPP packed-attention 热路径已经消除部分无意义的 `cat`、`slice`、`copy`、`fill` 和 GPU `.item()`。

当前没有启用的机制包括：

- 整个 Transformer 的 `torch.compile`；
- 正式训练的 CUDA Graph；
- full-iteration graph；
- DSPP fast/fused RoPE；
- fused tail pad/unpad 和通用 history-gradient scatter。

离线 chunk compute profiler 与正式训练要严格区分。`megatron/core/datasets/dspp_compute_profile.py` 当前默认允许对 F/W 任务使用 CUDA Graph replay 计时，而 B 因 autograd/RNG 生命周期使用 CUDA Event eager 计时；正式 DSPP 训练目前没有 CUDA Graph。因此 profile JSON 中的 `source=cuda-graph` 只表示离线测量方式，不能声称训练已经 graph 化。正式引入训练 graph 后，离线 profile 应与部署执行模式一致；过渡期间最好保留 eager 和 graphed 两套 profile，避免用 graph replay 的 F/W 时间预测 eager 训练。

## 3. 算法变化为什么带来新的编译机会

per-iteration chunk selector 并不意味着一个 iteration 内 shape 持续变化。当前结构是：

```text
iteration 开始选择一个 chunk_size
  -> 本 iteration 的 physical Q/K/V、hidden state 和 P2P payload shape 固定
  -> valid_tokens、item_count、history length 和任务顺序仍然变化
  -> 下一 iteration 才可能切换 chunk_size
```

因此需要把 Transformer 拆成两类区域：

1. **静态 physical-shape 区域**：RMSNorm、QKV projection、output projection、MLP、SwiGLU、residual、Linear dgrad 和 wgrad。它们主要由 chunk size、模型 shape、dtype 和 stage 决定；
2. **动态语义区域**：KV history 构造、packed FlashAttention、`cu_seqlens`、P2P、ready/credit、stage-local task selection 和 deferred-W 队列。它们随 microbatch 内容变化，暂时保留 eager 执行。

推荐的计算岛边界借鉴 DistCA，但不能直接复制其普通 backward：

```text
pre-attention island:
  RMSNorm -> QKV projection -> fused RoPE

dynamic boundary:
  KV assembly -> packed FlashAttention -> P2P / local scheduling

post-attention island:
  output projection -> residual -> RMSNorm
  -> gate/up projection -> SwiGLU -> down projection -> residual
```

跨层还可以把 `layer i post-attention + layer i+1 pre-attention` 合成一个较大的计算岛，减少 graph replay 次数。attention 和通信仍留在图外。

## 4. 密集 profile、候选 menu 与 Graph cache 必须解耦

离线 profile 点多并不会要求缓存同样多的 CUDA Graph。三个集合承担不同职责：

```text
离线 profile bank（可以很密，N 个点）
  -> 当前 iteration 做显存过滤和 cheap screening
  -> candidate menu（少量 K 个点）
  -> weighted DAG 选择一个 chunk c
  -> graph cache 只处理实际被选中过的 c
```

也就是说，即使未来离线测量 50 或 100 个 capacity，某次 iteration 仍只执行一个 chunk。Graph cache 不应在启动时为全部 N 个 profile 点生成图，而应采用：

1. **eager fallback**：任何合法 chunk 在没有 graph 时都能正常训练；
2. **lazy capture**：某个 chunk 被实际选中并达到复用阈值后才 capture；
3. **有界常驻**：只保留少量高频 graph set，初始建议上限为 2--4 个 chunk；
4. **按热度而非 profile 密度管理**：一次性出现的显存边界 chunk 始终走 eager，不值得支付 capture 和静态 buffer 成本；
5. **固定 chunk 特例**：固定 4096 等实验只需要一套 graph，是最先验证的场景。

这里所说的“动态 graph”不能理解为一个 CUDA Graph 原生支持任意 chunk shape。`torch.compile(dynamic=True)` 可以为局部 pointwise 子图生成 symbolic-shape kernel，但 CUDA Graph replay 本质上仍要求稳定的 kernel 序列、shape 和内存地址。PyTorch 的 cudagraph tree 或 reduce-overhead 模式内部也通常为遇到的 shape 建立多个 recording，而不是得到一个真正任意长度的 graph。对 DSPP 更可控的方案仍是“局部动态编译 + 每个热 chunk 一个静态 CUDA Graph + cache miss 时 eager”。

Graph cache 的初步 key 为：

```text
(chunk_size, physical_stage, virtual_chunk, F/B/W, dtype, model_signature)
```

由于 attention 被排除，`valid_tokens`、pack item 数和 history layout 不进入 dense-island graph key，否则 graph 数量会随输入组合爆炸。

## 5. 是否应将 profile 限制为 256 的倍数

当前实现已经把粒度参数化：

- `tools/profile_dspp_compute_shapes.py` 的 `--alignment` 默认是 256；
- `dspp_compute_profile.py` 先构造几何初始点，再围绕高效平台、局部峰值和未解析区间建议 refinement 点；
- tuner/selector 的默认 alignment 也是 256；
- tail 已使用向上取整和有效长度 metadata，sequence length 不需要被 chunk 整除。

建议保留“默认只从 alignment 网格选择运行 chunk”的规则，但不要把 256 写成跨 GPU、模型和 kernel 都最优的理论常数。256 的作用是控制离线测量、在线候选和 graph specialization 的数量；它不是正确性约束，也不是 Tensor Core 对 token 维度的硬性要求。

推荐采用两级采样：

1. **全范围粗采样**：默认以 256 为运行粒度，覆盖从最小合法值到目标硬件可行上限；当前几何点加自适应 refinement 机制可以继续使用；
2. **局部细化**：在高效平台边界、效率局部峰值或显存边界附近，用 64/128 的步长补测少量点，判断 256 网格是否遗漏明显峰值。

如果局部细化发现 256 网格相对更细网格的最优点损失始终很小，例如低于预先设定的 1%--2%，部署 selector 继续只输出 256 的倍数；如果发现稳定且显著的非 256 峰值，则针对该模型/硬件把 alignment 调为 128，而不是在所有环境硬编码更密的全范围扫描。

这一做法兼顾两件事：密集 profile 用来理解 GPU efficiency 曲线，运行候选粒度用来限制 planner、编译 specialization 和 Graph cache 的复杂度。两者不必使用完全相同的点集。

## 6. 分阶段实现计划

### 阶段 0：建立一致的 eager 基线

固定输入、固定 `chunk=4096`、固定 ordering，保存无 profiler 端到端指标和一轮 PyTorch profiler。分别统计 RoPE、SwiGLU、attention 外 pointwise、B task CPU launch interval、GPU idle gap 和每个 F/B 的 kernel 数。离线 compute profile 同时生成 eager 与 CUDA-Graph replay 版本，确认当前 cost model 是否因执行模式不同产生系统偏差。

验收条件：不修改调度，仅明确 launch-bound 部分占 iteration 的上限。若 4096 下 CPU launch gap 已低于总时间的约 2%，就不应期待 graph 带来大幅收益，同时增加一个较小 chunk 作为压力场景。

### 阶段 1：低风险融合

第一项是解除 DSPP 对 `--use-fast-rope` 的禁止，并适配已有 `fast_rotary_pos_emb`：

- position id 来自 DSPP metadata；
- Q heads 与 KV heads 可不同；
- 只处理 valid prefix 或显式保证 padding 语义；
- forward/backward 与 eager RoPE 做数值对比；
- 测量每层 kernel 数和端到端收益。

第二项是对比当前 dynamic SwiGLU 与新版 Megatron 的显式 compiled SwiGLU backward。只有 profiler 证明 kernel/launch 数继续下降时才移植；不为了形式上的 `torch.compile` 重写已经高效的 GEMM 或 FlashAttention。

### 阶段 2：重构计算岛，但暂不改变执行方式

把 Transformer layer 划分为 pre-attention、packed attention 和 post-attention 三个显式函数，先全部走 eager。重构后必须保持：

- loss、参数梯度和输入梯度一致；
- KV 生命周期不变；
- F/B/W task 数和顺序不变；
- timeline 任务边界不变；
- 不增加 materialization copy。

这一步的价值是把可 capture 区域与动态区域形成稳定接口。若没有清晰边界，直接套 graph 会把 Python KV 状态和 NCCL 意外捕获进去。

### 阶段 3：固定 chunk 的 CUDA Graph 可行性原型

先只支持固定 `chunk=4096`，不立即支持全部动态候选。预热所有 kernel 后，对 attention 外的稳定岛尝试 graph capture/replay，并保持 FlashAttention、P2P 和 scheduler 在图外。必须记录：

- capture 启动时间；
- graph 私有 allocated/reserved memory；
- replay 前后的 CPU launch time；
- GPU idle gap；
- iteration median/P95；
- 数值一致性。

这里有两个不能忽略的训练难点：

1. **多个未完成 forward 的激活生命周期**：CUDA Graph 常复用静态 input/output/saved-tensor buffer，而 pipeline warmup 会在 backward 前连续执行多个 F。若复用同一 graph buffer，后一次 F 可能覆盖前一次 B 仍需的保存值。因此不能简单地让所有 microbatch 共用一个训练 graph instance；需要按最大 live slot 提供可安全复用的 buffer slot，或移植 Megatron 的 graph runner/mempool 生命周期管理。该额外显存必须进入 chunk 的显存模型。
2. **B/W 分解中的 Python side effect**：当前 B 通过 `WeightGradStore` 将 wgrad 工作压入 FIFO，W 再弹出执行。CUDA Graph replay 只重放 GPU kernel，不会重新执行 capture 时发生的 Python `queue.append()`。普通整层 backward graph 还可能把 wgrad 提前放回 B，直接破坏调度语义。

因此阶段 3 先回答“稳定计算岛是否有足够收益”，不能直接宣称完整 F/B/W graph 已经可用。

### 阶段 4：使 B/W 与 graph 兼容

若阶段 3 收益达到继续条件，优先选择以下两条路径中改动较小的一条：

- **局部 capture**：只 capture 不包含 deferred-linear side effect 的纯计算子图，Linear backward 和 W queue 继续 eager；收益较小但正确性风险低；
- **显式静态 W slot**：plan 构造期已知每个 B 对应的 TaskId 和 W，因此在 graph 外预先建立 W slot。B graph 只把该 task 的 activation/grad-output 写入静态 slot，W task 从指定 slot replay wgrad graph 并累加到静态 `main_grad`，不依赖 replay 中的 Python queue mutation。

不建议一开始直接移植整套新版 Megatron/Transformer Engine delayed-wgrad graph，因为当前 fork、custom packed attention 和 stage-local schedule 差异较大。若选择静态 W slot，需新增以下验收：FIFO 顺序、每个 TaskId 恰好一个 W、iteration 末 slot 全空、与 eager 参数梯度逐层一致。

### 阶段 5：有界多 chunk Graph cache

固定 chunk 原型通过后才增加：

```text
chunk -> graph set / static-buffer pool / usage counter
```

初始策略为：cache miss 走 eager；同一 chunk 至少出现若干次才在安全 iteration 边界 capture；最多常驻 2--4 个 chunk；内存不足时不 capture，不能牺牲训练的 OOM 安全性。Graph 静态 buffer 和 live-slot 副本占用必须作为 runtime reserve 纳入 selector，而不能只统计模型 activation。

CUDA Graph capture 不能与正常 NCCL/compute 任意并发执行。capture 应在所有 rank 已经完成模型初始化、目标 shape warmup 且处于受控边界时进行，并避免把一次性 capture 停顿记入稳态 iteration；如果 chunk 只出现一次，则始终 eager 更合理。

## 7. 主要风险和停止条件

| 风险 | 后果 | 处理方式 |
| --- | --- | --- |
| profile 很密导致误以为要缓存同样多 graph | 启动慢、显存膨胀 | profile bank 与 graph cache 解耦，lazy + 有界常驻 |
| graph 静态 buffer 被多个 live microbatch 覆盖 | backward 错误 | live-slot aware buffer pool，并计入显存模型 |
| capture 吃掉过多显存 | selector 原本可行的 chunk 发生 OOM | capture reserve 纳入显存过滤，cache miss 可回退 eager |
| B capture 不执行 Python W enqueue | W 丢失或提前执行 | 局部 capture 或显式静态 W slot |
| 每轮选择不同 chunk | capture 成本无法摊销 | 使用复用阈值，一次性 chunk 不 capture |
| dynamic compile 重编译过多 | 首轮停顿和 cache 膨胀 | 对局部 pointwise 使用 symbolic shape；大岛按热 chunk 静态化 |
| graph 改变 compute/P2P overlap | timeline 与吞吐可能退化 | 同时检查端到端时间和 Chrome trace，不只看 kernel 数 |
| 编译后小 chunk 效率曲线变化 | 原 chunk profile 失效 | 以最终部署执行模式重新生成 profile bank |

停止条件如下：

1. 固定 4096 的计算岛原型端到端收益低于 2%，且小 chunk 压力场景也没有明显改善：停止 CUDA Graph 扩展，只保留低风险融合；
2. 收益为 2%--5%：保留 fused RoPE 和固定/少量 chunk graph，不实现复杂 cache；
3. 收益超过 5%，且额外显存不会改变 chunk 可行性：继续实现 B/W 兼容和有界多 chunk cache；
4. 任何方案若要求改变 stage-local 调度语义、合并 B/W 或降低通信可靠性，均不接受，即使隔离 microbenchmark 更快。

## 8. 在整体工作中的优先级

编译优化属于 P2，而不是 P0/P1：

```text
P0  正确性、任意合法输入的通信 progress、OOM 安全
P1  packing、chunk selection、stage-local schedule、attention bubble filling
P1.5  将 per-iteration planner 移出 GPU critical path
P2  fused RoPE、局部 compile、CUDA Graph compute island
P3  通用 KV routing/pad-unpad/CE 自定义 kernel 和复杂 graph cache
```

packing 和 chunk size 的已有实验差异达到数十个百分点，而编译的现实目标是个位数到低双位数。编译工作仍有必要，因为它能消除已观察到的 eager launch 缺陷，并避免把 runtime 开销误认为 DSPP 调度本身的损失；但它不应抢在通信稳定、显存模型和核心调度设计之前。
