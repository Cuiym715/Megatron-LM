# DSPP 实现复查问题

**代码级审计**。
> 第一轮请以“阅读、对比、定位、提出验证方案”为主，不要直接改代码。
> 对每个问题都请区分：
>
> 1. **当前代码实际上怎么做**（给出文件、函数、关键数据结构/调用链）；
> 2. **这是 correctness 必需条件还是当前 scheduler 的实现约束**；
> 3. **性能成本在哪里**（CPU、GPU kernel、copy、同步、显存、通信）；
> 4. **InfiniPipe / SlimPipe / DistCA / Zero-Bubble 等开源实现怎么做**；
> 5. **是否有优化空间**，以及怎么用 profiler / microbenchmark 证明；
> 6. 如果当前实现已经是合理或更优的，请明确说明理由，但不要在没有同条件测量时宣称端到端性能更强。

参考：
- 当前实现报告：`src/Megatron-LM-kwai/docs/dspp-implementation-report.md`
- InfiniPipe: src/InfiniPipe-code
- SlimPipe: src/Megatron-Kwai
- DistCA: src/DistCA
- Zero-Bubble PP: src/zero-bubble-pipeline-parallelism

我看完目前所有 milestone report 和最终 implementation report 后，还有下面这些疑问。有些是我没有完全理解清楚，有些希望结合当前代码以及 InfiniPipe / SlimPipe / DistCA 等开源实现再检查一下。

---

## 1. Historical KV 的 forward 实现

`dspp-milestone-a.md` 里把 forward 描述为：

```text
(layer_id, sequence_id)
↓
找到历史 KV
↓
拼 current KV
↓
FlashAttention
↓
如果还有 successor，则保留当前 KV
```

这里我主要不清楚“找到历史 KV”具体是怎么实现的。

因为我们已经要求同一条 sequence 的 slices 按顺序执行，所以理论上当前 slice 的历史 KV 就是它前面的几个 slice。想检查：

- 当前这里是真的通过 `(layer_id, sequence_id)` 做 lookup/search，还是只是简单取得已经保存的历史 KV？
- InfiniPipe、SlimPipe 是怎么实现这部分的？
- 当前实现里真正的性能开销主要是在 lookup，还是 historical KV 的拼接 / copy？
- 这里还有没有比较明显的性能优化空间？

---

## 2. Historical KV 的 backward

同样想检查反向的 historical K/V gradient 处理。

主要和 InfiniPipe、SlimPipe 等实现对比一下：

- historical K/V gradient 是怎么保存、累积、传回前面 segment 的？
- 当前实现有没有多余的 concat、slice、copy 或中间 tensor？
- 有没有进一步降低 backward 开销的空间？

如果当前实现从结构上已经比这些开源实现更省操作，也可以直接说明具体好在哪里；如果只是推测性能更强，需要区分“代码结构更优”和“实测更快”。

---

## 3. CPU planning 是否在 critical path

我记得有些方案会提前准备下一批数据和 schedule，让 CPU planning 和 GPU training overlap，避免 planning 工作落在 critical path 上。

想检查当前 DSPP 中这些工作什么时候发生：

- BatchPlan 构造；
- residual packing；
- ordering；
- metadata / `cu_seqlens` 准备；
- 数据 materialization。

然后对比一下：

- InfiniPipe 是否有 next-iteration planning / data prefetch / solver overlap？
- DistCA 是否也有类似的提前规划，还是主要做 communication-computation overlap？

如果当前 planning 已经很小，也可以直接测一下它实际占 iteration time 的多少，再决定要不要优化。

---

## 4. `batch_isend_irecv` 和 HOL blocking

B2 的报告里目前的流程大致是：

```text
ready
↓
batch_isend_irecv
↓
async submit
```

当前每条 physical edge 已经有 `next` / `prev` 两套独立 communicator。

我想再确认：

- 一个 `batch_isend_irecv` 里面具体 batch 了哪些 send / recv？
- send 和 recv 放在一起会不会仍然产生 HOL blocking？
- 当前两个方向的 communicator 是否已经足够避免 V-shape 正反方向互相阻塞？
- 现在大于 1 MiB payload 出现 NCCL progress / spinning 的问题，是否可能和这里的 send/recv launch 顺序有关？

---

## 5. recv 到底什么时候 launch

之前讨论过，如果希望通信真正异步，需要提前 post recv。

最终报告里只看到“recv 在 consumer 使用前才 wait”，但我不确定：

- recv request 是提前 launch，只是晚一点 wait；
- 还是 consumer 快要使用时才真正 launch recv。

想把当前实现的：

```text
allocate recv buffer
post recv
wait recv
```

三个时间点确认清楚。

另外也想看一下 trade-off：

- 提前 recv 能不能更好隐藏通信；
- 会不会因为同时保留太多 recv buffer 增加显存；
- 当前 P2P 通信本身比较小时，提前 recv 到底有没有实际价值。

---

## 6. 多条 CUDA stream 对计算的影响

现在每个方向有独立通信 stream。

想确认一个基本问题：

- CUDA stream 是只有上面真的在执行通信 kernel 时才占 GPU / SM 资源；
- 还是创建以后就会持续占一些资源？

以及实际应该关注的是不是 NCCL kernel 和 compute kernel 同时运行时的资源竞争，而不是 stream 数量本身。

---

## 7. Short-iteration padding

B2/B3 为了满足 V-ZB 的最小 task 数量，会增加 schedule padding。

我想确认这个机制到底是什么：

- 是逻辑上增加一个空的 schedule slot；
- 还是实际生成一个全零 physical microbatch 并进行通信；
- 是否会真的运行 Transformer / attention。

最终做性能实验时，我倾向于直接保证 physical microbatch 数量足够，让 `schedule_padding_ratio = 0`，否则这种场景本身可能就不适合拿来评价我的算法性能。

也想确认：相比增加 padding slot，是否存在修改 warmup task 数量的合法做法；如果不能保证 V-ZB schedule 正确，那就不考虑。

---

## 8. Ordering cost model

现在 ordering 好像已经可以使用真机 profile cost。

想把这一部分具体搞清楚：

- 真机到底测了什么？
- 一个 physical microbatch 的 cost 是 F、B、W 分开测，还是合成一个值？
- 不同 stage 的时间怎么合成？
- 是 lookup table，还是已经做了拟合？
- 没有 profile 到的新 shape 怎么处理？
- 这个 cost 是否应该进一步做 stage-specific / F-B-W-specific 的建模？

主要先说明当前真实实现，不需要为了复杂而额外做模型。

---

## 9. `iteration_000002_microbatches.md` 里的长度

我看：

`src/results/dspp_saturated_chrome/timeline/iteration_000002_microbatches.md`

感觉里面的 physical microbatch 看起来并没有统一成 `chunk_size`。

想确认：

- 这里展示的是 physical microbatch 的有效 token 数 / 内部 composition；
- 还是模型真正收到的 tensor shape；
- padding 是否只是没有在 report 里显示出来。

按照当前设计，我理解模型真正收到的 physical tensor 应该始终是固定 `chunk_size`。

---

## 10. 为什么每个 stage 都构造相同的 `DsppBatchPlan`

implementation report 里写的是：

```text
广播 lengths
↓
各 stage 独立构造相同 DsppBatchPlan
```

但普通 pipeline 中只有第一个 stage 直接处理 token，后面的 stage 主要接收前一个 stage 的 activation。

所以我想确认：

- 后面的 stage 为什么必须构造完整 BatchPlan？
- 是因为每个 stage 自己的 attention 都需要 sequence / segment / continuation / KV metadata？
- 后面的 stage 是否也在重复 materialize `tokens / labels / loss mask` 等实际上不需要的数据？
- 是否可以只让后续 stage 构造必要的 lightweight metadata？

如果当前重复工作的成本很小，也不一定需要优化。

---

## 11. Long-sequence chain 的连续性和未来调度空间

这里我想把约束明确区分一下。

当前我希望严格要求：

```text
入口 F 顺序：
A0, A1, A2, B0, ...
```

同一条 long sequence 的 segments 在入口 F 顺序中必须相邻。

因为如果变成：

```text
A0, B0, A1, ...
```

就会同时让多个 chain 的 KV / activation 存活，可能提高显存。

但是实际 stage-local 时间线上，可以在 `F(A0)` 和 `F(A1)` 中间插入 B/W 或另一个 vchunk 的任务，例如：

```text
F0(A0)
B1(...)
W(...)
F0(A1)
```

所以需要检查文档里类似：

```text
F(A0)
F(B0)
B(C)
W(...)
F(A1)
```

这种例子是否会让人误以为当前允许同一个入口 F stream 中把另一个 sequence 插到 long chain 中间。

另外，未来理论上可以考虑放松这个限制：

```text
A0, B0, A1, B1, A2, ...
```

这样可能增加 workload/slack matching 的自由度，但代价是：

- 更高的 KV / activation 显存；
- 更复杂的 dependency；
- 当前 chain 不跨 schedule group、组内 reverse B 的简单保证可能不能继续使用。

这个可以先作为以后可能的优化空间记录下来，当前实现仍然保持 long-chain adjacency。

---

## 备注：最终报告里已经比较明确的几点

下面这些如果当前代码与最终报告一致，就不需要重复当作未知问题重新调查：

- short-iteration padding 是 schedule-only slot，不会进入 Transformer / attention / KV / loss；
- 每条 physical edge 已经有 `next` / `prev` 两个独立 communicator；
- ordering cost 当前不是拟合模型，而是优先使用真机 profile cost，未匹配时 fallback 到 metadata proxy；
- 当前 long chain 在入口 F 顺序中仍保持连续，实际时间线上允许插入 B/W 或其他 vchunk 的任务。

如果代码实际行为和最终报告不一致，再单独指出。
