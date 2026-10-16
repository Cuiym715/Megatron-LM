# DSPP 历史 dKV 拆分：当前实现与后续优化

更新日期：2026-10-16。本文描述当前代码，不再按开发轮次追加阶段报告。

**当前已实现一个可运行、可独立测量 B/H 的重计算版本。** 它在 DSPP 训练中先完成当前 physical microbatch 的完整 stage 输入反向 B，发布 dx，再立即执行历史 dKV 任务 H。FA、GQA/MQA、变长与 packing 已在限定配置下通过正确性验证；较大负载上已经测到完整 B 缩短，但尚未获得稳定的端到端加速，也没有实现跨其他计算任务延后 H 的调度。

本文只说明四件事：[当前实现](#1-当前是怎么实现的)、[代码改动](#2-改了哪些代码)、[困难与已做优化](#3-实现中遇到的困难与已做优化)、[下一步](#4-下一步准备做什么优化)。原始实验材料保留在编号归档001–013；重组前的完整过程记录保存在[014归档](assets/fa-split/014-implementation-doc-reorganization-20261016/implementation-plan-before.md)，不作为当前功能说明。项目外层研究设计与收益验证计划未在本次整理中修改。

## 1. 当前是怎么实现的

### 1.1 B、H 和 dx 的边界

DSPP 将变长序列切片并 packing 成 physical microbatches，同一序列的 chunks 存在 continuation dependency。当前 query 属于 chunk i，KV 包含它的历史 chunks 和当前 chunk。本功能把当前 query 对历史 KV 的梯度贡献 `dK(i→j)、dV(i→j)，j<i` 从 B 中移出。

| 任务 | 计算与用途 | 完成边界 |
| --- | --- | --- |
| **B** | 每层计算扫描全部合法 KV 的 dQ、当前区域 dKV；合并后续 chunks 已累加给当前 chunk 的梯度；继续 RoPE、投影、MLP、残差等输入反向 | 返回完整 stage dx；DSPP 原有独立 W 不包含在 B 中 |
| **H** | 计算本次 query 对历史 KV 的 dK/dV，按历史 segment 路由并累加到共享梯度缓存 | 供后续历史消费者使用；不再改变刚交付的当前 dx |
| **W** | DSPP 原有参数梯度任务 | 保持原 schedule 的任务定义 |

当前执行顺序为：

```text
B：每层 attention 的完整 dQ + 当前 dKV → 其余 stage 输入反向 → 完整 dx 就绪
  → 本地交付，或将 dx 加入发送队列并尝试异步发送
  → H：历史 dKV 计算与累加
  → 下一项原 schedule 计算任务
```

“完整 dQ”“投影输入梯度”和“完整 stage dx”不是同一就绪时刻。当前区域 dKV 是计算完整 dx 所必需的，不能随历史部分一起移出 B。B/H 是两个调度和计时任务，各自内部仍有多个 kernel，并非整个 backward 只有两个 kernel。

**通信在 H 前尝试提交，但不保证 H 前已经发出或完成。** `submit_send` 要满足消息顺序、对端接收 credit 和在途额度；条件不足时 dx 留在队列。数据发送由主调度线程再次检查，当前没有独立后台线程持续发起数据发送，因此可能到 H 之后才发出。已有图中的 dx 标记表示本地就绪，不是对端接收完成。

### 1.2 FA 拆分与 GQA 的计算方式

前向仍使用安装环境中的 FA varlen forward。反向使用基于 **FA2 v2.7.4.post1** 的独立 CUDA 扩展；源码、补丁与 CUTLASS 版本被锁定，不覆盖环境中的默认 FA。

当前 after-dx 路径的接口为：

```text
prepare_backward(Q, K, V, O, dO, LSE, 序列边界) → 保存状态，计算一次 FP32 delta
backward_dq_current(state)                     → 完整 dQ、紧凑当前 dK/dV（B 内）
backward_dkv(state, "history", compact=True)    → 紧凑历史 dK/dV（H 内）
```

这是**重计算方案**：保留 Q/K/V/O/dO/LSE/delta 等状态，不保存完整 P/dS 矩阵。H 根据同一份完整上下文 LSE 和 delta 重计算历史区域中间量；不能对历史子区域重新独立归一化。

`backward_dq_current` 已经合并为一次主 kernel 扫描：完全在历史区的 KV tile 只算 dQ，与当前区相交的 tile 同时算 dQ/dK/dV。非对齐边界 tile 可以产生被丢弃的历史临时值，但 B 只返回当前行；历史贡献由 H 计算并累加一次。dQ 仍使用全部合法历史和当前 KV。

GQA 不复制 K/V heads。令 `r=Hq/Hkv`，Q head q 使用 KV head `floor(q/r)`；各 Q head 的 dQ 独立保留，对共享 KV head 的 dK/dV 贡献求和。当前沿用每 Q-head 部分梯度工作区、FP32 归约及 BF16 输出的数值合同。packed 中各序列使用独立边界，当前由 Python 逐序列分派，**还不是原生批量 varlen launcher**。

### 1.3 历史梯度和状态生命周期

`DsppSequenceKVState` 管理 `begin_backward_task → mark_dx_ready → run_history_task`。B 中只路由当前 dKV，并消费后续 chunks 已提供的贡献；H 中只累加历史贡献，不重复消费或弹出当前 segment。梯度缓存保持 KV-head 布局，在对应历史 chunk 的 RoPE/投影反向前合并。

每个 H 保存当前 physical task 各层的 tensor 引用和冻结布局，不读取后来切换的 microbatch metadata。tensor 使用 detach 引用保留数据，不保留完整 autograd 图；版本和 stream 检查防止状态被修改或误用。B/H 在同一计算 stream 上依次提交，当前 schedule 在 H 后才进入下一计算任务，因此消费者按流依赖看到完整历史梯度；这不意味着每个 H 都做主机侧同步等待。

无历史时不发射 H kernel，也没有 H 计时事件，但 **B 目前仍走拆分入口**。跨其他 F/B/W 任务延后 H 所需的独立队列、消费者期限及更长生命周期尚未实现。

### 1.4 支持范围、开关与验证结论

已验证环境为3张 NVIDIA L40（SM89）、PyTorch 2.10.0a0（NVIDIA 26.01）、CUDA toolkit 13.1、FA 2.7.4.post1。当前约束如下：

- BF16、head dimension=128、完整 causal attention、dropout=0；支持经测试的 MHA/GQA/MQA、不同 Q/K 长度、短尾、非对齐边界、packing 和非 fast RoPE。
- DSPP 的 TP=CP=DP=1；支持单卡路径，主要训练验收为 PP3/VPP2、local transformer。不与另一个 CP-KV 缓存路径混淆。
- 尚不支持 activation checkpoint/激活重计算、MoE、fast RoPE、ALiBi、softcap、local window 或任意 dtype/head dimension。attention 内部重计算不等于已支持 activation checkpoint。
- 内核使用 FP32 原子累加 dQ，不承诺逐位确定性；同一份 backward 状态要求同 stream 使用。

| CLI 开关 | 实际行为 |
| --- | --- |
| 均不启用 | 原融合 FA backward，作为默认和正确性/性能基线 |
| 仅 `--dspp-split-attention-backward` | 旧的对照路径：在 attention backward 内依次完成独立 dQ、当前/历史 dKV，再返回 |
| 再启用 `--dspp-history-dkv-after-dx` | 当前研究路径：B 内合并 dQ/current，交付完整 dx 后运行独立 H |
| `--dspp-task-timing-only --timing-log-level 0` | 仅记录 B/H 边界事件，关闭内部标注与诊断；实际采集迭代还由 timeline 配置选择 |

正确性依据不是只有“训练能跑通”：

| 检查对象 | 已验证的内容与证据 |
| --- | --- |
| attention 输出梯度 | 与融合 FA、小形状 FP64 数学参考比较；覆盖 MHA/GQA/MQA、区域拆分、短尾、非对齐与 packed 隔离。合并版本验收212项测试通过，27项新扩展边界用例 memcheck 无错误，见[009](assets/fa-split/009-combined-current-20261016/validation.json) |
| 完整 dx 与历史消费者 | 1/3层 block、chunk63/64：H 前完整 dx 与参考一致，H 后当前 dx/参数梯度不变，后续历史消费者得到正确累加，见[测试代码](../tests/unit_tests/dspp/test_history_after_dx.py) |
| DSPP 真实训练梯度 | MHA/GQA/MQA 分别进行融合/拆分3轮对照，共比较351个参数梯度张量和495个stage dx；通过预设 `atol=.004, rtol=.01`、相对L2≤.02，见[009](assets/fa-split/009-combined-current-20261016/validation.json)。计数跨rank/iteration，不是独立参数数量 |
| 更大性能形状 | 32/16 heads、Q2048/K6144或K14336：当前与历史dKV对融合逐位一致，dQ最大相对L2约2.905e-5，见[012](assets/fa-split/012-width4096-theory-20261016/shape-correctness.log)。这些形状的独立梯度检查与训练计时分开；loss一致不替代全模型梯度对照 |

上述为归档运行结果，不表示每次改文档都重跑了整套测试，也不外推到未验证硬件和模型组合。

## 2. 改了哪些代码

### 2.1 生产实现与配置

下表只列历史 dKV 拆分相关改动，不把仓库原有 chunk construction、ordering 和 DSPP base schedule 算作本次新增。

| 文件 | 当前职责/改动 |
| --- | --- |
| [fa_split.py](../megatron/core/kv_cache/fa_split.py) | 新增状态与独立 dQ/dKV API、合并 dQ/current API；区域选择、逐序列 varlen 分派、版本与 stream 检查 |
| [backward.cu](../megatron/core/kv_cache/fa_split_src/backward.cu)、[bindings.cpp](../megatron/core/kv_cache/fa_split_src/bindings.cpp)、[FA补丁](../megatron/core/kv_cache/fa_split_src/flash_bwd_split.patch) | 新增 prepare、dQ-only、dKV-only、合并 dQ/current CUDA 入口；工作区、输出转换及 GQA 归约；真正裁掉不用的梯度矩阵乘 |
| [fa_split_build.py](../megatron/core/kv_cache/fa_split_build.py)、[upstream.json](../megatron/core/kv_cache/fa_split_src/upstream.json)、[THIRD_PARTY.md](../megatron/core/kv_cache/fa_split_src/THIRD_PARTY.md) | 独立扩展构建、固定 FA/CUTLASS revision、源码校验与许可证；仅显式 `--fetch` 下载依赖 |
| [dspp_packed_attention.py](../megatron/core/kv_cache/dspp_packed_attention.py) | 接入开关；保存前向布局；B/H状态生命周期；当前/历史紧凑梯度分别路由，保留后续chunk梯度累加 |
| [schedules.py](../megatron/core/pipeline_parallel/schedules.py) | 向KV state传递模式；完整B返回后先发布dx，再执行H；保留原F/B/W顺序及通信credit协议 |
| [arguments.py](../megatron/arguments.py)、[run_dspp_b23_l40.sh](../examples/run_dspp_b23_l40.sh) | 新增拆分、after-dx、纯任务计时开关与非法组合校验；launcher支持对应环境变量和GQA配置 |
| [dspp_timeline.py](../megatron/core/pipeline_parallel/dspp_timeline.py) | 增加H事件、纯B/H边界采样、事件预初始化及延后输出；纯B/H模式不把未采集区间算作bubble |
| [initialize.py](../megatron/initialize.py)、[profile_utils.py](../megatron/profile_utils.py)、[core/utils.py](../megatron/core/utils.py) | 纯任务计时模式关闭标注autograd节点与纯标注backward；关闭调试同步时不再插入无效包装节点 |

### 2.2 测试、测量与复现入口

| 入口 | 用途 |
| --- | --- |
| [build_dspp_fa_split.py](../tools/build_dspp_fa_split.py) | 构建固定版本扩展；依赖未缓存时才使用 `--fetch` |
| [test_fa_split.py](../tests/unit_tests/dspp/test_fa_split.py)、[test_fa_split_regions.py](../tests/unit_tests/dspp/test_fa_split_regions.py)、[test_fa_combined_current.py](../tests/unit_tests/dspp/test_fa_combined_current.py) | 基础拆分、区域/GQA与合并入口的数学和融合对照 |
| [test_packed_split.py](../tests/unit_tests/dspp/test_packed_split.py)、[test_history_after_dx.py](../tests/unit_tests/dspp/test_history_after_dx.py)、[test_clean_task_timing.py](../tests/unit_tests/dspp/test_clean_task_timing.py) | packed路由、完整dx/H合同、状态隔离、CLI限制和计时清理回归 |
| [validate_dspp_split_training.sh](../examples/validate_dspp_split_training.sh)、[validate_dspp_split_training.py](../tools/validate_dspp_split_training.py) | 真实PP训练参数梯度和dx对照；含快照/检查，不能用其日志报告性能 |
| [measure_dspp_combined_tasks.sh](../examples/measure_dspp_combined_tasks.sh)、[summarize_dspp_history_tasks.py](../tools/summarize_dspp_history_tasks.py) | 当前融合/after-dx成对B/H计时，按stage、VPP和相同layout比较；支持宽度/head/序列长度参数 |
| [audit_dspp_task_timing.py](../tools/audit_dspp_task_timing.py)、[diagnose_dspp_attention_cost.py](../tools/diagnose_dspp_attention_cost.py)、[diagnose_dspp_training_cost.py](../tools/diagnose_dspp_training_cost.py) | 独立计时审计和profiler诊断；不加载到正式性能运行。`dspp_cost_probe.*`仅为历史诊断代码，不是生产入口 |

在现有容器环境中，`/home/cuiym` 挂载为 `/workspace`，Python为 `/workspace/src/venvs/megatron/bin/python`。依赖已缓存时的基本入口：

```bash
# 正确性运行：输出和快照目录须尚不存在，不用于计时。
DSPP_SPLIT_CHECK_AFTER_DX=1 DSPP_SPLIT_CHECK_TASK_TIMING_ONLY=1 \
DSPP_SPLIT_CHECK_ROOT=/tmp/dspp-validation-new \
DSPP_SPLIT_CHECK_SNAPSHOT_ROOT=/workspace/.cache/dspp-validation-new \
  bash examples/validate_dspp_split_training.sh

# 当前hidden4096长负载的B/H配对测量：输出目录须尚不存在。
DSPP_COMBINED_HIDDEN_SIZE=4096 DSPP_COMBINED_FFN_HIDDEN_SIZE=8192 \
DSPP_COMBINED_Q_HEADS=32 DSPP_COMBINED_QUERY_GROUPS=16 \
DSPP_COMBINED_SEQ_LENGTH=16384 DSPP_COMBINED_CHUNK_LENGTH=2048 \
DSPP_COMBINED_DATA_PATH=/workspace/.cache/dspp-fa-split/workload-scaling-20261016/long-data \
DSPP_COMBINED_MEASURE_ROOT=/tmp/dspp-measure-new \
  bash examples/measure_dspp_combined_tasks.sh
```

精确环境、输入校验和、当时源码快照和运行命令见[012 manifest](assets/fa-split/012-width4096-theory-20261016/manifest.json)。旧版测量/绘图脚本保留供历史实验复现，当前入口以上表为准；不需要按001–013顺序重跑。

## 3. 实现中遇到的困难与已做优化

### 3.1 已解决什么，哪些原因还没有定量证明

| 困难 | 已采取的实现/优化 | 证据与剩余边界 |
| --- | --- | --- |
| 独立dQ并不等于完整stage dx | 把H从attention backward移到完整B返回之后；状态显式持有各层历史计算所需数据 | 多层及真实PP梯度对照通过；当前仍要求H紧跟B，未实现任意延后 |
| 历史贡献容易漏加、重加，GQA还需要跨Q heads求和 | 当前/历史分开路由；冻结每次前向布局；保持KV-head缓存及一次组内归约 | 覆盖多源累加、packed隔离、短尾和非对齐边界；不能删掉消费者与生命周期检查 |
| dQ与当前dKV分开扫描导致重复计算和启动 | **已合并全部dQ与当前dKV主kernel**，历史tile只算dQ，当前tile共享中间量 | 独立同轮消融支持局部收益；已接入训练。跨轮完整B差值不当作严格合并消融 |
| 完整长度区域输出和拼接产生无效工作 | 改为当前/历史紧凑最终输出；单序列直接返回，减少无意义拼接及重复状态检查 | 最终输出已紧凑，但每Q-head中间dKV工作区仍按全部KV长度分配，二者不要混淆 |
| 测量路径中的标注和初始化影响B/H | 纯任务计时模式关闭内部标注节点及两次纯标注backward，预初始化边界events，metadata/写文件移到末尾 | 独立Python入口审计60个B/24个H无违规。原同步Timer本就未启用，不能把此前回退归因于它 |
| attention局部更快，但小模型完整B仍慢 | 分开比较eager/Graph、Python/native、MHA/GQA及真实模型宽度/历史长度 | 包装/分派有独立对照证据；工作规模会改变结果。尚未精确拆分完整B中的各项成本 |
| 少做矩阵乘，kernel其他成本没有同比下降 | 核实仍沿用原FA的tile、较重共享内存和搬运/同步结构 | **尚未逐项优化或消融**；没有实际occupancy/Tensor Core/stall数据，不把“不饱和”作为唯一已证实原因 |

必要的CPU元数据检查、tensor版本/stream检查和历史状态协议仍保留；正式计时不加载梯度比较、CPU快照或profiler。优化目标是去除重复与无效工作，不以删除正确性保障换取时间数字。

### 3.2 当前性能结论及其范围

小模型曾出现完整B回退：hidden512、GQA4:2、Q2048/history4096的同轮融合/拆分为5.914/6.201 ms，约慢4.9%；关闭GQA也没有消除该回退。扩大真实负载后，hidden2048的长历史任务已出现B缩短，但部分短历史任务复测方向反转，因此不能说所有形状都受益。原因诊断和规模对照见[008](assets/fa-split/008-b-cost-diagnosis-20261016/)、[010](assets/fa-split/010-mha-comparison-20261016/summary.json)、[011](assets/fa-split/011-workload-scaling-20261016/summary.json)。

目前最直接的完整B证据来自hidden4096、FFN8192、GQA32:16、6层、PP3/VPP2、chunk2048。短数据长度模式为6144/4097/1537/257，长数据只将最长序列改为14336；仍是合成变长输入，不是完整LLaMA7B或真实业务分布。固定stage1/VPP chunk0、当前Q2048，每行每模式2个实际样本，单位ms：

| 数据/历史长度 | 原融合B中位数 | 拆分B中位数 | B缩短 | H中位数 | 配对B+H中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短数据 / 4096 | 8.015 | 7.478 | 6.7% | 2.025 | 9.503 |
| 长数据 / 4096 | 8.469 | 7.350 | 13.2% | 2.122 | 9.472 |
| 长数据 / 12288 | 13.183 | 10.613 | 19.5% | 6.284 | 16.897 |

各行对应B样本均缩短，但样本数有限，不代表所有任务的普遍收益。B/H时长是计算流边界区间，含辅助操作与流内等待；不是attention主kernel之和。B+H先按任务求和再取中位数，不等于训练iteration time。

**现有端到端日志尚未显示加速。** 同配置第2轮未采B/H，短/长负载iteration时间分别839.7→849.0 ms（+1.1%）、1334.6→1364.6 ms（+2.2%）。第3轮采集B/H，分别786.4→807.4 ms（+2.7%）、1587.9→1673.1 ms（+5.4%）；这一轮包含采样、同步和时序文件输出，不能作为无采样端到端结论。每配置只有一组运行；不同迭代的变长batch不同，不跨迭代平均作受控比较。第1轮初始化影响明显，不参与判断。数据均来自[012短负载](assets/fa-split/012-width4096-theory-20261016/hidden4096-short/)和[长负载](assets/fa-split/012-width4096-theory-20261016/hidden4096-long/)日志，仍需专门的无采样配对验证稳定幅度。

完整B/H样本见[012 summary](assets/fa-split/012-width4096-theory-20261016/summary.json)。[013放大时序图](assets/fa-split/013-measured-bh-timeline-20261016/paired-long-history.png)展示两个实际任务，[三卡概览](assets/fa-split/013-measured-bh-timeline-20261016/three-stage-bh-overview.png)包含全部采集事件。只采B/H，没有F/W和通信区间；空白不等于bubble，各rank时间原点也不是精确同步的跨GPU时钟。

### 3.3 为什么B变短，B+H仍可能更长

当前合并实现中，历史区域的主要矩阵乘在B内由5项变为3项，H执行4项，其中2项是额外重计算；因此历史区域总量是5→7，当前区域仍为5。这个40%的增加只针对被拆出的历史attention区域，不是完整模型或耗时增加40%。

令当前长度q、历史长度p、hidden为D、KV宽度Dkv、SwiGLU中间维F；`Ah=q·p`、`Ac=q(q+1)/2`。按乘加2 FLOPs计，不含独立W：

```text
投影/MLP输入反向 L = 2q[D(D+2Dkv) + D² + 3DF]
原B主要矩阵乘       = L + 10D(Ah+Ac)
拆分B主要矩阵乘     = L + 6D·Ah + 10D·Ac
从B移出的量 R       = 4D·Ah
H主要矩阵乘         = 8D·Ah = 2R
B+H新增重计算量     = R
```

在本轮hidden4096配置中，history4096/12288的B主要矩阵乘理论减少13.11%/23.76%，对应完整B实测时间减少6.7%–13.2%/19.5%。方向一致，但FLOPs不是耗时预测，公式不含softmax、归约、访存、边界tile冗余、分派或通信。宽度增大时投影/MLP约按D²增长，attention约按D增长，所以不能推断hidden越大可移出比例越大。完整推导和明确标注假设的等速率参考保留在[012理论对照](assets/fa-split/012-width4096-theory-20261016/theory-comparison.json)。

## 4. 下一步准备做什么优化

先做目标明确、便于对照的局部优化，再根据结果决定是否改写主kernel。以下均为**候选下一步，尚未实现**；已经完成的dQ/current合并不再重复列为待办。

| 优先级 | 优化 | 预期作用与验收重点 |
| --- | --- | --- |
| 1 | **无历史任务走原融合backward** | 没有H可移出时避免拆分成本。先覆盖整项physical task无历史的情况；混装中有/无历史序列的选择另评估，验证默认路径和缓存语义 |
| 2 | **合并dK/dV归约与输出处理** | 先将当前两次独立GQA归约合成一次kernel，再评估H归约与缓存累加融合。必须保持FP32累加/BF16舍入及每份贡献只入账一次；后者可能改变舍入或写入冲突，需要单独验收 |
| 3 | **压缩并复用中间工作区** | B只需要当前dKV，却按全部KV长度分配每Q-head中间量；修改区域地址并处理非对齐tile。复用不得覆盖H仍持有的数据；显存节省不自动等于耗时收益 |
| 4 | **合并准备入口并批量处理packed序列** | 减少Python状态封装、逐序列prepare/发射、LSE复制及结果拼接；保留必要生命周期检查，重点验证序列隔离和混合长短尾 |
| 5 | **针对dQ-only/H重新调整kernel资源** | 在独立诊断中测量共享内存、寄存器、访存与同步成本，再调整tile或分工。dQ最终转换需要等待部分结果累加，不能直接塞回主kernel；需先设计正确的归约机制 |

优先推进1、2，每次只改一类因素：先通过对应梯度/状态回归，再使用相同权重、变长batch、chunk和ordering比较完整B/H，报告原始样本及范围；方向冲突或波动妨碍判断时再加反序重复。性能运行与正确性/profiler运行分开。

端到端验证需要同时补齐两件事：在关闭B/H采样和文件输出的配对运行中判断真实iteration变化；在单独诊断中检查dx发送实际发起/接收时刻和消费者就绪，不能只看B结束。随后若设计跨任务延后的H调度，需要明确消费者期限、状态生命周期及通信推进，修改顺序后重新传播依赖和资源占用。当前B局部变短是继续研究的依据，是否能利用等待空隙抵消H成本仍是待验证问题。
