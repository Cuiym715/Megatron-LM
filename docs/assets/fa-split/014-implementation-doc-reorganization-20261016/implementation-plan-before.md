# DSPP 历史 dKV：FA backward 拆分实现计划

更新日期：2026-10-16  
状态：阶段 A、B、C 已实现并通过各自限定范围内的 GPU 正确性验收；完整 stage dx 后的独立 H 见 §14，纯 B/H 计时清理与最终重复结果见 §15。当前样例清理后 B 仍慢于融合，尚未达到性能目标。代码基点：`abd39e566`（`cuiym-dspp`），本轮改动尚未提交。阶段 A 历史记录见 §9–10，阶段 B 见 §11–12，原阶段 C 见 §13。当前为重计算版本，H 紧接 B 执行，尚不在二者间插入其他计算任务。

§16 完成首轮性能原因排查。§17 已将合并 dQ/current dKV 接入 after-dx 训练并通过正确性验收；两组纯 B/H 对照中，关注位置的完整 B 仍慢于融合，未证明完整 dx 提前或 iteration 收益。

## 1. 本轮目标与文档职责

原 A/B/C 范围是实现并验证 attention backward 的正确拆分：保留 FA 前向，提供独立的 dQ 和指定 KV 区域 dK/dV 计算入口，支持 MHA/GQA，最终接入 DSPP packed backward；历史部分在同一次 backward 内立即执行完毕，再返回梯度。用户随后补充的完整 dx／H 任务边界及独立计时要求由 §14 记录，扩展了原范围。

原 A/B/C 不以调度重放、完整 stage dx 提前或 iteration 加速作为交付。用户随后追加的局部测量见 §10/12，完整 dx 与 H 任务边界及计时见 §14；仍未开展跨其他计算任务延后的调度优化或端到端收益验证。

本文记录代码实现计划、阶段验收和最终已实现能力。项目外层 `docs/v4/design/history-dkv/plan.md` 继续承担研究收益验证计划的职责；其中原有“推演后再原型”的安排不作为本轮正确拆分开发的前置条件。本文不修改外层 design.md。

## 2. 开发前的代码依据

- `megatron/core/kv_cache/dspp_packed_attention.py`：`_DsppPackedFlashAttention.backward` 调用一次完整 `_wrapped_flash_attn_varlen_backward`，然后通过 `distribute_key_value_grads` 分发已经算好的历史梯度，并合并当前 segment 收到的贡献。尚无独立 D/H 执行接口。
- `megatron/model/transformer.py`：FA 分支直接传入 Q/K/V；GQA 的 KV heads 没有在该分支展开为 Q heads。新实现应保持这一紧凑布局。
- `tests/unit_tests/dspp/test_packed_attention.py`：现有基础 Q/K/V 测试构造使用相同 head 数，不足以证明 GQA 的拆分与历史累加正确性。
- 已有运行环境记录使用 FlashAttention `2.7.4.post1`。开发前还需核对实际安装版本、源码对应关系、CUDA/PyTorch ABI 和目标 GPU，不能仅凭版本名确定修改位置。

首轮固定 CP=TP=DP=1、BF16、causal、dropout=0、无 activation checkpoint，先支持 head dimension 128。RoPE 沿用现有非 fast 路径；后续测试可扩展 head dimension，但未验证组合须显式拒绝或清晰记录，不能悄悄以融合路径代替拆分后宣称通过。

## 3. 执行方式：建议三个阶段交付

此前讨论的四个技术步骤不建议一次全部执行。将“接口与参考实现”和“基础 MHA kernel 拆分”合为阶段 A；区域/GQA 为阶段 B；接入现有 backward 为阶段 C。每阶段内部连续完成代码、正确性测试和结果记录，阶段末汇报，再由用户决定是否推进下一阶段。

| 阶段 | 对应技术步骤 | 可独立验收的交付 | 本阶段结束时不代表什么 |
| --- | --- | --- | --- |
| A：基础 FA 拆分 | 原步骤 1、2 | 独立 dQ/dKV GPU 入口，MHA 单序列正确性通过 | 不代表历史区域选择、GQA 或 DSPP 集成完成 |
| B：区域与 GQA | 原步骤 3 | 当前/历史 KV 区域选择，原生 GQA/MQA 和 varlen 边界正确性通过 | 不代表真实模型的历史梯度路由已经接入 |
| C：DSPP 集成 | 原步骤 4 | 拆分路径可开关，packed、RoPE、历史累加及多层梯度验证通过 | 不代表历史计算已跨任务延后，也不代表性能改善 |

用户先后明确要求“执行阶段 A”“执行阶段 B”和“执行阶段 C”，目前已完成 A/B/C。随后又要求先交付完整 dx、立即执行独立历史 dKV 并分别计时，已按 §14 完成；跨其他计算任务延后的调度优化另行确定。

## 4. 拆分合同

### 4.1 接口与状态

以下是拟定接口职责，名称可在实现时调整：

```text
prepare_backward(Q, K, V, O, dO, LSE, metadata) -> backward_state
backward_dq(backward_state)                   -> 完整 dQ
backward_dkv(backward_state, kv_regions)      -> 指定区域 dK/dV
```

`backward_state` 显式持有所需 tensor、cu_seqlens、每序列 Q/K 长度、scale、dtype、head 映射和区域坐标。预处理计算一次 `delta = rowsum(dO * O)`；具体 LSE/delta 布局和数值精度须与所用 FA 内核对齐。区域不能依赖后续可变化的 `state.current_meta`。

正式实现保持分块融合与局部重计算，不生成完整 S/P/dS 矩阵。小规模稠密实现仅作为独立正确性参照。不能调用完整 FA backward 后丢弃 dKV 来冒充独立 dQ，也不能先算完历史 dKV 再仅把其路由推迟。

### 4.2 D/H 分工

- D：完整 dQ，加当前 KV 区域的 dK/dV。完整 dQ 必须包含全部合法历史区域的贡献。
- H：当前 query 对全部历史 KV 区域的 dK/dV。
- 首版 D 可由 `backward_dq(all)` 和 `backward_dkv(current)` 两个 kernel 路径组成，H 调用 `backward_dkv(history)`。不要求 D 融合成一次启动，也不承诺整个 backward 只有两次 kernel 启动。
- 首版只支持全部历史/当前的划分，不实现任意历史子集选择。无历史时 H 为显式空操作，dQ 和当前 dKV 仍必须完整。
- 两部分使用同一份完整上下文 LSE 和 delta；不能在 KV 子区域上独立归一化。简单拆法也可能在当前区域重复生成中间量，不能直接套用特定拆分区域的“5→7”作为实际执行成本。

## 5. 阶段 A：独立 dQ/dKV 的 MHA 基础实现

1. 核查 FA2 backward 的预处理、mainloop、dQ 累加/转换及 dKV 输出路径，固定源码版本和可复现构建入口。
2. 优先基于当前 FA2 对应 CUDA 实现建立独立扩展，保留其分块计算机制。新增源码/补丁、版本锁定、构建说明及许可证信息纳入仓库；不直接修改环境中的安装文件作为最终交付，不替换整个环境的默认 FA。
3. 实现独立 dQ 和完整 KV 范围的 dKV 入口，显式初始化累加工作区并完成必要转换。先支持单序列 MHA、causal、无历史区域选择；依次立即执行两个入口。
4. 建立小规模高精度数学参考，用非平凡随机 dO 对照梯度；同时与原 FA 前向/融合 backward 比较 BF16 结果。

验收：扩展可按记录构建，两个入口独立调用有效，合并输出的 dQ/dK/dV 正确；重复调用无残留梯度。未经证实的 CUDA 源码可复用性若阻碍实现，应先报告具体障碍和替代实现方式，不悄悄换成稠密 GEMM 版本后称为 FA 拆分。

阶段 A 不修改生产 custom autograd、KV 路由或 pipeline scheduler。接口从一开始区分 Hq/Hkv，但本阶段仅声明 MHA 已验收。

## 6. 阶段 B：历史区域与原生 GQA

### 6.1 区域、varlen 与 causal 坐标

先在 MHA 上加入每序列的当前/历史 KV 区域：设历史长度为 h，局部 query 位置 a 对应完整序列位置 h+a，key 位置 b 的合法条件为 `b <= h+a`。区域限制只选择输出的 KV 位置，不能错误裁剪完整 dQ 所需的历史扫描。

支持每序列独立 cu_seqlens、短尾、非 tile 对齐的区域边界，以及多个独立序列组成的 varlen 输入。首版允许各区域写入一个完整形状梯度 buffer 的互不重叠位置，以便后续复用路由；必须明确哪些区域已经写完，不能将未初始化位置交给消费者。

验收：当前与历史区域无重叠、无遗漏，二者组成完整 dKV；无历史、全历史前缀、短尾和边界 tile 均正确，不串序列。

### 6.2 GQA head 映射与归约

要求 `Hq % Hkv == 0`，令 `r = Hq / Hkv`。现有模型的分组布局下，Q head q 映射到 KV head `g = floor(q/r)`：

```text
dQ[q] 使用 K[g]、V[g]，保留 Q-head 维度
dK[g] = sum(dK_from_q[q] for q in group(g))
dV[g] = sum(dV_from_q[q] for q in group(g))
```

LSE/delta 仍按 Q head 索引；每个 Q head 有自己的概率与梯度，不能在计算前对 Q heads 取平均，也不能将 dKV 的组内求和误写为平均。

优先采用明确的部分梯度工作区加归约步骤，核对是否可复用 FA 原有 GQA 归约，保证只归约一次。工作区精度、清零、写入所有权、归约顺序及输出转换均须明确；正式路径不复制 K/V heads。参考实现可显式展开 KV，并由 autograd 归约，作为独立 oracle。

验收至少覆盖 `(Hq,Hkv)=(16,16)、(16,8)、(16,4)、(16,1)`，并与历史/当前区域划分、varlen/非对齐边界组合。使用组内不同的 Q/dO，暴露漏加、重复加及错误 head 映射。

阶段 B 可先交付区域划分，再交付 GQA；二者都通过才标记整个阶段完成。

## 7. 阶段 C：接入 DSPP packed backward

在 `_DsppPackedFlashAttention.backward` 增加实验路径，执行顺序为：

```text
prepare delta
  -> 完整 dQ
  -> 当前区域 dKV
  -> 历史区域 dKV
  -> distribute_key_value_grads
  -> 返回当前 Q/K/V 梯度
```

保留原融合路径为默认值，通过明确开关选择拆分。先复用现有 assembled dKV buffer 与 `distribute_key_value_grads`，不同时重写缓存协议。两部分均写完后才调用路由，因此此阶段无异步 pending 状态或跨任务 H 队列。

需要保证：

1. 当前 segment 的 dK/dV 在返回前合并后续 chunks 已产生的贡献；每份历史贡献只入账一次。
2. GQA 的历史缓存/累加缓冲保持 Hkv 维度；已归约的 dKV 不再次执行 Q-head 归约。
3. dK 在当前实现的 RoPE 反向入口之前合并，不对历史梯度重复或遗漏旋转反向。
4. physical padding 的梯度为零；输出保持原 Q/K/V 物理形状，包括不同 head 数。
5. 多层、不同 logical microbatch 和连续 iteration 的状态隔离正确，缓存按现有逆序反向合同排空。

验证从独立 attention 扩展到带 RoPE、QKV/输出投影的模块，再到多层 backward：比较输出、输入梯度及参数梯度。覆盖 continuation 加短序列混装、多源向同一目标累加和连续 iteration；最后在现有支持的 PP 路径做正确性 smoke，不采性能或开展调度实验。

完成后可声明“FA backward 已拆分，支持经测试的 GQA/packed 配置，历史部分立即执行”。不能声明“完整 stage dx 已提前”或“历史 dKV 已延后”。

## 8. 验证与结果记录规则

数值验收同时参考高精度数学实现与原融合 FA；根据 dtype 和原路径自身误差预先确定容限，报告最大绝对误差及适当的相对/范数误差，不用近零元素的相对误差单独判定。浮点归约顺序变化允许合理舍入差异，不要求逐位相等，也不能为通过测试临时放宽容限。

GPU 测试被 skip、仅 Python 接口可调用或只有 CPU 公式通过，均不算 kernel 验收完成。保留原融合路径回归测试，验证默认行为未变。每阶段汇报修改文件、已支持形状、执行环境、测试命令/结果和限制。

| 项目 | 状态 | 已验证功能／证据 |
| --- | --- | --- |
| 实现计划 | 已写入并更新，2026-10-16 | 本文 |
| A：MHA 独立 dQ/dKV | 已完成 | 独立 CUDA 入口；29 项拆分正确性测试、10 项原路径回归通过；详见 §9 |
| B：区域与 GQA | 已完成 | 当前／历史区域、逐序列 varlen 分派、原生 GQA/MQA；60 项阶段 B 测试通过，详见 §11 |
| C：DSPP 集成 | 已完成（限定范围） | 默认关闭的 packed backward 拆分；RoPE／投影／多层、真实 PP=3/VPP=2 的 MHA/GQA/MQA 训练对照通过，详见 §13 |
| C 后补充：完整 dx 后独立 H | 已完成（限定范围） | 重计算方案；B 返回完整 dx、发布边界后立即执行 H；正确性和 B/H 独立 CUDA-event 时间见 §14 |

最终实现功能与限制集中更新本表及相应阶段，不另建重复报告。用户追加的局部性能测量、dx/H 边界分别见 §10/12/14；进一步重放及跨其他任务延后仍另行确定。

## 9. 阶段 A 实现与验收（2026-10-16）

### 9.1 实际实现

新增独立实验模块，生产 `dspp_packed_attention.py`、KV 路由和 pipeline scheduler 均未修改：

| 文件 | 职责 |
| --- | --- |
| [fa_split.py](../megatron/core/kv_cache/fa_split.py) | `prepare_backward`、`backward_dq`、`backward_dkv`；状态持有、修改检测和同 stream 合同 |
| [backward.cu](../megatron/core/kv_cache/fa_split_src/backward.cu)／[bindings.cpp](../megatron/core/kv_cache/fa_split_src/bindings.cpp) | 三个独立 GPU 操作的参数校验、工作区分配和调用入口 |
| [flash_bwd_split.patch](../megatron/core/kv_cache/fa_split_src/flash_bwd_split.patch) | 为原 FA2 column-block backward 增加编译期开关，真正裁掉不需要的梯度 GEMM 和写回 |
| [fa_split_build.py](../megatron/core/kv_cache/fa_split_build.py)／[构建脚本](../tools/build_dspp_fa_split.py) | 锁定依赖、校验源码、在缓存目录生成补丁头文件与独立扩展 |
| [test_fa_split.py](../tests/unit_tests/dspp/test_fa_split.py) | 与融合 FA、独立 FP64 autograd 参考比较，检查边界和状态合同 |

源码固定为 FA `v2.7.4.post1` / `5231d95fe13733fb534c01895f7ea88c6a6c7793`、CUTLASS `c506e16788cb08416a4a57e11a9067beeee29420`。源码版本、原始头文件 SHA256 和许可证见 [upstream.json](../megatron/core/kv_cache/fa_split_src/upstream.json)、[THIRD_PARTY.md](../megatron/core/kv_cache/fa_split_src/THIRD_PARTY.md) 及同目录许可证文件。安装环境中的 FA 不被替换，显式 `--fetch` 才获取依赖；正常调用只使用已经取得的固定源码。

实际执行分工：

- prepare 复用 FA2 的 `compute_dot_do_o`，计算一次 FP32 delta。
- dQ-only 复用分块 P/dS 重计算与 dQ GEMM，禁用 dK/dV GEMM 及写回；每次调用独立清零 FP32 dQ 累加缓冲，再调用原 FA2 转换 kernel。
- dKV-only 复用分块 P/dS 重计算与 dK/dV GEMM，禁用 dQ GEMM 和累加写回；不分配 dQ 输出或累加缓冲，可先于 dQ 独立调用。
- 采用 FA2 在 L40 上的 BF16/head-dim-128 tile 配置；保留共有的加载/同步结构，不宣称已完成各分支性能调优。没有完整 S/P/dS 显存矩阵，也没有调用完整 backward 后丢弃梯度的替代实现。

### 9.2 已支持范围与限制

接口输入为单序列 `[S,H,128]` 连续 BF16 Q/K/V/O/dO，Q/K 长度相等、MHA；LSE 为匹配 FA 前向输出的连续 FP32 `[H,S]`，使用自然对数。前向必须为全上下文 causal、dropout=0、无 softcap/ALiBi/local window，softmax scale 与 backward 一致。

本轮验证 NVIDIA L40（SM89）、PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`、CUDA toolkit 13.1、FA `2.7.4.post1`、Python 3.12。扩展按当前 GPU 架构构建，其他架构没有本轮验证结论。

GQA/MQA、不同 Q/K 长度、varlen、多序列 packing、历史区域选择与生产 autograd 集成仍未支持。输入不符时显式报错，不回退为原融合路径假装拆分。接口不提供高阶 autograd；调用者必须提供来自同一前向的 O/LSE/scale，不能只凭形状判断它们的语义匹配。

prepare/dQ/dKV 必须在同一 CUDA stream 执行；允许非默认 stream，也验证了当前设备不同于输入设备的情形。状态 tensor 持有引用并检查版本，调用间不得修改；这不是后续异步延后所需的完整生命周期协议。dQ 沿用 FP32 原子累加，不承诺逐位确定性。

### 9.3 正确性结果与数值边界

机器可读结果及原始测试输出归档于 [001-stage-a-20261016](assets/fa-split/001-stage-a-20261016/)。没有采集 iteration 性能或 profiler trace。

- 新增测试 **29 passed、0 skipped**：覆盖 S=1/17/63/64/65/127/128/129/257/513，H=1/2/4/16，自定义 scale、随机 dO、零 dO、两入口不同调用顺序、重复调用、状态不被改写、修改检测、非法配置、非默认 stream 及跨设备 guard。
- 原 `test_packed_attention.py`、`test_kv_state.py` 回归 **10 passed、0 skipped**。
- Compute Sanitizer 仅检查新扩展的 `dspp_split_detail` kernels：memcheck 的普通形状/重复调用 **12 passed、0 errors**；racecheck 的重复调用 **1 passed、0 hazards**。这覆盖所选用例的越界/内存访问和 shared-memory 竞态检查，不是全配置无缺陷证明。
- 数值结果来自 11 组普通形状/scale 和 3 组较大输入幅度的 seed。最终普通测试运行中 dK/dV 与原融合 FA 逐位一致；dQ 对原 FA 最大绝对差 `9.765625e-4`，最大相对 L2 差约 `1.995e-5`。dQ 的小差异与原子累加顺序有关，此处不是确定性保证。
- 对独立 FP64 数学参考，非零梯度的最大相对 L2 误差分别为 dQ `0.004786`、dK `0.004675`、dV `0.002305`；这是所测样例的数值误差，不是性能指标。S=1 的数学零 dQ/dK 单独检查绝对误差 `<=1e-5`，相对误差记为 null。

**保留的验收修正依据：** 首轮预设普通输入的 FP64 逐元素 `atol=rtol=0.04`，对幅度为 2、S=65/H=2/seed=1234 的压力样例，拆分 dQ/dK 未通过。独立复核发现原融合 FA 也未通过同一阈值，且该样例三项拆分梯度与原 FA 完全一致；原/拆分的 dQ、dK 最大绝对误差分别约 `0.08503`、`0.09683`。没有改 kernel 掩盖该现象，也没有放宽普通样例阈值。

因此压力样例单列：要求与原融合 FA **逐位相等**，同时保持对 FP64 的相对 L2 `<=0.012`，再用 seed=2026/17 验证。三组均通过。普通样例继续使用原来的融合对照 `atol=0.004/rtol=0.01` 和 FP64 逐元素/范数条件；不能把 BF16 梯度描述为与 FP64 精确相同。

### 9.4 构建与复现

在 GPU 可访问、依赖齐全的环境中，从仓库根目录执行：

```bash
python tools/build_dspp_fa_split.py --fetch
python -m pytest -q -s tests/unit_tests/dspp/test_fa_split.py
python -m pytest -q tests/unit_tests/dspp/test_packed_attention.py tests/unit_tests/dspp/test_kv_state.py
compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=dspp_split_detail \
  python -m pytest -q tests/unit_tests/dspp/test_fa_split.py -k 'matches_fused or repeat_zero'
compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=dspp_split_detail \
  python -m pytest -q tests/unit_tests/dspp/test_fa_split.py -k repeat_zero
```

需要 C++/CUDA 编译器、ninja、git、patch、PyTorch、pytest、现有 Megatron Python 依赖和 FA。默认依赖/构建缓存为 `~/.cache/dspp-fa-split`，可用 `DSPP_FA_SPLIT_CACHE` 指定；`MAX_JOBS=2` 可限制编译并行度。导入模块不会下载依赖，首次显式使用会构建/加载缓存中的扩展。

本轮使用独立临时容器 `nvcr.io/nvidia/pytorch:26.01-py3`，挂载 `/home/cuiym:/workspace`，工作目录 `/workspace/src/Megatron-LM-kwai`，Python 为 `/workspace/src/venvs/megatron/bin/python`。以宿主 UID/GID 1001 运行，设置 `USER=cuiym`（该 UID 未登记在镜像 passwd）、`TORCHINDUCTOR_CACHE_DIR=/tmp/dspp-split-inductor`、`DSPP_FA_SPLIT_CACHE=/workspace/.cache/dspp-fa-split`，pytest 使用可写的 `/tmp` cache_dir。原训练容器未重启或修改。

阶段 A 到此结束。下一阶段是区域划分和 GQA，需按后续指令推进；当前没有历史 dKV 跨任务延后、完整 stage dx 提前或端到端收益结论。

## 10. 阶段 A 的可视化执行证据（2026-10-16）

用户在阶段 A 验收后要求提供执行图和更直观的结果，因此补充一次独立 attention 的真实 profiler 采样。它不扩展到阶段 B/C，不修改 kernel、模型或生产调度，也不是变长负载/iteration 性能基线。

### 10.1 真实 GPU 时间线

固定 L40、BF16、S=2048/H=16/d=128、causal、dropout=0，使用相同随机 Q/K/V/dO（seed=1234）。预热后采集五个执行样例：原融合路径、dQ→dKV、dKV→dQ、仅 dQ、仅 dKV。每例都使用原 FA 前向；拆分例计算共享 delta。通过 CUDA runtime correlation 将 GPU kernel 对应到标记范围，记录真实设备起止时间，不用 CPU 标签宽度冒充 GPU 执行时间。

![实际 GPU 执行与独立 kernel 启动次数](assets/fa-split/002-execution-evidence-20261016/gpu_execution.png)

采到的独立主 kernel 名称是 `dspp_split_detail::split_kernel<true>`（dQ）和 `split_kernel<false>`（dKV）。程序检查了它们与原融合 main kernel 的启动次数：

| 执行样例 | 独立 dQ 主 kernel | 独立 dKV 主 kernel | 原融合主 kernel |
| --- | ---: | ---: | ---: |
| 原 FA | 0 | 0 | 1 |
| dQ→dKV | 1 | 1 | 0 |
| dKV→dQ | 1 | 1 | 0 |
| 仅 dQ | 1 | 0 | 0 |
| 仅 dKV | 0 | 1 | 0 |

仅 dKV 时也没有启动 dQ 转换 kernel。这些设备执行记录与 §9 的编译期计算裁剪共同支持“已独立拆分”，不只是增加两个 Python 标签。

**图的范围：** F_attn 只是 attention 前向，B_attn 只是 attention backward。完整 stage B 还包含其他层、投影等反向，W 是参数梯度任务，阶段 A 未接入它们。橙色表示完整 dKV，不是已经选出来的历史贡献 H。因此当前不能画一张真实的“完整 F/B/W＋历史 H”训练图，也不能将绿色 dQ 的结束标为完整 stage dx 就绪。

本次单个 profiler 样例中，原融合主计算 kernel 约 358.91 μs；顺序拆分的 dQ 主 kernel 约 263.01 μs、dKV 主 kernel 约 300.96 μs，另有预处理、清零、转换和间隔。它明确没有给出加速证据。这里的数字仅说明本次执行记录，不能当作稳定成本基准；不能用两行总跨度的差推导加速比，或把空白当作可插入历史任务的空隙。

### 10.2 独立调用与数值结果

本次 S=2048/H=16 的所有拆分执行方式都通过原融合 FA 对照：dK/dV 逐位相同；dQ 最大绝对差 `4.8828125e-4`，最大相对 L2 差 `1.145e-5`。仅 dQ 和仅 dKV 的结果也分别通过对应梯度比较；dKV 先执行不依赖 dQ 的调用完成。

另外，将 §9 已归档的 14 组数值结果绘图，展示原融合/拆分 BF16 backward 对同一个独立 FP64 数学参考的误差。S=1 的数学零梯度用绝对误差验收，未放入相对误差图。

![原融合与拆分的梯度误差](assets/fa-split/002-execution-evidence-20261016/gradient_accuracy.png)

绿点与灰圈基本重合，支持所测样例没有因拆分显著损失精度；BF16 本身仍有舍入误差。此前压力样例的初始失败及单列验收理由仍保留在 §9.3，没有删除或改写为全部初次通过。

### 10.3 材料与复现

新增 [trace_dspp_fa_split.py](../tools/trace_dspp_fa_split.py) 和 [plot_dspp_fa_split.py](../tools/plot_dspp_fa_split.py)。[002 归档](assets/fa-split/002-execution-evidence-20261016/) 保存 GPU 事件及 kernel 全名、correlation、梯度比较、图片 PNG/SVG、文件校验值与运行输出。

在 §9.4 的 GPU 环境中运行（输出目录必须尚不存在）：

```bash
python tools/trace_dspp_fa_split.py --output /tmp/fa-split-evidence --length 2048 --heads 16
python tools/plot_dspp_fa_split.py \
  --evidence /tmp/fa-split-evidence/evidence.json \
  --validation docs/assets/fa-split/001-stage-a-20261016/validation.json \
  --font /path/to/NotoSansCJK-Regular.ttc
```

本机原始 `execution.pt.trace.json` 可用 Perfetto 等 trace viewer 打开，按现有 `.gitignore` 不进入普通 Git 提交；其 SHA256 保存在 `evidence.json`。可提交的 `evidence.json` 已保存绘图使用的全部 GPU 事件，原始 CPU/GPU trace 可通过脚本重新采集。上述 kernel 时间线是新增采样，梯度误差图则复用 001 的已有结果，两者不混作同一实验。

## 11. 阶段 B 实现与验收（2026-10-16）

阶段 B 已完成，阶段 C 未开始。本节描述当前代码；§9–10 保留阶段 A 当时的实现范围和测量，不能将旧时间线当作本版性能结果。实现仍是独立 attention backward，没有接入生产 autograd、历史梯度路由或 pipeline scheduler。

### 11.1 新增能力与执行依据

- **当前／历史区域：** `backward_dkv(state, region="all"|"current"|"history")`。每条序列的历史长度为 `Sk-Sq`，Q 对应 KV 的末尾当前区域；允许 `Sk>=Sq`，沿用 FA2 的 bottom-right causal mask。`backward_dq` 始终扫描全部合法 KV，两种区域共享同一次完整前向的 O/LSE 和 delta，不在子区域重新归一化。
- **实际计算范围：** CUDA launcher 仅启动与所选区域相交的 64-token KV tiles。非对齐边界最多使当前／历史两次调用重复一个边界 tile；边界多算位置不写入返回梯度。输出保持完整 `[Sk,Hkv,128]`，未选区域严格为零，因此当前与历史结果可直接相加。空历史返回全零，不启动 dKV 主 kernel；首版没有任意历史子集选择。
- **原生 GQA/MQA：** 使用 FA2 原有的 `q_head // (Hq/Hkv)` 访问紧凑 K/V，不复制输入 heads。复用原 FA2 backward API 的梯度精度合同：main kernel 在 FP32 累加每个 Q head 的 dKV，转换为 BF16 部分梯度工作区；ATen 按组以 FP32 求和、最终转换为 BF16，每份贡献只归约一次。区域选择后只读取已写入的工作区行，返回 Hkv 维度梯度。
- **变长：** `prepare_backward(..., cu_seqlens_q=..., cu_seqlens_k=...)` 接受整数 tuple/list 或 CPU／同 GPU 的 int32/int64 tensor；输入为 `[total_tokens,heads,128]`，LSE 为 `[Hq,total_q]`。首版逐序列调用原生 CUDA kernels，并拼接梯度，未实现单次批量 varlen 启动。CUDA 边界数据在 prepare 时同步到主机，冻结为不可变坐标；各序列使用自己的完整上下文 LSE、历史长度和 causal 偏移。
- **状态合同：** 保留同 stream、设备 guard、tensor 版本检测；增加原始 packed LSE、metadata tensor 和逐序列 delta 的修改检测。只接收严格递增的序列边界，拒绝空序列、`Sk<Sq`、非法 head 比例；空历史合法。BF16、head dim 128、causal、dropout=0 等限制沿用前述合同。

修改集中在 [fa_split.py](../megatron/core/kv_cache/fa_split.py)、[backward.cu](../megatron/core/kv_cache/fa_split_src/backward.cu)、[bindings.cpp](../megatron/core/kv_cache/fa_split_src/bindings.cpp)。原 FA header 的拆分补丁没有变化；新增 [test_fa_split_regions.py](../tests/unit_tests/dspp/test_fa_split_regions.py)，原测试中“拒绝合法 GQA”改为“拒绝 Hq 不能被 Hkv 整除”，原阶段 A 的数值容限不变。

首版有逐序列分派、输出置零／拼接和完整形状的部分梯度工作区；工作区规模为每序列两份 `[Sk,Hq,128]` BF16，而非 Hkv 大小。这些是明确的实现成本，当前没有调优或收益结论。部分梯度工作区不等于复制 K/V 输入，也不是保存完整 S/P/dS。

### 11.2 GPU 验证结果

环境仍为 L40 SM89、PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`、CUDA 13.1、FA `2.7.4.post1`；使用独立临时容器 `dspp-fa-split-stage-b`，原训练容器不变。本版扩展缓存标识 `e10d6e9353428102`。原始输出、机器可读结果、源码身份和启动检查归档于 [003-stage-b-20261016](assets/fa-split/003-stage-b-20261016/)。

| 验证 | 结果与范围 |
| --- | --- |
| 阶段 B 新测试 | **60 passed，0 skipped**：28 组单序列区域/GQA/FP64 对照、12 组 varlen 对照与隔离、2 组长上下文 GQA/MQA，以及状态、非法输入、stream／device、启动范围检查 |
| 阶段 A 与生产路径回归 | **29＋10 passed，0 skipped**；合计 **99 passed** |
| memcheck | 41 项区域/GQA/varlen/重复调用通过；过滤检查新扩展 kernels，**0 errors** |
| initcheck | 4 项 CUDA metadata 的 varlen MHA/GQA/MQA 通过；不做 kernel 名过滤，包含归约和拼接，**0 errors** |
| racecheck | 1 项 GQA 区域重复调用通过；检查新扩展 kernels，**0 hazards** |

数值验收在首次执行前沿用阶段 A 普通样例标准：对融合 FA 的 `atol=.004, rtol=.01`；小形状对 FP64 的 `atol=.04, rtol=.04` 加相对 L2 `<=.012`，数学零梯度单独检查绝对误差。没有为阶段 B 放宽阈值。

- 覆盖 `(Hq,Hkv)=(16,16)/(16,8)/(16,4)/(16,1)`、无历史、63/64/65-token 历史边界、短尾、多个独立序列、随机且组内不同的 Q/dO、自定义 scale、不同调用顺序及重复调用。
- 每次区域检查中，**current＋history 与 all dKV 逐位一致**，未选区域严格为零。设置仅一条序列 dO 非零时，其他序列三项梯度严格为零。另用错误的“只扫描当前 KV 的 dQ”作负对照，确保没有漏掉历史贡献。
- 阶段 B 数值样例中 dK/dV 与原融合 FA 逐位一致；dQ 最大绝对差 `4.8828125e-4`、最大相对 L2 差约 `2.6043e-5`。这些是实测样例结果，不承诺原子累加的逐位确定性。
- 小形状对独立 FP64 参考的最大相对 L2：dQ `0.002882`、dK `0.003351`、dV `0.003042`；原融合 FA 的对应最大值相同。
- 长形状额外验证 `Sq=2048, Sk=6144, Hq=16, Hkv=4/1`，对应当前 2048 tokens 加历史 4096 tokens；两种配置均通过原融合 FA 对照和区域合并检查。长形状没有运行 FP64 稠密参考，也没有测量性能。

实际 GPU 启动检查进一步确认区域选择不是“全算完再裁剪”：

| Sq / Sk，Hq=16、Hkv=4 | history KV tiles | current KV tiles | all KV tiles |
| --- | ---: | ---: | ---: |
| 129 / 1025（历史 896，边界对齐） | 14 | 3 | 17 |
| 129 / 1026（历史 897，边界不对齐） | 15 | 3 | 17 |
| 17 / 17（无历史） | 0，不启动主 kernel | 1 | 1 |

这些数字来自 profiler 记录中的实际 kernel 名称与 launch grid；未启动原融合 backward 或 dQ 转换 kernel。第二行明确显示边界 tile 的重复工作；这里没有用时间线推算收益。原始 trace 以 `.pt.trace.json` 归档在本机，按现有规则忽略 Git；可提交的 `validation.json` 保留启动次数、grid 和数值结果，manifest 保存文件 SHA256。

### 11.3 复现与后续边界

在 §9.4 相同 GPU 环境中，从仓库根目录执行：

```bash
python tools/build_dspp_fa_split.py
python -m pytest -q -s tests/unit_tests/dspp/test_fa_split.py \
  tests/unit_tests/dspp/test_fa_split_regions.py \
  tests/unit_tests/dspp/test_packed_attention.py tests/unit_tests/dspp/test_kv_state.py
compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=dspp_split_detail \
  python -m pytest -q tests/unit_tests/dspp/test_fa_split_regions.py \
  -k 'regions_gqa_fused or varlen_regions or repeat_order'
compute-sanitizer --tool initcheck --error-exitcode 99 \
  python -m pytest -q tests/unit_tests/dspp/test_fa_split_regions.py -k 'varlen_regions and cuda'
compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=dspp_split_detail \
  python -m pytest -q tests/unit_tests/dspp/test_fa_split_regions.py -k region_repeat_order
```

下一阶段仍是 C：接入现有 packed backward，验证 RoPE、QKV/输出投影、多层梯度与历史梯度路由。当前完成的 GQA/varlen kernel 验证，不等于这些模型级功能已通过，也不等于完整 stage dx 提前或历史计算已延后。本轮不修改外层研究计划或 `history-dkv/design.md`，不提交／推送 Git。

## 12. 阶段 B：chunk 场景的耗时与具体正确性证据（2026-10-16）

用户要求补充与阶段 A 类似的执行图、实际耗时和具体测试解释，因此新增本节；不进入阶段 C。脚本 [measure_dspp_fa_regions.py](../tools/measure_dspp_fa_regions.py) 生成输入、独立执行 attention、计时并核对梯度；[plot_dspp_fa_regions.py](../tools/plot_dspp_fa_regions.py) 绘图。材料集中在 [004-region-evidence-20261016](assets/fa-split/004-region-evidence-20261016/)。本节是新的局部实测，不改变 §11 当时没有采性能的记录。

### 12.1 测量对象与方法

在真实 L40 上，将生成的 6144-token Q/K/V 切成 3 个 2048-token query chunks，第 1／2／3 段分别使用长度 2048／4096／6144 的完整前缀 K/V。Hq=16、Hkv=16/4/1（MHA/GQA/MQA）、head dim=128、BF16、causal、dropout=0、seed=1234。输入是独立随机张量，不是模型投影输出或真实业务数据。

比较同一 chunk 的三条路径：原融合 FA、独立 dQ＋全部 dKV、独立 dQ＋当前 dKV＋历史 dKV。重复计时不包含 forward，也不包含将两个完整形状区域结果相加的操作；包含当前 backward API 的共享预处理、清零、转换、复制/GQA 归约，以及设备流上观察到的 CPU 提交间隔。每项预热 10 次，使用 CUDA events 计时 30 次，每轮轮换三条路径的测量顺序；报告中位数和 P10/P90。计时标记自身也有开销，因此这不是无仪器扰动的理想 kernel 吞吐。

完整路径总时间直接用起止 events 测量，不能将各部分中位数简单相加。另在重复计时结束后单独采 profiler，用 CUDA runtime correlation 关联 GPU kernels；图中时间是该单次样本，不能与重复计时中位数混用。图中额外的“历史先算”用于调用顺序正确性展示，未纳入三条路径的重复计时。

![阶段 B 真实 GPU 执行](assets/fa-split/004-region-evidence-20261016/gpu_region_execution.png)

### 12.2 耗时结果及其含义

第 3 个 chunk（当前 Q=2048、历史 KV=4096、总 KV=6144）的调用总耗时，单位 ms：

| 配置 | 原融合 FA | dQ＋全部 dKV | dQ＋当前＋历史 dKV | 后者 P10–P90 |
| --- | ---: | ---: | ---: | ---: |
| MHA，Hkv=16 | 1.643 | 2.551 | 2.680 | 2.653–2.704 |
| GQA，Hkv=4 | 1.625 | 2.437 | 2.569 | 2.553–2.590 |
| MQA，Hkv=1 | 1.604 | 2.421 | 2.540 | 2.532–2.565 |

GQA 第 3 段的拆分部分计时中位数：共享预处理 **0.043 ms**，完整 dQ **1.107 ms**，当前 dKV **0.342 ms**，历史 dKV **1.055 ms**。这些是各 API 部分的 event 时间，含其清零／转换／归约和提交间隔，不是单个主 kernel 的耗时。对应单次 profiler 中三个主 kernel 分别约 **1.014／0.303／0.970 ms**；两种口径不能互换。

从拆分调用起点到当前 dKV 完成，直接计时中位数为 **1.512 ms**；原融合调用为 **1.625 ms**。这只表示本次独立 attention 的完整 dQ 和当前 dKV 贡献在这一时刻完成，不是完整 stage dx；此处没有模型投影反向、后续层、历史梯度缓存消费者或流水线等待。历史计算仍须完成，区域拆分总耗时反而增加约 **58.1%**；相对 dQ＋全部 dKV，进一步划分当前／历史又增加约 **5.4%**。本次结果没有证明端到端收益。

![三个 chunk 的重复计时](assets/fa-split/004-region-evidence-20261016/repeated_backward_timing.png)

原阶段 A 的 359/263/301 μs 来自等长 Q/K=2048 的主 kernel 单次采样；这里第 3 段 K 长度为 6144，且表格是包含辅助操作的重复 API 计时，不能将两组数直接作性能回归比较。

### 12.3 测了什么，具体观察到什么

| 要排除的问题 | 对照方法与实际观察 |
| --- | --- |
| 只是改标签，实际仍算全部 dKV | 第 3 段的真实 launch grid：dQ 为 `[96,1,16]`，当前 dKV 为 `[32,1,16]`，历史 dKV 为 `[64,1,16]`；三种 head 配置都符合，仅选区相交的 KV tiles 被启动，拆分路径没有启动原融合 backward |
| 漏算、重复计算或写错区域 | 第 1／2／3 段分别与原融合 FA 对照；当前结果在历史区为零、历史结果在当前区为零；第 3 段三种 head 配置的合并 dK/dV 对原融合最大绝对差均为 0，dQ 不超过 `4.8828125e-4`；交换历史计算的先后顺序也通过 |
| GQA 共享 KV 的梯度归约不正确 | Hq=16，Hkv=16/4/1 各自与原生 FA 对照；小规模 FP64 参考通过显式展开 KV 并交给 autograd 求导，提供独立于本实现 head 归约的参照。§11 还覆盖 Hkv=8 |
| 各 chunk 的历史贡献累加后不正确 | 新增完整 145-token 序列切为 **63＋65＋17** 三段的检查：逐段算 Q 与前缀 KV，当前／历史结果按源位置手动以 FP32 累加。三种 head 配置下，组合 dQ/dK/dV 对“逐 chunk 原融合 FA 后同样累加”的最大绝对差均为 **0** |
| 切分对照也可能共享同一个错误 | 同一 145-token 序列再做一次完整 causal FA backward，并独立做 FP64 数学 autograd；三种 head 配置的拆分组合梯度对 FP64 最大相对 L2 为 **0.003077（0.308%）**，原整段 FA 最大为 **0.003058（0.306%）**，均通过预设阈值。拆分组合与整段 FA 不逐位相等，存在分段 BF16 舍入及累加顺序差异 |
| 非对齐边界、混装序列串扰 | 复用 §11 的直接证据：历史 897 时区域 tiles 为 15＋3，对完整 17，多出一个边界 tile；结果仍正确。多序列样例中只给一条序列非零 dO，其他序列梯度严格为零。此项来自 003 测试归档，不冒充 004 新采样 |

三段梯度组合检查是测量脚本里的独立 attention 数学验证，没有调用生产 KV cache 路由，也没有执行完整模型训练。它支持“切分后的当前／历史贡献计算与手动累加正确”，不能替代阶段 C 的缓存、RoPE、投影和多层集成验收。

### 12.4 复现与归档

新容器第一次采样在 CUDA 初始化处失败，未执行任何计时；失败日志保留。显式令 `LD_LIBRARY_PATH` 优先使用镜像的 `/usr/local/cuda/compat/lib.real` 后恢复 GPU 访问；宿主驱动仍为 550.120，没有改动宿主驱动或原训练容器。实际环境与源码 SHA256、各次原始时间、梯度误差和完整 kernel 事件保存在 `evidence.json`、`manifest.json`。原始 profiler trace 按已有规则不提交 Git。

在 §9.4 的环境中设置兼容库路径后运行（输出目录须不存在）：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/nvidia/lib64
python tools/measure_dspp_fa_regions.py --output /tmp/fa-region-evidence --warmup 10 --repeats 30
python tools/plot_dspp_fa_regions.py --evidence /tmp/fa-region-evidence/evidence.json \
  --font /path/to/NotoSansCJK-Regular.ttc
```

本轮只新增证据脚本、归档和本文，没有修改阶段 B kernel 或生产执行路径。

## 13. 阶段 C：接入真实 packed backward 与训练（2026-10-16）

用户明确要求执行阶段 C 后完成本节工作。§9–12 是各阶段当时的历史记录；其中“尚未接入模型／C 未开始”不再描述当前状态。当前交付是**拆分后立即执行的模型集成**，没有 H 延后队列，也没有提前发送完整 stage dx。

### 13.1 实际改动与执行合同

- [arguments.py](../megatron/arguments.py) 新增 `--dspp-split-attention-backward`，默认关闭；要求 DSPP、BF16、head dim=128、local transformer、无 ALiBi。沿用 DSPP 的 TP=CP=DP=1、dropout=0、无 activation checkpoint 等约束。
- [dspp_packed_attention.py](../megatron/core/kv_cache/dspp_packed_attention.py) 在前向保存开关和 metadata；backward 使用前向保存的 Q/K/V/O/LSE 和该次 dO，依次执行 `prepare → 完整 dQ → current dKV → history dKV → 区域结果相加 → 原有梯度路由`。host 端 cu_seqlens 来自冻结的 metadata，不读取后来被选中的 microbatch，也不新增 CUDA metadata 回读。
- 当前和历史区域输出在区域外均为零，合并后只调用一次 `distribute_key_value_grads`。当前 segment 收到的后续 chunks 梯度继续按原协议合并，再经原有 RoPE 反向及投影反向传播。GQA 缓存保持 Hkv 维度，不再次做 head 归约。所有历史计算完成后才从 attention backward 返回。
- [schedules.py](../megatron/core/pipeline_parallel/schedules.py) 只向单 stage 和 PP=3/VPP=2 的 KV state 传递开关；任务顺序、通信及缓存生命周期协议未改。阶段 B 的 CUDA kernel 本轮未修改。
- [run_dspp_b23_l40.sh](../examples/run_dspp_b23_l40.sh) 支持 `DSPP_SPLIT_ATTENTION_BACKWARD=1` 和可选 `NUM_QUERY_GROUPS`。不启用时沿用融合 backward。

正确性首版仍有逐序列 Python 分派、完整形状区域输出与合并、部分梯度工作区等开销，未做性能调优。支持范围为本轮验证的 L40、FA2 `2.7.4.post1`、BF16/d128、causal、无 dropout、非 fast RoPE；不据此扩展到 CP/TP、多种 dtype 或任意 head dimension。

### 13.2 模块与缓存正确性：具体检查及观察

新增 [test_packed_split.py](../tests/unit_tests/dspp/test_packed_split.py)。最终组合运行 **128 passed、0 skipped**：24 项 C 测试，加 104 项 A/B、原 packed/KV state 和单 stage runtime 回归。原始输出和逐 tensor 数值集中在 [005-stage-c-20261016](assets/fa-split/005-stage-c-20261016/)。

| 检查目标 | 实际覆盖与观察 |
| --- | --- |
| 真正进入拆分路径 | 拆分测试把原融合 backward 替换为报错函数；默认路径则令拆分入口报错，均正常完成。另检查前向保存开关，之后改 state 开关也不会切换该次 backward |
| packed 历史路由正确 | 长度 145/130/17/3、chunk=64，另含纯短序列 17/3；Hq=16，Hkv=16/8/4/1。检查每层每个 physical microbatch 只路由一次、历史累加 buffer 保留 Hkv、最后缓存排空。最终 dK/dV 对融合均逐位一致；dQ 最大绝对差 `3.8147e-6` |
| padding 与布局隔离 | padding 上给予非零上游梯度，仍检查输出及 Q/K/V 梯度的 padding 全为零；backward 前故意选择错误的 `current_meta`，验证使用前向保存布局 |
| RoPE、投影和多层梯度 | 1 层／3 层 block，包含仓库实际 RoPE、QKV/输出投影、LayerNorm、残差、GELU MLP；Hq=4、Hkv=4/2/1。同一 state 复用 2 iteration × 2 种 logical batch，比较每个输出、输入梯度和全部参数梯度 |
| 独立数学参照 | 对上述 block 做未切分整序列 FP64 autograd，独立实现 RoPE、显式展开 KV heads，权重/输入取同一 BF16 值。最终所有检查的拆分对 FP64 最大相对 L2 为 `0.00968971`（约 0.969%） |
| 接口与单 stage | 不支持的 dtype/dim 在修改缓存前报错；5 种非法 CLI 配置被拒绝；真实单 stage scheduler 搭配小模型完成 backward，检查开关传递和参数梯度有限 |
| GPU 内存访问 | 对实际 packed 路由的 8 项 MHA/GQA/MQA 用例运行 Compute Sanitizer memcheck，新扩展 kernel 过滤为 `kns=dspp_split_detail`，**0 errors**；不表示所有第三方 kernels 都经过本次检查 |

预设容限：拆分对融合 `atol=.004, rtol=.01`；多层模型对 FP64 `atol=.04, rtol=.04` 且逐 tensor 相对 L2 ≤ .03。多层测试包含 BF16 中间激活和梯度累加，因此与阶段 B 独立 attention 的 FP64 范数阈值不同；在首次运行前写入测试，并未按结果放宽。最终多层样例对融合最大绝对差为 0，初次测试出现过 `1.2207e-4`；保留两次日志，不将逐位一致当作确定性保证。

### 13.3 真实三卡训练对照

入口是实际 `pretrain_llama.py`，由 [validate_dspp_split_training.py](../tools/validate_dspp_split_training.py) 包装训练循环：在 optimizer 更新前保存全部模型参数梯度，保存各次 stage backward 返回的输入梯度，并记录实际拆分调用、混装任务与 state 排空。融合运行保存完整参考张量，拆分运行逐 tensor 对照，初始模型权重要求完全一致。

配置：3 张 L40，PP=3/VPP=2，TP=CP=DP=1；6 层、hidden=512、FFN=1024、Hq=4、d128、BF16、RoPE、RMSNorm、SwiGLU、dropout=0；合成训练序列长度 **193/129/33/17**、chunk=64、micro batch=4、global batch=8，seed=1234。沿用 `warmup-short-steady-long` 排序和 baseline packing。分别测试 MHA/Hkv=4、GQA/Hkv=2、MQA/Hkv=1，各自融合／拆分两组、每组 3 iteration。这里是变长、continuation 与短序列混装的正确性用例，不代表真实业务长度分布或长上下文性能规模。

下表为**首轮相同权重**下，跨三 rank、全部相关 tensor 的最大误差；绝对误差与相对 L2 的最大值可能来自不同 tensor：

| 结构 | 参数梯度最大绝对差 | 参数梯度最大相对 L2 | stage dx 最大绝对差 | stage dx 最大相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| MHA，Hkv=4 | `4.88281e-4` | 0.2308% | `6.10352e-5` | 0.6828% |
| GQA，Hkv=2 | `1.70985e-9` | `8.52649e-9`（比值） | 0 | 0 |
| MQA，Hkv=1 | `9.45874e-11` | `4.13639e-10`（比值） | 0 | 0 |

连续更新 3 轮后，MHA 参数梯度最大绝对差为 `0.00252695`、最大相对 L2 为 **1.3470%**；stage dx 最大绝对差仍为 `6.10352e-5`、最大相对 L2 为 **1.3809%**。GQA/MQA 的上述误差上界未增加。后两轮包含优化器轨迹的舍入差异，不能全部解释为同权重下单次 backward 差异。MHA 首轮 loss 完全相同，第三轮融合／拆分 loss 分别为 `10.0158539 / 10.0225849`；GQA/MQA 三轮 loss 均一致。全部通过首次运行前规定的 `atol=.004, rtol=.01` 与相对 L2 ≤ .02 检查，没有调整阈值。

每种结构共比较 **117 个参数梯度 tensor、165 个 stage dx tensor**（按 iteration/rank 计数，非独立参数数量）；记录 **198 次实际拆分 attention backward、42 次含历史的混装 backward、36 次 state 排空**。三种结构合计比较 351 个参数梯度和 495 个 stage dx。检查覆盖完整模型梯度及 stage 输入梯度，已超出阶段 B 的独立 attention；但 stage dx 的交付时机仍遵循原任务流程。

### 13.4 复现、归档与使用

使用 §9.4 的容器／Python 环境，固定扩展依赖已缓存时执行下列命令。`python` 应指向 `/workspace/src/venvs/megatron/bin/python`；本机目录 `/home/cuiym` 在容器内挂载为 `/workspace`。

```bash
export USER=cuiym
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/nvidia/lib64
export TORCHINDUCTOR_CACHE_DIR=/tmp/dspp-split-inductor
export TRITON_CACHE_DIR=/tmp/dspp-stage-c-triton
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export DSPP_FA_SPLIT_CACHE=/workspace/.cache/dspp-fa-split
python tools/build_dspp_fa_split.py
python -m pytest -q -s tests/unit_tests/dspp/test_packed_split.py \
  tests/unit_tests/dspp/test_fa_split.py tests/unit_tests/dspp/test_fa_split_regions.py \
  tests/unit_tests/dspp/test_packed_attention.py tests/unit_tests/dspp/test_kv_state.py \
  tests/unit_tests/dspp/test_training_runtime.py
compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=dspp_split_detail \
  python -m pytest -q tests/unit_tests/dspp/test_packed_split.py -k real_route_and_padding
DSPP_SPLIT_CHECK_ROOT=/tmp/dspp-stage-c-new \
DSPP_SPLIT_CHECK_SNAPSHOT_ROOT=/workspace/.cache/dspp-stage-c-new \
  bash examples/validate_dspp_split_training.sh
```

两处输出目录须尚不存在。脚本 [validate_dspp_split_training.sh](../examples/validate_dspp_split_training.sh) 包含数据生成、完整参数与六组运行；它只用于正确性，不用于计时。首次 PP 启动因 Triton 默认尝试写 `/.triton` 在训练前失败；归档保留失败日志，指定可写的 cache 路径后重跑全部对照，未改宿主驱动或原训练容器。

[validation.json](assets/fa-split/005-stage-c-20261016/validation.json) 保存逐项模块误差与 PP 汇总；`pp/` 保存每个 rank/iteration 的对照 JSON 与完整训练日志；[manifest.json](assets/fa-split/005-stage-c-20261016/manifest.json) 记录环境、命令、源码/证据/输入及外部参考张量的 SHA256，`source-snapshot.tar.gz` 保存对应源码。完整参考张量存于本机 `/home/cuiym/.cache/dspp-fa-split/stage-c-pp-20261016-r2/`，不复制进 Git。日志中的 iteration 时间受到张量转 CPU、保存与逐项比较的显著扰动，不能用于加速比或拆分开销结论。

已配置数据路径和运行环境后，原训练 launcher 可这样打开拆分：

```bash
HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=1024 NUM_ATTENTION_HEADS=4 NUM_QUERY_GROUPS=2 \
DSPP_SPLIT_ATTENTION_BACKWARD=1 bash examples/run_dspp_b23_l40.sh
```

该示例 head dim=128；原 launcher 默认 hidden=64、Hq=4，其 d16 不支持拆分。`NUM_QUERY_GROUPS=2` 会改变模型结构，融合对照必须使用同样配置，不能拿之前不同结构的日志作基线。这里完成的是 A/B/C 拆分计划；后续若研究 dx 提前交付，需要另设计历史任务状态生命周期、消费者依赖和调度，再评估端到端收益。

## 14. 完整 dx 边界后的立即历史任务（2026-10-16，已完成）

用户进一步明确：先交付完整 stage dx，随后立即执行该 physical microbatch 的历史 dKV；两者成为可独立计时的任务，不在二者间插入其他计算任务。本节补充 §13 尚未提供的任务边界，继续使用重计算方案，保存紧凑 backward 状态而非完整 P/dS。

实现合同：新增显式 after-dx 模式，保留融合及 attention 内立即拆分模式作对照。B 中计算每层完整 dQ、当前 dKV，消费已到达的后续 chunk 梯度，再完成 RoPE／投影及其余 stage 输入反向；各层历史计算状态交由当前物理任务持有。B 返回完整 dx，先交付本地边界或原通信队列，随即执行 H（历史 dKV 与缓存累加），完成后才允许下一计算任务。现有独立 W 仍沿用原 schedule。通信实际发起受原 credit 协议约束，dx 就绪/交付队列不等于对端已接收。

H 必须先于历史消费者完成，保存的 Q/K/V/O/dO/LSE/delta 在 H 完成前保持有效；禁止漏执行、重复执行及未完成 H 时开始下一 B 或结束 iteration。无历史时不启动历史 kernel。PP timeline 分别记录 B0/B1 与 H0/H1（H 即历史 dKV），计时为 CUDA event 计算流区间，包含任务内辅助操作；不将 attention 局部时间当作完整 B。

验收先检验 H 尚未执行时的完整多层 dx 与原融合一致、H 执行不再改变此 dx，再检验后续历史消费者、全部参数梯度、GQA/MQA、短尾混装及状态生命周期。最后运行真实三卡训练对照；计时使用单独无梯度张量保存/比较的训练运行，和正确性运行分开。

### 14.1 实现与状态生命周期

- 新开关 `--dspp-history-dkv-after-dx` 要求同时启用 `--dspp-split-attention-backward`。原单开关仍选择 §13 的 attention 内立即拆分；默认融合路径保持可用。
- `DsppSequenceKVState` 新增 `begin_backward_task → mark_dx_ready → run_history_task` 合同。每个 H 持有该次 physical microbatch 各层冻结的布局、Q/K/V/O/dO/LSE/delta；保存的 tensor 使用 detach 引用，保留数据和版本检查，不要求保留整个 autograd 图。prepare/dQ/current/H 沿用同一计算 stream。
- B 中仅路由当前 dKV 并消费后续 chunks 已产生的 pending 梯度，不把历史区域的零占位结果写入缓存。H 计算真正的历史 dKV 后只执行历史累加，不再次消费当前梯度、不再次弹出当前 segment。无历史时无 H kernel、无 H timeline 事件。
- `_DsppVTaskBatch.backward` 完成全 stage 输入反向后返回 dx；schedule 先设置本地边界或调用 `submit_send`，再独立调用 `task_batch.history`。H 必须完成后才推进下一个原 schedule 任务，W 继续单独执行。单 stage 路径也在完整 backward 返回后立即 drain H。
- 未 begin 就 backward、H 早于完整 dx、H 未执行又开始下一 B、重复 H、H 前改写状态 tensor、带未完成 H 结束／清空 state 均报错。失败任务不被悄悄清空或重试。
- timeline/profiler/progress 新增 H0/H1，0/1 指 VPP model chunk。CUDA-event 的 B 终点表示该输入梯度的计算流就绪边界，不等于接收端已经完成通信；原 credit/队列拥塞仍可能推迟实际发送。本轮没有改变通信协议，也没有运行旧 F/B/W 调度重放；旧重放须先增加 H 的依赖与资源模型才能使用这类 trace。

这是**重计算方案**：H 根据保存的紧凑状态再次生成所需 S/P/dP/dS，未把完整 P/dS 矩阵存到显存供 H 读取。当前区域和 H 的独立调用仍有准备／工作区／分派等成本，不能套用 FLOPs 的 40% 直接推断耗时。

### 14.2 正确性与任务边界证据

新增 [test_history_after_dx.py](../tests/unit_tests/dspp/test_history_after_dx.py)，扩展单 stage 运行测试，并让真实训练验证工具支持 after-dx 模式。

1. **H 前的完整 dx**：MHA/GQA/MQA，1/3 层 block，变长 145/70/17/3（chunk=64）和无历史 17/3；连续复用 state 两轮。测试在 H 之前克隆完整 block 输入梯度，与原融合及 §13 立即拆分分别比较。最终这些样例的 dx 最大绝对差为 **0**。spy 保证 B 中没有任何历史 dKV API 调用，不是将 H 算完后再改标签。
2. **H 的写入范围**：执行 H 前后当前 dx 和参数梯度逐位不变；历史状态保持 KV-head 维度，后续更早 chunk 反向及全部参数梯度仍通过对照。测试包含 pending 未排空／重复执行／错误 task id／修改保存 tensor 等负例。
3. **真实 PP=3/VPP=2**：沿用 §13.3 的 6 层、Hq=4、BF16/d128、193/129/33/17、chunk=64 配置。MHA/Hkv=4、GQA/Hkv=2、MQA/Hkv=1 各自原融合与 after-dx 两组、3 轮。每种结构比较 117 个参数梯度、165 个 stage dx tensor，执行 **114 个非空 H 任务**并排空 36 个 state；合计 342 个 H、351 个参数梯度与 495 个 stage dx 对照。
4. **回归与内存**：最终 **149 passed、0 skipped**，含 14 项新边界／错误配置测试、扩展单 stage 模式测试及 A/B/C、排序和 timeline 回归。新任务边界的 12 个多层 GPU 用例 memcheck **0 errors**（只过滤新扩展 kernels）。没有修改数值验收阈值。

三轮真实训练对原融合的误差上界如下；首轮参数完全相同，后两轮包含优化器轨迹舍入差异：

| 配置 | 首轮 stage dx 最大绝对差 | 三轮 stage dx 最大绝对差／相对 L2 | 三轮参数梯度最大绝对差／相对 L2 |
| --- | ---: | ---: | ---: |
| MHA，Hkv=4 | 0 | `3.05176e-5` / 0.5143% | `6.10359e-4` / 0.5056% |
| GQA，Hkv=2 | 0 | 0 / 0 | `3.41970e-9` / `1.70448e-8`（比值） |
| MQA，Hkv=1 | 0 | 0 / 0 | 0 / 0 |

预先规定的逐元素 `atol=.004, rtol=.01` 与相对 L2 ≤ .02 全部通过。初次测试有三项测试工具／展示问题：spy 把 varlen wrapper 及递归调用重复计数，CLI 负例缺必填 micro-batch-size，timeline 图例漏掉原回归要求的说明文字；修正后完整回归通过。失败日志一并保留，没有数值失败被放宽阈值掩盖。

### 14.3 真机独立 B/H 计时

使用 [measure_dspp_history_after_dx.sh](../examples/measure_dspp_history_after_dx.sh) 直接运行 `pretrain_llama.py`，**不加载正确性验证的 CPU 梯度保存／比较 hook**。分别运行融合、§13 attention 内立即拆分、完整 dx 后执行 H 三种模式，前两轮预热、第 3 轮采 CUDA-event timeline。变长合成数据源长度为 **6144/4097/1537/257**，chunk=2048；同一 seed、PP=3/VPP=2、6 层、hidden=512、FFN=1024、Hq=4/Hkv=2、BF16/d128，micro batch=4/global batch=8。各 iteration 由数据采样决定实际长度组合，并非每个 batch 恰好各取一条。

[summarize_dspp_history_tasks.py](../tools/summarize_dspp_history_tasks.py) 核对三种模式每个 stage/model-chunk/task 的实际布局一致。after-dx 测得 **60 个 B、24 个非空 H**；每个 H 都紧接对应 B，`B.end <= H.start`，下一个计算任务不早于 H 结束；无历史 B 没有 H。

以下是 **stage 1、VPP model chunk 0**、当前 2048 tokens／历史 4096 tokens（完整 KV=6144）的同形状任务。每种模式在该测量 iteration 中有 **2 个任务样本**，取中位数，单位 ms：

| 模式 | B：至完整 stage dx | 独立 H | 备注 |
| --- | ---: | ---: | --- |
| 原融合 | 8.198 | 无 | 历史计算包含在 B 内 |
| §13 attention 内立即拆分 | 9.883 | 无 | 历史计算仍包含在 B 内 |
| 完整 dx 后立即执行 H | **8.710** | **0.940** | B 与 H 已分成两个独立任务 |

此配置一个 VPP model chunk 含一个 Transformer block；这里 B 包含原框架交付该 stage 输入梯度所需的 MLP、归一化、残差、投影、RoPE、dQ 与当前 dKV 等计算，**不是仅 attention**。原框架已分离的 W 不计入 B。H 包含该任务各层历史 dKV 的辅助操作、GQA 归约和缓存路由；不是只报主 kernel 时间。

按每对事件先相加再取中位数，B+H 为 **9.650 ms**；从 B 开始到 H 结束的整个区间为 **10.062 ms**，差额包含二者之间的通信提交和主机提交间隔。CUDA-event 是计算流区间，任务内也可能包含主机提交不及时导致的空隙及通信资源竞争，不能视为纯 kernel 吞吐。

该样例将 H 从 dx 前的依赖链中移除了，但 B 的 8.710 ms 仍大于原融合的 8.198 ms，**没有证明相对原融合的 dx 耗时优势或 iteration 加速**。这只是一个测量 iteration、每组两次同形状任务样本；尚无充分重复统计。完整每任务数据、样本量和其他 shape/stage 结果见 [task-times.json](assets/fa-split/006-after-dx-20261016/measurement/task-times.json)，[真实 FBWH 时间线](assets/fa-split/006-after-dx-20261016/measurement/after_dx/timeline/iteration_000003.svg) 展示全部 stage 事件。

### 14.4 复现与归档

沿用 §13.4 环境变量及 Python 路径，启用方式为：

```bash
HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=1024 NUM_ATTENTION_HEADS=4 NUM_QUERY_GROUPS=2 \
DSPP_SPLIT_ATTENTION_BACKWARD=1 DSPP_HISTORY_DKV_AFTER_DX=1 \
  bash examples/run_dspp_b23_l40.sh
```

独立正确性与任务计时的复现入口（输出目录必须尚不存在）：

```bash
DSPP_SPLIT_CHECK_AFTER_DX=1 \
DSPP_SPLIT_CHECK_ROOT=/tmp/after-dx-correctness-new \
DSPP_SPLIT_CHECK_SNAPSHOT_ROOT=/workspace/.cache/after-dx-snapshots-new \
  bash examples/validate_dspp_split_training.sh
DSPP_AFTER_DX_MEASURE_ROOT=/tmp/after-dx-measure-new \
DSPP_AFTER_DX_DATA_ROOT=/workspace/.cache/after-dx-data-new \
  bash examples/measure_dspp_history_after_dx.sh
python tools/summarize_dspp_history_tasks.py --root /tmp/after-dx-measure-new
```

材料集中在 [006-after-dx-20261016](assets/fa-split/006-after-dx-20261016/)：`validation.json` 保存具体误差与 PP 汇总，`pp/` 保存正确性日志，`measurement/` 保存三种模式的真实训练日志和 timeline；manifest 与源码快照记录可复现版本。正确性参考张量和测量输入分别在本机 `/home/cuiym/.cache/dspp-fa-split/after-dx-pp-20261016/` 与 `/home/cuiym/.cache/dspp-fa-split/after-dx-measure-20261016/`，不复制入 Git。前面的 001–005 归档保持原样，代码未提交／推送。

## 15. B/H 计时路径清理与区域缓冲精简（2026-10-16，已完成）

用户要求核查并去除 B/H 内部的计时、诊断及多余正确性检查，只保留任务边界计时。核查 006 日志：timing_log_level=0、kaimm_cuda_synchronize_level=0、torch profiler/性能模型关闭，因此没有启用 B 内的同步 Megatron timer，不能归因于它；但 NVTX 包装仍插入 autograd 节点，每个 B 还执行两次纯标注用的小 backward。

本次增加显式纯任务计时模式：关闭内部标注和无效同步包装，预建/初始化 B/H 边界 CUDA events，GPU 完成后才生成 metadata/JSON，禁止同时开内部 profiler/progress。保留任务所有权、历史消费者和输入内存安全校验，去掉 varlen 外层与子序列重复的版本检查、单序列拼接复制；当前/历史生产输出改为只返回所选区域，保留完整形状 API 作对照。GQA 的每 Q-head 部分梯度工作区暂不改变，以避免同时改变非对齐 tile 地址和归约合同。

先运行 GPU 正确性回归和选定 memcheck，再做融合、attention 内立即拆分、after-dx 的独立重复 B/H 测量。输入采样、初始参数及数据顺序固定；测量运行 lr=0，避免不同优化器轨迹改变模型权重。只报告边界事件区间，不把此次运行日志当作无测量开销的 iteration 吞吐实验。

### 15.1 核查结果与改动范围

| 项目 | 006 的实际状态 | 本轮处理 |
| --- | --- | --- |
| `backward-compute` 的同步 Timer | timing_log_level=0，训练入口传入 timers=None，未启用 | 纯任务计时模式显式要求 level=0，不能声称此前慢是它造成的 |
| CUDA 调试同步 | synchronize_level=0，未执行同步；但仍可能插入空操作 autograd 节点 | 关闭同步时直接返回输入，不插入该节点 |
| NVTX 前后向包装 | 即使没有 profiler 也插入包装节点；每个 B 的 annotate_backward_range 另执行两次小 backward | `--dspp-task-timing-only` 关闭这些内部标注与节点 |
| B/H 中的数值正确性对照 | 正式计时没有加载验证脚本，原本没有梯度逐项对照／CPU 保存 | 继续与正确性运行分离，独立审计验证无 GPU 数值回读／对照 |
| 边界事件及输出 | 在 task 调用处创建 event 对象，每任务后生成 metadata | 测量前预建并初始化事件；B/H 内只记录开始与结束；最后统一同步、metadata 转换和落盘 |
| varlen 状态校验 | 外层与子序列对共享 Q/K/V/O/dO 重复版本检查 | 共享 tensor 由子状态检查；外层只检查原始 LSE 与 metadata，保留必要的流与修改检查 |
| 区域输出 | 当前／历史都返回完整 KV 长度、区域外清零 | after-dx 生产路径返回所选区域的紧凑输出，路由分别按 current/history 索引 |
| 单序列拼接 | 即使只有一个输出仍调用 torch.cat | 单序列直接返回；compact H 过滤零长度项，避免无意义拼接 |

`--dspp-task-timing-only` 禁止同时启用内部 profiler、GPU progress、调试同步与细粒度同步 timer；默认训练及其他 profile 工作流仍可选择原标注行为。B/H 时长必须在三种对照模式下使用相同测量设置。

这里没有删除 native 入口的 shape/device/bounds 校验，没有移除防止重复 H、历史消费者尚未就绪等协议检查。它们读取 CPU 元数据，不运行 GPU 正确性对照或强制同步。原始 LSE 可能被复制为连续的子序列 LSE，因此仍需单独保留其版本校验；最终新增 tuple metadata 下的相关回归。

紧凑输出并不是将未初始化的完整缓冲交给旧路由。B 的路由只读取 current 行，H 的路由只读取 history 行；混装中无历史的独立短序列没有 H 行。GQA 的 per-Q-head 中间工作区和 FP32 归约、BF16 舍入合同保持不变。该工作区仍按完整 KV 地址分配（但不整体清零），后续若缩小它，需要进一步修改 tile 写回地址并处理非对齐边界；本次没有冒险同时改变这一部分。

早期完整形状输出是为了复用现有梯度路由、先验收正确性，不是算法必需，也不意味着很难优化。本轮已完成紧凑最终输出和单序列路径的精简；还没有把完整 dQ 与 current dKV 融合成一次执行，当前区域的重复重计算仍存在。

### 15.2 正确性与独立审计

- 最终回归 **162 passed、0 skipped**；覆盖原 A/B/C、H 前完整 dx、MHA/GQA/MQA、varlen、current/history 紧凑输出逐元素对照、关闭标注后的梯度、预初始化边界事件、metadata 延后输出及复制 LSE 的版本检查。
- 20 项紧凑区域／H 边界用例通过 Compute Sanitizer memcheck，过滤新扩展 kernels，**0 errors**。
- 在关闭内部标注的真实 PP=3/VPP=2 路径上，MHA/GQA/MQA 各自融合／after-dx 跑 3 轮，比较 351 个参数梯度与 495 个 stage dx，全部通过原有 atol=.004、rtol=.01、相对 L2≤.02 容限。MHA 三轮参数梯度／dx 最大相对 L2 为 1.2732%／1.2300%；GQA 为 0.4746%／0.5541%；MQA dx 逐位一致。后两轮包含优化器轨迹差异；这些正确性运行不用于计时。
- [audit_dspp_task_timing.py](../tools/audit_dspp_task_timing.py) 单独运行真实三卡训练，在 B/H 主体内拦截 Python CUDA event.record/synchronize、全局同步、Timer.start/stop、NVTX、CUDA tensor 的 cpu/item/tolist 和显式数值对照。选定 iteration 共审计 **60 个 B、24 个 H，0 次违规**；并禁止插入仅用于标注／同步的 autograd 节点。审计只覆盖相应 Python 入口，不假装是对所有第三方 C++ 的动态跟踪；本仓库 native 拆分入口仍保留 CPU 元数据检查及非同步 launch error check。
- 正式测量使用原 `pretrain_llama.py`，**不加载上述审计 hook，也不加载训练正确性 hook**。纯 B/H trace 不含 F/W，所以不输出供排序使用的完整任务成本文件，也不把未记录的 F/W 间隔误称为 pipeline bubble。

### 15.3 解释“5→3”与验收标准

“历史区 5→3”指移到 H 之后，B 在历史区域少做 dK/dV 的两项主要矩阵乘。这给出了 B 变短的合理预期；它不是实际时间直接乘 3/5 的依据。当前实现还把 current 区域的主要矩阵乘由 5 变成 7，因为 dQ 与 current dKV 分别重计算其共享中间量。

对于 Q=2048、history=4096，按有效 causal 位置、忽略边界 tile 冗余估算，B 的 attention 主要矩阵乘 FLOPs 仍减少约 24%。因此不能只用“重计算变多”解释 B 变慢，更不能把尚未定位的差额都归于测量。主机分派不及时形成的计算流空隙也是当前实现的实际执行成本，不应当自动当作可忽略的测量误差。

本任务的性能验收必须比较相同输入、相同权重、相同测量方式下的完整 B。H 已在 B 之后，只说明拆分边界成立；若相关关键 B 的 dx 没有早于原融合就绪，当前实现就没有达到这条收益路径所需要的前提，不能据此宣传 iteration 加速。一个 shape 的结果也不能代替所有负载／关键路径的结论。

### 15.4 固定版本的最终重复测量

[measure_dspp_clean_tasks.sh](../examples/measure_dspp_clean_tasks.sh) 使用 §14 的长 chunk 变长数据源、模型形状和排序。每种模式独立启动 **3 次**，运行顺序轮换为 fused/immediate/after-dx、after-dx/fused/immediate、immediate/after-dx/fused；每次两轮预热，第三轮只采 B/H。所有模式均关闭内部标注，lr=min-lr=0，使权重保持初始值，seed 和数据顺序相同。三轮各模式对应 task 的 layout 一致，训练日志的三轮 loss 序列一致（日志输出精度下）。不加载审计或正确性验证 hook。

`measurement/` 是开发期间的预采样，期间完善了原始 LSE 元数据校验，因此**不作为最终统计**。以下只使用 `measurement-final/`：最终测量开始前保存源文件 SHA256，结束后逐文件确认一致，全部九组运行使用同一版本。数据生成脚本相同，每模式有两条同形状任务，共 6 个样本（2 个对应任务 × 3 次独立启动），不是跨不同 stage 的平均。

stage 1、VPP model chunk 0，Q=2048、history=4096、KV=6144，Hq=4/Hkv=2、d128、BF16。B 是对应一个 Transformer block 的完整 stage 输入反向，原独立 W 不包含在内。单位 ms：

| 模式 | B 中位数 | B 最小–最大 | H 中位数 | H 最小–最大 |
| --- | ---: | ---: | ---: | ---: |
| 融合 | **5.839** | 5.223–5.962 | 无 | 无 |
| attention 内立即拆分 | 6.488 | 5.858–6.613 | 无 | 无 |
| after-dx，紧凑区域输出 | **6.462** | 6.283–6.568 | **0.654** | 0.629–0.659 |

after-dx 的 B 比融合慢 **10.66%（约 0.623 ms）**。按每次启动分别取两个同形状任务的中位数，融合／after-dx 依次为 **5.547/6.337、5.797/6.484、5.839/6.525 ms**，三次方向一致。相同形状在六个 stage/model-chunk 组合中的 B 中位数均慢于融合，差幅约 2.76%–10.66%；这仍只覆盖本轮配置，不能外推所有模型／负载。

因此，本轮完成了测量路径和区域缓冲清理，但**当前实现仍未实现相对融合更快交付 dx，不能据此声称 iteration 收益**。结果不能再归因为一个未启用的内部 timer，也不能用“历史区 5→3”替代测量。现存 dQ/current 分别执行、当前区域重复计算、额外启动与辅助操作及计算流提交开销，仍需后续独立归因／优化；本轮没有在 B 内添加细分 profiler 来给这些因素分摊时间。

H 在 B 返回并提交 dx 后由原 schedule 紧接调用；纯 B/H 事件验证对应 H 晚于 B，但不将未采 F/W 的 trace 冒充完整 FBWH 观测。GPU 上因主机提交不及时形成的间隔和通信资源竞争仍计入真实任务区间，它们属于当前执行环境／实现成本，不等同于计时器导致的测量污染。此次仅报告任务时间，未把带边界 events 的日志当作无测量开销的端到端吞吐结论。

[combined-task-times.json](assets/fa-split/007-clean-task-timing-20261016/measurement-final/combined-task-times.json) 保存所有 shape/stage 的样本、范围、分轮结果；[summarize_dspp_clean_tasks.py](../tools/summarize_dspp_clean_tasks.py) 提供聚合入口。正确性结果、独立审计、预采样和最终计时均在 [007-clean-task-timing-20261016](assets/fa-split/007-clean-task-timing-20261016/) 分目录保留。

### 15.5 使用与复现

在 §13.4 的相同环境变量下，给原 launcher 增加 `DSPP_TASK_TIMING_ONLY=1` 即可关闭内部诊断，只在选定 iteration 记录 B/H；after-dx 仍同时设置前述两个拆分开关。计时与数值对照应使用不同运行：

```bash
# 真实三卡数值对照；不用于报告耗时。
DSPP_SPLIT_CHECK_TASK_TIMING_ONLY=1 DSPP_SPLIT_CHECK_AFTER_DX=1 \
DSPP_SPLIT_CHECK_ROOT=/tmp/clean-task-correctness-new \
DSPP_SPLIT_CHECK_SNAPSHOT_ROOT=/workspace/.cache/clean-task-snapshots-new \
  bash examples/validate_dspp_split_training.sh

# 纯边界计时，共三次独立启动/模式，输出目录须尚不存在。
DSPP_CLEAN_MEASURE_ROOT=/tmp/clean-task-measure-new \
DSPP_CLEAN_DATA_ROOT=/workspace/.cache/clean-task-data-new \
  bash examples/measure_dspp_clean_tasks.sh
python tools/summarize_dspp_clean_tasks.py --root /tmp/clean-task-measure-new
```

最终测量源文件校验见 `final-measure-source-hashes.json`，其哈希与本轮 manifest 交叉记录。native 扩展缓存键为 `6b7a77a01a54ad92`，仍基于原固定 FA2/CUTLASS 源码，未替换环境中的默认 FA。完整张量参考和输入数据留在本机 `.cache/dspp-fa-split/clean-timing-pp-20261016/`、`clean-timing-measure-final-20261016/` 等目录，不加入代码仓库。001–006 的原始证据未改，代码未提交或推送。

## 16. B 变慢的原因排查（2026-10-16）

本轮按 kernel 执行结构、当前区域重复执行、Python 分派及辅助操作的顺序核查。生产路径与 007 最终测量的源文件 SHA256 一致；没有重跑完整 B/H 的重复统计，没有修改生产 kernel 或 schedule。诊断独立运行，正式计时仍只保留任务边界。

### 16.1 理论计算范围

对 stage 1/VPP chunk 0 的单块 Q=2048、history=4096、Hq=4/Hkv=2、d=128、hidden=512、SwiGLU FFN=1024，令历史有效位置数 A_h=2048×4096，当前 causal 位置数 A_c=2048×2049/2。一次 multiply-add 计 2 FLOPs：

- 原 attention backward 的主要矩阵乘为 10×512×(A_h+A_c)=53.69 GFLOPs。
- QKV/输出投影及 MLP 输入反向为 2×2048×[512×(512+2×256)+512²+3×512×1024]=9.66 GFLOPs。
- 原 B 主要矩阵乘合计 63.36 GFLOPs；历史 dK/dV 两项为 4×512×A_h=17.18 GFLOPs，占 27.1%。
- 当前原型在当前区重复计算 4×512×A_c=4.30 GFLOPs，故 B 净减少约 20.3%，保留 50.47 GFLOPs。
- H 因重计算需要 8×512×A_h=34.36 GFLOPs，不能把它的独立耗时当成原融合 B 中历史 dKV 的耗时。

以上不包含独立 W，也不包含 softmax、归一化、激活、访存与提交成本；不是完整 B 的时间占比。007 的 H/(B+H) 中位数比值约 9.2%，同样不是原融合历史 dKV 的时间占比。

### 16.2 首轮诊断与证据

旧 004 trace 的 Hq=16，不能直接解释当前 Hq=4。新增 [008-b-cost-diagnosis-20261016](assets/fa-split/008-b-cost-diagnosis-20261016/)：

- `training/`：融合和 after-dx 各一次三卡训练，前两轮预热，第三轮只在 rank 1 开独立 profiler。入口为 [diagnose_dspp_training_cost.py](../tools/diagnose_dspp_training_cost.py)，生产计时运行不会加载它。
- `attention-final/`：匹配 Q2048/K6144/GQA4:2 的独立 attention，外层 CUDA events 与 profiler 分开运行；比较 eager、直接 native 调用以及相同计算的 CUDA Graph replay。
- `combined-probe/`：增加一个诊断消融，仅将当前区域的 dQ/dKV 合并进同一次扫描；历史区只算 dQ。共享内存配置不变。代码位于 `tools/dspp_cost_probe.cu`，不被训练路径导入。
- 每种独立计时默认只取 3 个样本；当前结果已足以决定下一步，不为确认同一方向继续重复。CUDA Graph 是消除逐次主机提交的诊断手段，不代表 PP 已支持 Graph。

**第一项：少做矩阵乘尚未转化为更快的 attention B。** 真实训练中选定位置的两条同形状任务，融合主 kernel 分别为 391.876/390.116 μs；拆分 dQ 为 283.939/284.098 μs，current dKV 为 124.385/124.801 μs，两者合计 408.324/408.899 μs。即 dQ 已缩短，但再加当前 dKV 后反而更慢。

三个主 kernel 的共享内存申请均为 81920 bytes；寄存器/thread 为融合 255、dQ 166、dKV 224。当前 grid 为 dQ [96,1,4]、current [32,1,4]，每 block 256 threads。寄存器减少不等于共享内存或同步开销随之减少。尚未分别消融共享内存布局、搬运和 barrier，不能把它们某一项宣布为已证明的主要原因，也不使用 profiler 返回的异常 occupancy=0 字段推断实际占用率。

**第二项：当前区域重复执行确实损失了收益。** 同一轮诊断消融的外层事件中位数如下，单位 μs，均只包括 attention 的相应 backward 计算和辅助操作，不是完整 stage B：

| 路径 | eager 调用 | CUDA Graph replay |
| --- | ---: | ---: |
| 原融合 attention backward | 546.816 | 407.552 |
| 当前生产拆分 API：prepare+dQ+current | 624.640 | 424.960 |
| 同样拆分的直接 native 调用 | 462.848 | 425.984 |
| 诊断合并扫描的 native 调用 | 435.200 | 386.048 |

合并对照的 Graph 比原拆分 native 约快 9.4%，比融合约快 5.3%；它同时消除了当前区重复扫描和一次主 kernel 启动，不能把全部差值归给纯 FLOPs。诊断主 kernel 约 361.826 μs，仍申请 80 KiB 共享内存。只在本次单序列、64 对齐边界、GQA4:2/d128/BF16 的输入上检查：相对融合 dQ 的相对 L2≈2.58e-5，current dK/dV 逐位一致；沿用 atol=.004/rtol=.01，没有放宽容限。这不是覆盖 varlen/短尾/MHA/MQA 的生产实现验收。

**第三项：Python 状态封装和分派成本有明确证据。** 原拆分 API 与直接 native 路径执行相同 GPU 操作，Graph 耗时约相同（425/426 μs），eager 则为 625/463 μs。首轮独立诊断也得到相同方向：Graph 422/424 μs，eager 549/447 μs。不能跨轮直接相减，因为 eager 提交开销有变化；但两轮都显示封装、逐次提交的额外成本。native 消融绕过 Python 状态对象及重复校验，不代表生产可以删除所有生命周期检查。

训练 profiler 进一步定位到 attention 调用：选定 B 的 GPU 事件数由 37 变成 40，kernel 时间合计的中位数由 611.846 变为 623.575 μs；CPU B 区间由 9768.372 变为 10957.935 μs，其中 attention backward CPU 区间由 1523.608 变为 2699.461 μs。诊断中的增加主要出现在 attention 主机调用路径。**profiler 已明显扩大 CPU 区间，不能把这些数字作为正式 B 时间，也不能据此精确分摊 007 的 0.623 ms 回退。** 所选 B 的 GPU 事件均在同一计算流，未归入 NCCL kernel；并不由此排除并发通信的资源竞争。

原始 trace 的事件通过 CUDA launch correlation 关联；完整任务范围允许覆盖同步 autograd 调用的工作线程，嵌套算子按同一线程关联。嵌套统计有重叠，不相加。解析工具为 [summarize_dspp_cost_diagnosis.py](../tools/summarize_dspp_cost_diagnosis.py)，汇总见 [diagnosis.json](assets/fa-split/008-b-cost-diagnosis-20261016/diagnosis.json)。

### 16.3 下一步与停止条件

三个猜测的验证状态（008 诊断完成时；后续结果另记 §17）：

| 原先的猜测 | 验证程度 | 尚未回答的问题 |
| --- | --- | --- |
| 删掉矩阵乘后，仍保留较重的共享内存、搬运和同步结构 | 部分核实：确认这些结构仍存在，尚未逐项消融 | 各自造成多少回退 |
| dQ 与当前 dKV 分开执行，造成重复计算和额外启动 | 有消融证据：合并后局部耗时下降 | 重复计算、访存和启动各占多少；完整 B 是否变短 |
| Python 状态封装与分派增加成本 | 有对照证据：相同 GPU 操作，绕过封装后 eager 变快，Graph 耗时基本一致 | 对完整 B 的影响尚未精确量化 |

首轮已把原因从泛泛的“拆分有开销”缩小到两项有对照证据的问题：GPU 上当前区域的重复执行，以及主机状态封装/分派。共享内存、搬运和同步结构仍是优化候选，尚未分别量化。

下一步优先把合并扫描扩展为满足现有区域/varlen/GQA 合同的实现，并合并 prepare/执行入口以减少重复封装；必要的消费者与生命周期检查应集中执行，而非直接移除。先验收正确性，再做一组相同设置的融合/拆分完整 B/H 对照；仅在差异接近波动、方向反复或最终报告时增加重复。现在不继续调度搜索和重放。

仍以完整 stage B 为性能门槛：诊断中的约 5% attention 优势尚不能证明完整 B 提前，更不能证明 iteration 加速。生产版本的性能结论仍是 §15 的 B 慢 10.66%。初次诊断曾因默认流不能用于 CUDA Graph capture 退出，修正为整个独立诊断使用同一非默认流；失败日志保留，未混入统计。

## 17. 合并 dQ/current dKV 接入与完整 B 验证（2026-10-16，已完成）

先明确实现合同：一次扫描所有合法 KV；完全落在历史区的 tile 只计算 dQ，与当前区相交的 tile 同时计算 dQ/dK/dV。非对齐边界 tile 可能算出少量历史 dKV 临时值，但 B 只返回 current 行，历史贡献仍由 H 独立计算并累加，不提前路由、不重复加。输出 dQ 是全部合法 KV 的贡献，current dKV 是紧凑 KV-head 布局；保留原 FP32 累加/BF16 舍入合同。

本轮先复用原 prepare 状态，将 after-dx 中独立 dQ/current 调用替换为合并入口，减少一次主 kernel 和一次 Python 状态检查/分派；不同时重写 prepare/varlen 调度、H、共享内存布局或计时边界，以便解释变化。原独立 dQ/dKV API 和立即拆分路径保留作对照。

验收顺序：MHA/GQA/MQA、非对齐与短尾、packed 序列隔离、mutation/stream 合同；多层 block 的 H 前完整 dx 与历史消费者；真实 PP=3/VPP=2 训练梯度对照；最后一组同权重/同数据的融合与新 after-dx 纯边界 B/H 计时。必要时增加一次反序测量，避免结果接近波动时过早下结论。原始材料归档 009，profiler/正确性 hook 不进入正式计时。

### 17.1 实现与正确性结果

- `fa_split_src/backward.cu::dq_current_kernel` 按 KV tile 是否与当前区相交选择 dQ-only 或 dQ+dKV；`split_dq_current` 返回完整 dQ 与紧凑 current dK/dV。未修改 FA2 上游 patch、共享内存布局、精度合同或 H。
- `fa_split.py::backward_dq_current` 保留 stream/version 检查，逐序列处理 varlen，按各自序列边界拼接输出。`dspp_packed_attention.py` 的 after-dx 路径使用它；历史缓存路由与 H 调用位置不变。原独立接口和 attention 内立即拆分仍可用于对照。
- **212 passed、0 skipped**。新增直接对照覆盖无历史、单 token、63/64/65 附近的历史边界、短尾、MHA/GQA/MQA、packed 隔离、输入修改与跨流拒绝。小形状对照融合和 FP64；长形状 Q2048/K6144 对照融合。另检查一次合并调用只发射一个主 kernel。
- 小形状相对融合最大相对 L2≈2.07e-6；相对 FP64 最大相对 L2≈0.003583。长形状 dQ 相对融合最大相对 L2≈2.62e-5，current dK/dV 逐位一致。没有放宽容限。
- 1/3 层 block、chunk_size=63/64、MHA/GQA/MQA 的两轮反向：H 之前的完整 dx 与融合及立即拆分参考相符，H 后当前 dx/参数梯度不变，历史消费者收到正确累加。27 项直接边界/packed 用例通过 Compute Sanitizer memcheck，**0 errors**。
- 真实 PP=3/VPP=2，MHA/GQA/MQA 各自融合/after-dx 跑 3 轮，共比较 **351 个参数梯度、495 个 stage dx**，全部通过 atol=.004、rtol=.01、相对 L2≤.02。每种结构 114 个非空 H、36 个已清空状态。跨三轮（含优化器轨迹差异）的最大参数梯度/dx 相对 L2：MHA 1.2263%/1.1856%，GQA 0.7151%/0.6744%，MQA 参数梯度约8.20e-10、dx逐位一致。

数值记录见 [unit-results.json](assets/fa-split/009-combined-current-20261016/unit-results.json) 和 [validation.json](assets/fa-split/009-combined-current-20261016/validation.json)。以上运行不用于报告耗时。

### 17.2 完整 B/H 计时与结论

使用 [measure_dspp_combined_tasks.sh](../examples/measure_dspp_combined_tasks.sh)，保持 007/008 的 3×L40、PP3/VPP2、CP=TP=DP=1、6层/hidden512/FFN1024、GQA4:2/d128/BF16、变长数据源6144/4097/1537/257、chunk2048及原排序。lr=min-lr=0，固定 seed/数据顺序，原训练入口不加载 profiler 或正确性 hook，只在第三轮采 B/H 边界。计时前后 SHA256 一致，四次启动三轮 loss 序列均相同（日志精度下）。

先跑融合→合并后 after-dx 一组；首组配对样本方向不一致且差异接近波动，因此只补一组 after-dx→融合，不继续扩大重复次数。两组各有 60 个 B、24 个 H，layout 一致，所有非空 H 在匹配 B 之后；无历史任务没有 H。这里只测任务区间，不报告 iteration 加速。

选定 **stage 1/VPP model chunk 0、Q2048/history4096**，每组每模式 2 个对应任务。B 是完整 block/stage 输入反向，独立 W 不在内，单位 ms：

| 运行顺序 | 融合 B 中位数 | 合并后 B 中位数 | H 中位数 |
| --- | ---: | ---: | ---: |
| 融合→合并后 | 5.856 | 6.001 | 0.612 |
| 合并后→融合 | 5.856 | 6.154 | 0.667 |
| 两组合计（每模式4个样本） | **5.856** | **6.154** | **0.645** |

合计中位数比较，B 仍慢约 **5.09%（0.298 ms）**。融合范围5.834–5.878 ms，合并后范围5.698–6.305 ms；四个配对任务中三个更慢。B+H 的配对求和中位数为6.821 ms。两组内的中位数方向一致，但样本数量有限，不将此范围外推到所有模型/负载。

**本轮正确性与接入完成，完整 B 提前的性能验证未通过。** 008 的约5.3%是独立 attention Graph 诊断收益，不能代替这里的完整 B；007 的6.462 ms来自另一轮未合并版本，不能用跨轮差值宣称本轮合并带来的因果收益。本轮仅替换了 dQ/current 入口，prepare 状态创建、逐序列 Python 分派、H 仍保留。

下一步最值得继续核查的是 prepare/状态封装和分派在完整 B 中的成本，以及必要检查能否集中执行；共享内存/访存/barrier 仍未逐项消融。现阶段不启动调度搜索或将局部收益当作端到端可行性结论。

所有样本、范围和各 stage/layout 结果见 [combined-task-times.json](assets/fa-split/009-combined-current-20261016/combined-task-times.json)。复现设置与源码快照见 009 的 manifest；native 缓存键 `51ac35cd4a46e186`。完整正确性张量保存在仓库外 `.cache/dspp-fa-split/combined-current-pp-20261016/`，原 001–008 材料未覆盖，代码未提交或推送。

## 18. 关闭 GQA 的完整 B/H 对照与工作规模诊断（2026-10-16）

本轮复用 009 的生产计算与计时代码，不修改 kernel 或调度。在测量 launcher 增加 `DSPP_COMBINED_QUERY_GROUPS`，默认仍为2；设为4时省略 GQA 开关，真正使用 `group_query_attention=False`。日志中的默认 `num_query_groups=1` 此时不生效，`transformer.py` 按 Q heads 设置实际 KV heads；本轮 MHA 为 Hq=Hkv=4。

### 18.1 MHA 实测

保持 hidden512/FFN1024、6层、PP3/VPP2、CP=TP=DP=1、相同变长数据源/chunk2048、固定权重与排序；只把 KV heads 从2改为4。融合与拆分各启动一次，前两轮预热、第三轮仅采 B/H 边界，不加载 profiler/梯度对照 hook。每模式在 stage 1/VPP chunk 0、Q2048/history4096 有两个对应样本，单位 ms：

| 模式 | B 中位数 | B 范围 | H 中位数 | B+H 配对求和中位数 |
| --- | ---: | ---: | ---: | ---: |
| MHA 融合 | **5.765** | 5.650–5.879 | 无 | 5.765 |
| MHA 合并 dQ/current、H 在 B 后 | **6.369** | 6.192–6.545 | **0.648** | **7.016** |

B 仍慢约 **10.48%**，两个配对任务方向一致且两组范围分离，因此未增加重复次数。这是有限样本的当前形状结果，不能外推所有 MHA 负载。所有60个B/24个H边界、layout 对齐检查通过；两模式对应三轮 loss 在日志精度下一致。关闭 GQA 未消除完整 B 回退。

009 的 GQA 当前版本融合/拆分 B 为5.856/6.154 ms，007 的 GQA 未合并版本为5.839/6.462 ms。数值上回退从10.66%缩至5.09%，拆分 B 下降约4.8%，但两轮不同，不能把跨轮差值作为严格的合并消融收益。008 独立诊断中的合并收益有同轮对照证据；完整 B 尚未获得相对融合的稳定收益。GQA 与 MHA 是不同模型结构和运行，也不通过跨行绝对时间宣称 GQA 的性能收益。

### 18.2 为什么局部更快但完整 B 仍慢

此前“attention 更快”特指独立 kernel/Graph 诊断，并未证明带生产 Python 包装的完整 attention 调用更快。本轮使用已经接入的 `backward_dq_current`，独立 attention 固定 Q2048/K6144/d128/BF16，将 heads 从4/4增加到16/16；每项仅3个外层事件样本，profiler 另跑，单位 μs：

| MHA heads | 普通调用：融合 | 普通调用：合并 B 部分 | Graph：融合 | Graph：合并 B 部分 |
| --- | ---: | ---: | ---: | ---: |
| 4/4 | 475.136 | 516.096 | 394.240 | 384.000 |
| 16/16 | 1517.568 | 1312.768 | 1500.160 | 1251.328 |

4 heads 下 Graph 约快2.6%，但普通 API 约慢8.6%。同样4 heads 的直接 native 合并调用为414.720 μs，Graph为384.000 μs；生产 Python API 为516.096/384.000 μs。因而无需假设“其他反向部分变慢”：局部GPU节省本身就可能被状态封装、分派及辅助操作抵消。Graph 与 eager 的差值包含执行方式/缓存等影响，不能直接当作精确的纯 Python CPU 时间。

16 heads 下普通调用约快13.5%，Graph约快16.6%，说明工作规模/并行度确实影响收益。这仅是 attention 诊断，不是16-head完整模型训练结果。两形状的 dQ/current dKV 均通过原有融合对照容限，current dK/dV逐位一致，dQ最大相对L2约2.46e-5。

### 18.3 GPU 是否不饱和，以及后续优先级

L40 实测142个SM，每SM102400 bytes共享内存、65536寄存器、最多1536 threads。当前主 kernel 的grid从[96,1,4]变为[96,1,16]，即384→1536个CTA；每CTA256 threads、80 KiB共享内存。共享内存已将同时驻留上限限制为每SM一个CTA，且寄存器也较重。这个静态限制不等于实际GPU利用率或Tensor Core利用率；本轮未采集实际occupancy、Tensor Core利用率、带宽和stall计数，不能宣称已证明GPU未饱和是唯一原因。

需要区分：主机提交间隙导致GPU等待，和kernel内部并行度/资源利用率不足。Graph/eager对照支持前者，head规模对照支持工作规模与收益有关；仍需硬件计数器区分后者的具体瓶颈。4-head的384个CTA已多于142个SM，不能简单说“任务数不足以覆盖SM”。此外GQA4:2→MHA4:4并没有增加Q heads或主kernel的CTA数量；它主要改变KV投影/存储和梯度归约，不能替代增加并行工作量的实验。

后续建议：

1. 当前区重复扫描已消除，优先精简/合并 prepare、状态封装与native调用，保留必要生命周期检查；不重复验证已解决的重复扫描问题。
2. 用更接近目标模型的hidden/head/FFN及每stage层数验证完整B。增大模型也会改变MLP占比，不能把16-head attention收益直接推成完整B收益。
3. 无历史任务没有H可移出，应评估直接使用原融合backward的选择性路径，避免额外拆分成本；多序列packing还可评估native varlen批量发射，减少逐序列调用。
4. 对共享内存、寄存器、访存/barrier做有硬件计数器支持的定向优化；迄今尚未完成这些因素的单独消融。

结果、原始样本、kernel属性和限制见 [010/summary.json](assets/fa-split/010-mha-comparison-20261016/summary.json)。本轮未继续调度搜索，不声称端到端收益；源文件校验和复现命令归档于010 manifest。

## 19. 增大模型宽度与历史长度的完整 B/H 对照（2026-10-16）

### 19.1 验证依据与预定设置

用户所说的“不饱和”指实际模型/序列工作量可能太小。本轮先用完整训练任务检验规模影响，不修改生产 kernel，不把静态 occupancy 或独立 attention 收益当作完整 B 收益。扩大宽度增加 attention 并行工作量，也增加投影/MLP；扩大历史则保持当前 chunk=2048，增加可移出的历史 dKV。先不增加深度，因为增加层数也会重复每层 Python 分派。

| 配置 | hidden / FFN | Q / KV heads | 数据源训练长度模式 | 最大序列配置 |
| --- | --- | --- | --- | --- |
| small-short | 512 / 1024 | 4 / 2 | 6144,4097,1537,257 | 8192 |
| wide-short | 2048 / 4096 | 16 / 8 | 同上 | 8192 |
| small-long | 512 / 1024 | 4 / 2 | 14336,4097,1537,257 | 16384 |
| wide-long | 2048 / 4096 | 16 / 8 | 同上 | 16384 |

两种数据仍为变长序列；long 只延长模式中最长的序列，不改变其他三种长度，因而不是同一 token budget 的吞吐比较。固定 d128、GQA 比例2、BF16、6层、PP3/VPP2、CP=TP=DP=1，chunk2048、原排序/packing 策略、lr=0。每配置融合/拆分先各跑一次，两轮预热后只采第三轮完整 B/H 边界；不加载内部 profiler、梯度快照或内部计时。仅在结果方向冲突或波动影响判断时追加反序配对。

预先指定 stage1/VPP chunk0、Q2048/history4096 作为四组共有比较形状；long 另报告 Q2048/history12288。同时保留所有 stage/layout 的对应任务与汇总，避免只选有利样本。B 为交付完整 stage dx 的输入反向，不含分开的 W；H 为紧随其后的历史 dKV 计算及累加。B+H 和全体 B 求和不等于 iteration time。归档编号011，实测结论见下文。


### 19.2 实测结果：负载规模会改变完整 B 的收益方向

生产计算、调度和计时代码的校验和与010一致。计时 launcher 增加 hidden/FFN/Q heads/最大序列/chunk 配置入口，默认保持原值，校验 d128 与 Q/KV 整除关系；KV heads 等于 Q heads 时继续省略 GQA 开关。模型仍为6层，宽度512→2048，FFN1024→4096，Q/KV heads4/2→16/8；日志统计总参数约47.06M→358.12M，不代表已测多十亿参数模型。

下面均为 **完整 stage B 的边界 CUDA-event 时间**，不是 attention-only、不是纯 GPU kernel 时间之和。固定 stage1/VPP chunk0，当前Q=2048；每次运行同形状有2个任务样本。单位ms，正变化表示拆分B更慢：

| 配置 | 历史长度 | 样本数/模式 | 融合 B 中位数 | 拆分 B 中位数 | B 变化 | H 中位数 | 配对 B+H 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-short | 4096 | 2 | 5.914 | 6.201 | +4.86% | 0.658 | 6.859 |
| wide-short | 4096 | 2 | 6.093 | 5.844 | −4.09% | 1.244 | 7.088 |
| small-long | 4096 | 2 | 5.961 | 6.224 | +4.41% | 0.918 | 7.142 |
| small-long | 12288 | 2 | 5.902 | 6.345 | +7.50% | 1.016 | 7.362 |
| wide-long，两次运行合并 | 4096 | 4 | 5.913 | 6.188 | +4.65% | 1.307 | 7.487 |
| wide-long，两次运行合并 | 12288 | 4 | **7.279** | **5.748** | **−21.03%** | **3.100** | **8.880** |

B+H 先对每个对应任务求和再取中位数，不要求等于两列中位数之和。各模式在不同运行中执行，提前量指同形状B的耗时差，不能解释成整个iteration中该dx的绝对时间提前相同数值。

重复策略及波动：

- small-short 的两个对应B均回退，范围5.902–5.925→6.112–6.289；wide-short 两个均缩短，范围6.091–6.096→5.834–5.854；未追加重复。
- small-long 的预定stage1两种历史长度均为两个对应B回退；其他stage可能有收益，不能说小模型所有任务都变慢。
- wide-long 首轮历史12288的拆分B范围5.098–6.655，且历史4096两个配对方向相反，因此仅此配置追加一次反序配对（首轮融合→拆分，第二轮拆分→融合）。历史12288两轮中位数分别7.279→5.877（−19.26%）、7.348→5.748（−21.77%）；**四个对应任务全部缩短**，逐任务减少0.568–2.345 ms。全部stage/VPP中同一Q2048/history12288形状两轮共24个对应任务也全部缩短。
- wide-long 的历史4096首轮−4.01%、第二轮+14.89%，方向反转；合并4个样本不能掩盖这个不稳定性。少量任务样本不是独立运行，以上范围不是置信区间。

为避免只选长历史有利任务，以下保留每次运行所有stage/layout的B与H之和。它是跨rank的任务工作量汇总，**不是iteration time**，不同输入长度之间也不能按这些和比较吞吐：

| 配置/运行 | B数量 | H数量 | 融合B总和 ms | 拆分B总和 ms | 拆分B总和变化 | 拆分B+H总和 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-short | 60 | 24 | 364.844 | 399.920 | +9.61% | 416.617 |
| wide-short | 60 | 24 | 369.485 | 364.885 | −1.24% | 387.966 |
| small-long | 108 | 72 | 695.202 | 708.832 | +1.96% | 779.504 |
| wide-long 首轮 | 108 | 72 | 704.777 | 674.928 | −4.24% | 814.986 |
| wide-long 反序复测 | 108 | 72 | 672.903 | 680.973 | +1.20% | 821.922 |

因此，宽模型长历史任务确有提前交付完整dx的测量证据，但**全体B没有稳定的整体缩短，B+H总工作量均增加**。不能把−21.03%写成训练加速比，也不能据此声称增大模型可使所有拆分任务受益。

### 19.3 正确性、解释与下一步取舍

新增形状独立数值验证在计时前执行，6个case覆盖原MHA/GQA/MQA Q2048/K6144，以及GQA16/8 K6144、GQA4/2与16/8 K14336。检查dQ、当前dKV和历史dKV分别与融合FA backward一致：本次所有current/history dK、dV逐位一致，dQ最大绝对误差0.00048828125、最大相对L2约3.461e-5，通过原atol=.004/rtol=.01阈值。这是attention梯度检查；没有把训练loss相同当作更大模型全参数梯度的证明。

五组配对共444个对应B、264个H通过layout与B/H顺序检查，H只出现在有历史的任务上并紧跟对应B。每组融合/拆分三轮loss在日志精度下一致，未发生NaN或跳步。原009的完整训练梯度验证继续作为接入依据，本轮未在正式计时中加载梯度快照或profiler。

本轮回答了“实际工作量可能太小”的问题：**工作规模是影响因素，小模型回退不能直接否定目标在较大负载上的可行性。** 但还不能将原因归结为单一的GPU饱和度：增加宽度同时改变主kernel并行工作量、投影/MLP工作量、CPU提交与GPU执行的相对占比；增加历史也改变kernel网格、缓存和任务上下文。完整B边界区间包含计算流等待主机提交的空隙，没有实际occupancy/Tensor Core/stall计数，未完成硬件瓶颈归因。固定当前chunk增大历史已覆盖本轮“序列更长”的一条路线；增大当前chunk、深度或换真实大模型/长度分布仍是未测范围。

下一步优先级据此调整：

1. 以宽模型长历史形状作为已有正向证据的负载，保留小模型和短历史负例；不再用“小模型拆分B更慢”推出拆分无法提前交付dx。
2. 优先评估无历史任务直接走原融合路径，以及prepare/状态封装与逐序列分派的精简；无历史任务没有H可移出，却在多数本轮配置中增加B总耗时。短历史是否拆分也需要按实测成本选择。以上是下一步候选优化，本轮没有实现。
3. 共享内存、寄存器、搬运/barrier的定向优化仍未完成，不因本轮正向结果宣称三个原因已全部解决。当前区重复扫描已经消除，无需重复做同一合并。
4. 宽模型长历史的B已能提前，具备在该负载上继续评估调度空间的前提；但H仍约3.1 ms，B+H中位数8.880大于融合B的7.279。后续推演必须纳入H、B/H之间的间隙、消费者依赖与通信，重新传播就绪时间；当前立即执行H的测量不能证明端到端收益。

配置、所有原始样本、逐stage/layout汇总、每次运行与合并结果见 [011/summary.json](assets/fa-split/011-workload-scaling-20261016/summary.json)；复现命令、源码快照和校验和见011 manifest。仅维护代码仓库实现计划，未修改外部研究design/plan；本轮不提交或推送。

## 20. Hidden4096 实测与主要矩阵乘理论对照（2026-10-16）

### 20.1 预定范围

在011的宽度扩展基础上继续测hidden4096、FFN8192、Q/KV heads32/16，保持d128、GQA比例2、6层、PP3/VPP2、CP=TP=DP=1、BF16、chunk2048及两种变长数据源不变。这里只将宽度扩大到4096，不冒充LLaMA7B：层数、FFN和MHA/GQA等配置不同。短数据与长数据各先跑一组融合/拆分，若方向冲突或波动影响判断再做反序复测。正式计时只含完整B/H边界；新增32/16-head形状的独立梯度检查先执行，与计时分开。

预先报告stage1/VPP chunk0的Q2048/history4096，以及长数据的history12288，保留全体stage/layout数据。理论只计算有效causal位置的主要矩阵乘FLOPs，区分完整B中的投影/MLP输入反向与attention、移出的两项历史dK/dV和H额外重计算的两项矩阵乘。不将FLOPs比例写成实测时间比例，也不包含独立W或声称iteration收益。使用当前已合并dQ/current的实现，不能沿用16.1中旧版当前区重复计算的额外项。原始材料归档012。

### 20.2 实测：hidden4096 下完整 B 缩短，但 B+H 仍增加

两种数据各完成一组配对，没有改生产计算、调度或计时代码。短数据顺序为融合→拆分，长数据为拆分→融合；前两轮预热，只记录第三轮B/H。模型总参数为1,169,215,488（约1.17B），仍为6层宽度扩展实验，不是完整7B模型。

固定stage1/VPP chunk0、Q2048，每行每模式2个对应样本，单位ms：

| 数据负载 | 历史长度 | 融合B中位数 | 拆分B中位数 | B缩短 | H中位数 | 配对B+H中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 4096 | **8.015** | **7.478** | **6.70%** | **2.025** | 9.503 |
| long | 4096 | **8.469** | **7.350** | **13.21%** | **2.122** | 9.472 |
| long | 12288 | **13.183** | **10.613** | **19.50%** | **6.284** | 16.897 |

这些B是完整stage输入反向，含attention之外的投影/MLP等，排除独立W；H包括历史dKV重计算、GQA归约与梯度路由累加。短/长两种负载中相同Q/history的任务执行上下文不同，不应把两行差值解释为kernel因输入形状发生了变化。

样本与范围：short/history4096融合7.905–8.125、拆分7.100–7.856；long/history4096融合8.122–8.816、拆分7.347–7.353；long/history12288融合12.490–13.877、拆分10.243–10.982。六个预定对应B任务均缩短，各行融合/拆分范围分离，故未追加反序复测。长历史H范围5.847–6.721 ms，报告约6.28 ms而不声称高精度常数；单次运行中的两个样本也不能当作两次独立运行或统计显著性证明。

全量结果也保留：

| 负载 | B/H数量 | 缩短的B数量 | 融合B总和 ms | 拆分B总和 ms | B总和变化 | 拆分B+H总和 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short | 60 / 24 | 36 / 60 | 494.922 | 476.534 | −3.72% | 518.102 |
| long | 108 / 72 | 88 / 108 | 1171.453 | 1038.171 | −11.38% | 1333.917 |

有历史任务中short为23/24、long为71/72个B缩短；无历史任务仍有快有慢。全量总和是跨rank任务工作量，不是iteration time。B+H总和分别增加4.68%、13.87%，本轮仍未证明端到端加速。全部原始样本和逐stage/layout结果见 [012/summary.json](assets/fa-split/012-width4096-theory-20261016/summary.json)。

### 20.3 与理论计算量对照

令当前长度q=2048、历史长度p、模型hidden为D、KV投影宽度为Dkv、SwiGLU中间维为F。TP=1，一次乘加计2 FLOPs，历史有效位置数Ah=q·p，当前causal有效位置数Ac=q(q+1)/2。

原融合B的主要矩阵乘由两部分构成：

- attention backward：`10·D·(Ah+Ac)`。每区域5项主要矩阵乘为S=QKᵀ、dP=dOVᵀ、dQ=dSK、dK=dSᵀQ、dV=PᵀdO。
- QKV/输出投影和MLP的输入反向：`L=2q[D(D+2Dkv)+D²+3DF]`。这里不加独立W；SwiGLU的两个输入投影加一个输出投影贡献3DF。

当前已合并dQ/current的实现：

- 原B：`B₀=L+10D(Ah+Ac)`。
- 拆分B：`B₁=L+6D·Ah+10D·Ac`，历史只保留S、dP、dQ三项，当前区仍一次计算5项。
- 从B移出：`R=4D·Ah`，即历史dK/dV两项。
- 独立H：`8D·Ah=2R`，包含重新计算S、dP及计算dK、dV四项；P、dS等逐元素操作另计。
- 因此B+H比原B多`R`，来源是H的两项额外重计算。这是历史区域5→7；当前区保持5，不再有旧版当前区重复计算的额外项。

FLOPs只数有效位置的主要矩阵乘，忽略边界tile冗余、softmax、归一化、激活、GQA归约、数据搬运、分派、通信等，不是GPU实际执行指令总数，也不是时间上界。GQA在固定D下不减少按Q heads计算的这些attention矩阵乘数量，主要影响KV投影宽度、存储与归约。

本轮D=4096、Dkv=2048、F=8192，单位GFLOPs：

| 历史长度 | 投影/MLP输入反向L | 原attention B | 原完整B主要矩阵乘 | 从B移出R | 拆分B | H | B主要矩阵乘减少 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 618.475 | 429.539 | 1048.014 | 137.439 | 910.575 | 274.878 | **13.11%** |
| 12288 | 618.475 | 1116.733 | 1735.209 | 412.317 | 1322.892 | 824.634 | **23.76%** |

与实测对照：历史4096理论主要矩阵乘减少13.11%，实测完整B耗时减少6.70%（short）或13.21%（long）；历史12288理论减少23.76%，实测耗时减少19.50%。理论和实测方向一致，但不是同一种比例，不能把差额全归因于某一项开销。

为了给出毫秒量级的参考，额外列一个**假设所有主要矩阵乘共享相同有效执行速率**的简化换算：先用同组融合B实测时间t₀标定`B₀/t₀`，再计算`t_B_ref=t₀·B₁/B₀`、`t_H_ref=t₀·H/B₀`。它将融合B的非矩阵乘成本也折入了一个平均速率，没有单独校准attention/MLP或CPU开销，**不是独立性能预测，更不是实测拆分时间**。

| 负载/历史 | 等速率参考B ms | 实测B ms | 等速率参考H ms | 实测H ms |
| --- | ---: | ---: | ---: | ---: |
| short / 4096 | 6.964 | 7.478 | 2.102 | 2.025 |
| long / 4096 | 7.358 | 7.350 | 2.221 | 2.122 |
| long / 12288 | 10.051 | 10.613 | 6.265 | 6.284 |

长历史的B比此参考多约0.562 ms，H约同量级；H数值接近参考不表示模型精确，样本波动已明显大于二者差值。不能把原融合历史dKV耗时直接称为H/2，也不能把这套等速率换算用于未经验证的调度成本。

宽度越大，不代表可移出比例越大。固定GQA比例2、F=2D、q和p时，attention按D增长，而投影/MLP按D²增长：

| Hidden | history4096：B主要矩阵乘可移出比例 | history12288：B主要矩阵乘可移出比例 |
| --- | ---: | ---: |
| 512 | 27.12% | 34.53% |
| 2048 | 18.60% | 28.91% |
| 4096 | 13.11% | 23.76% |

实测更大hidden能改善此前的小模型回退，与“理论可移出比例下降”并不矛盾：执行效率和固定开销占比也在变化。对比011，历史12288的H中位数3.100→6.284 ms，约2.03倍，与D翻倍后H主要矩阵乘翻倍一致；这是跨轮规模趋势，不作为受控的纯kernel效率归因。完整公式、计算结果、等速率假设及逐组实测对应关系见 [012/theory-comparison.json](assets/fa-split/012-width4096-theory-20261016/theory-comparison.json)，可用同目录compare_theory.py复算。

### 20.4 验证与当前结论

新增GQA32/16、Q2048/K6144与K14336两个独立数值case均通过原阈值：当前及历史dK/dV与融合FA逐位一致；dQ最大绝对误差0.000244140625、最大相对L2约2.905e-5。正式训练计时不含这些数值检查；两组配对共168个对应B、96个H通过layout和边界检查，三轮loss均在日志精度下一致，无NaN或跳步。未将loss一致当作本轮大模型全参数梯度对照的替代。

本轮进一步支持在较大hidden下继续研究历史dKV拆分：完整B已获得正向测量，长历史约提前2.571 ms（B耗时中位数之差）。但紧随其后的H约6.284 ms，配对B+H为16.897 ms，高于原B的13.183 ms。最终能否缩短iteration仍取决于H安排、消费者期限和依赖传播。当前没有把H插入其他任务之间，也没有测完整LLaMA7B/13B或新增优化实现。

复现脚本、日志、原始样本、源码快照及校验和归档于012；生产源码与011一致，仅扩展独立测试形状并维护实现计划。保持001–011原始材料不变，未提交或推送。

### 20.5 真实 B/H 时序图（归档013）

直接使用012/hidden4096-long第3轮融合与after_dx的6份rank JSON绘图，未重跑训练、未增加计时、未改生产源码。当前采集只有B/H，没有F/W或通信区间，因此这些图是实测B/H时序，不是完整FBWH图；空白不等于流水线bubble。

- [两个对应长历史任务的放大对照](assets/fa-split/013-measured-bh-timeline-20261016/paired-long-history.png)：stage1/VPP chunk0、Q2048/history12288的全部两个样本，分别为logical0/physical6、logical1/physical6。每个模式只将各自B起点平移至0，保留该任务真实B/H长度及B→H间隔；没有用中位数拼接虚构任务。B分别13.877→10.982 ms、12.490→10.243 ms，H分别6.721、5.847 ms。标记的是完整stage dx本地就绪，不是上游通信接收完成；跨运行的B起点对齐不表示整轮中的绝对dx时间提前相同数值。
- [三卡全部B/H事件概览](assets/fa-split/013-measured-bh-timeline-20261016/three-stage-bh-overview.png)：包含融合108个B与拆分108个B/72个H，按stage分面，紫色框标出各stage的Q2048/history12288、VPP chunk0任务。每rank/运行保留自身CUDA锚点，没有将CPU发射时间戳当作精确跨GPU时钟；此图不用于判定跨卡依赖时延或iteration加速。
- [stage1原始时序局部窗口](assets/fa-split/013-measured-bh-timeline-20261016/stage1-local-window.png)：保留已采集事件原始起止位置，标出B0/B1、H0/H1及logical/physical microbatch。可见H紧跟对应B，当前没有插入其他任务后再执行H。

每张图同时导出SVG；013中plot_measured_bh.py可复现，plotted-data.json保留实际绘制的样本与原始事件、源JSON SHA256及时间原点约定。图与第20.2节中位数一致，进行了输出图的文字/布局检查。001–012原始材料保持不变。
