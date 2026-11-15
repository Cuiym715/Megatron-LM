# 原生 DSPP 模型与待补 profile（2026-11-08）

本次只做 CPU 数据接入、拟合与导出，没有启动 GPU 作业。目标为 A800-SXM4-80GB，PP8/VPP2/CP1/TP1，61 层 endpoint-3-2，hidden 2048、FFN 5504、MHA16/head-dim128、vocab32128、BF16，原生 attention 和 after-dx，CUDA_DEVICE_MAX_CONNECTIONS=32。CP2、其他词表或硬件需单独模型。

## 已完成的接入

可加载模型为 `profiles/dspp-native-pp8/model.json`，来源清单为同目录 `sources.json`。运行时只加载模型文件一次，不读取历史目录，不重新拟合。完整数据源路径与 SHA256 在模型中保留；相对路径按 sources.json 所在目录解析。

| 数据 | 本次用途 | PP8 实测键数（重复测量合并后） |
| --- | --- | ---: |
| pp8-contracts-20261106 原生 2048/4096 F | 查表与拟合 | 208 |
| 同上 B_fused | 查表与拟合 | 208 |
| 同上 W | 按容量合并测量的中位数，查表与拟合 | 32 |
| controlled-frozen-v2 的原生 2048 D | 查表与拟合 | 256 |
| 同上 H | 查表与拟合 | 240 |
| 局部 PP2 [3,4,4,2] 原生 profile | 按同层数/端点角色匹配，通过共同形状校准后只参与拟合 | 不冒充 PP8 实测 |
| p2p-wide-20261106 | 原样使用 14 个方向、8/16/24/32MiB、isolated/compute，共112点 | 独立通信数据 |
| 原生 2048/4096 fused 显存与历史2048 after-dx显存 | 合并已有上界，按 token 缩放动态项 | 尚未完成当前大容量 H 延后校准 |

历史 D/H 从本地 `dspp/archive/completed/runtime-opt-20261031/compact-archive-v1/gzip-profile-v1/stage-profile.json.gz` 读取：这是25776条受控 profile 的无损压缩副本，并非重新测量，也不是确定性版本改标签。

共80个 stage/VPP/op 模型、944个实测键。取消每组32点截断；完整匹配实测键时始终直接用测量值。W采用容量键，多个形状的 W 测量取中位数，修复此前后写入记录覆盖前记录的问题。粗筛仍使用批量公式。

局部数据共检查18个容量；当前配置域2048～8192实际使用9个容量：2048、2560、3072、3840、4096、4608、5120、6144、8192，共18份 fused/split 文件。对相同 stage 角色和算子，以共同形状的 PP8/局部耗时比中位数校准；得到1680个用于拟合的推算点。这些点不写入 exact 表，`measured_capacities` 也不计入它们。补充真实 PP8 数据后，相同键会优先采用 PP8 数据。

旧文件没有 arithmetic_modes 字段时，从明确列出的来源声明读取其执行方式；源文件若明确为 canonical/deterministic 仍拒绝混入原生模型。仅缺标签不再成为重测理由。硬件、模型、stage语义仍需相同；老的内核摘要与 `not-checked-at-runtime` 字符串差异不再被当成模型配置不同。原始上下文保留在来源记录中。

## 当前模型的准确性边界

CPU按整个目标容量留出，校准比例也只由训练容量计算，避免使用留出容量的信息：F相对误差中位7.44%、P95 17.59%；B_fused为7.00%、16.73%；W为5.00%、9.59%。这些只是已有2K/4K数据的交叉验证，不能当作6K/8K或训练吞吐误差。

D/H只有2K的目标PP8实测，每组与局部数据只有一个共同形状，不能给出可信的跨容量验证结果；报告中为null，不把单点校准残差0解释为预测准确。新代码与历史测量间的性能差异，需要下面少量原生2K锚点确认。

显存包复用当前目标配置的原生2K/4K常驻测量，以及历史after-dx的动态存储项。历史文件base_policy明确说明stage0约18.86GiB的resident继承自63层/Qwen大词表，故不把这个不同模型的上界带入当前61层/vocab32128；当前使用目标配置实测约7.47GiB。历史after-dx的3GiB workspace allowance暂时保留并标注为旧预算，不能称为当前实测workspace。实际Adam初始化后resident、临时workspace和H延后存活量仍安排少量对照。包中的 `status` 和 `limitations` 明确记录这些待验证项；本次没有声称自动容量已完成端到端验收。

2048～8192是本次实验的搜索域，写在sources.json中，不是算法B的数学限制。更大域需对应的数据依据。当前F模型为普通前向，split捕获额外前向成本尚未单独建模；补测时单列F_after_dx，不把它混进普通F。

## 等待 GPU 的具体工作

机器可读任务单为 `profiles/dspp-native-pp8/profile-plan.json`，包含每个容量的逐形状 query/history、算子、预热次数、测量次数、显存任务、留出点和后续端到端窗口。此任务单是未来执行依据，不是已完成记录。

D表示先执行的反向主体，H表示稍后执行的历史KV梯度部分。每个目标成本点都覆盖8个物理stage×2个虚拟stage；采完整stage耗时，不能仅用attention kernel耗时替代。每形状2次预热＋5次记录，保存毫秒值、容量、形状、stage/op和执行上下文。允许同模式复用模型实例，避免每个形状重新启动模型。

| 优先级 | 需要补什么 | 容量和范围 | 为什么 |
| --- | --- | --- | --- |
| P0 | 原生执行方式锚点 | 2048：短历史整块、近32K历史整块、混合packing，共3形状；F/B/W/D/H及F_after_dx | 确认旧原生数据与当前算子耗时一致，不重跑整个2K库 |
| P0 | 目标PP8 D/H及捕获前向 | 4096、6144、8192各7个有历史形状 | 当前对应容量只有局部推算，缺目标运行方式的校准 |
| P0 | 目标PP8大容量普通前后向 | 6144、8192：上述7形状加2个无历史形状，测F/B_fused/W | 这些容量已有完整PP8数据是canonical版本；原生只有局部测量 |
| P0 | 当前完整训练显存 | 2048、4096、6144、8192，各template和有实际H延后的尾部B；详见下文 | 明确Adam常驻、H延后和大容量workspace，替换旧保守上界 |
| P0 | 独立预测验证 | 3072、5120各1个非整块尾部形状，F/B/W/D/H | 不参与首次拟合，验证容量内插及尾部预测；记录误差后才可加入最终模型 |
| 按目标数据启用 | 长历史跨容量 | 若正式数据超过32K：4096、8192，64K/96K/128K上下文，整块和半块query | 当前接入PP8数据最长32K；先查历史2K长history局部数据，避免重测 |

P0成本采样共28个形状配置（不同算子可共享同一形状），另2个容量留出形状。不是对所有容量、历史长度、packing穷举。

对容量C，7个有历史形状明确为：

- 整块query=C，history分别为C、`floor(16384/C)*C`、`floor((32768-C)/C)*C`。
- 非整块query=`C//2+1`，history取上述中、长两个值。
- 混合packing：第一段query=`C//2+1`、第二段query=`C//4-1`；第一段history分别为C和上述最长值，第二段history=0。
- 6K/8K另加无历史query=C及`C//4+3`。D/H不在无历史形状上制造假任务。
- 留出容量3072/5120：query=`3*C//4+7`，history=`floor(16384/C)*C`，不沿用训练形状。

上述query/history是算子输入形状。完整流水线中相应任务需在真实前缀和相邻通信下采样。若先用独立stage探针采样，输出必须标为isolated，加入local_compute；不能把它改名为compute-p2p。PP8成本校准需完整8卡窗口；只有少量卡时可以先做局部补测，但不称为完成PP8校准。

显存分别使用 `16条2048+1条16384`、`16条2048+1条32768` 两组输入。每个容量/调度先执行3次完整Adam更新，再采1次；尾部B必须实际选择H延后，否则这一轮只能算template显存数据。分别记录初始化后常驻、F激活、W闭包、attention capture、KV、pending-H、workspace及allocated/reserved峰值。成本计时、存储归属采集、正式吞吐分三类离线运行，不能相互混用。

## 不需要重复做的内容

原生2K/4K完整F/B/W库、历史2K完整D/H库和当前物理peer映射下的P2P直接复用。P2P仅在peer拓扑、通信实现/重叠策略或消息大小范围变化时补测。无需恢复a4–a6冻结入口、canonical适配器、逐轮hash校验或训练中的存储扫描。

## 执行与产物

当前CPU导出命令（在Megatron-LM-kwai目录执行）：

```bash
python3 tools/export_dspp_chunk_model.py \
  --manifest profiles/dspp-native-pp8/sources.json \
  --output profiles/dspp-native-pp8/model.json
python3 tests/unit_tests/dspp/test_chunk_export.py
```

新测量支持raw `records` 或compact `groups`，JSON/gzip均可；在sources.json追加对应来源后用同一命令导出，不需要新建预测分支。实际测量值只进入compute，局部数据进入local_compute。F_after_dx另存，不合并F；先量化其差异，再决定是否需要将其作为B选择的前向增量成本。

历史run_window脚本依赖已删除的timeline/owner参数，不能直接执行。GPU采集时需将必要事件/存储采集绑定到独立离线profile入口，不能把诊断开关重新加入正式schedules。本文和JSON任务单规定了输入和输出；没有把未移植、未经GPU执行的旧采集命令标成可用命令。

补测后先完成上述留出耗时和实际显存对照，再以 `tools/benchmark_dspp.py` 的5预热＋20计时窗口比较：固定2048 template、固定2048尾部B、自动容量template、自动容量尾部B；四组用同一模型和输入。此时只记录完整更新时间、有效tokens/s、显存峰值和计划选择/回退，不携带profile事件。正式统计重复次数按实际噪声确定，不用CPU预测收益替代实测收益。
