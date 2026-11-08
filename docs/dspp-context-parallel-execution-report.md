# DSPP CP 实现与实验报告

## 当前实验范围与profile盘点（2026-11-06）

后续正式实验统一61层/H2048/FFN5504/Q16/词表32128，比较PP8×CP1与PP4×CP2；16层/29层仅保留为历史结果。八卡总计划与独立profile定时任务均已按用户要求暂停；本次确认没有仍在运行的profile进程。下一步只维护在 [plan](dspp-context-parallel-implementation-plan.md)，以下旧启动/门槛记录属于对应实验历史，不代表当前运行状态。

**61层PP4×CP2尚无可直接用于性能版B的完整计算＋显存＋P2P profile包。** 两卡真实stage入口已部署，CPU预检确认目标PP4的61层归属、采样world=2且未初始化CUDA。首个功能窗口在60秒空闲确认期间遇外部任务占卡，保护退出、未启动GPU计算；当前仍无61层有效profile。旧数据的用途如下。

| 已有数据 | 实际适用范围 | 对61层B的用途 |
| --- | --- | --- |
| KV完整stage计算、显存、P2P | `dspp-cp-longtask-20261101-1602/profile-long-v512/`；PP4CP2、16层/H1024/FFN2816/MHA8、词表512、chunk2048，绑定当时源码 | 可复用方法与旧实验结论，不能直接当61层/H2048成本 |
| Q/O完整stage小模型包 | `dspp-qo-20261105-exec01/candidate-online-correctness-v1/`及`candidate-small-memory-profile.json`；PP2CP2、4层/H512，共享GPU | 已支持小模型在线B正确性；不是目标性能profile |
| 单卡Q/O局部profile | `single-gpu-qo-local-v1/`，仅attention局部F/B/D/H；隔离证据有限 | 局部诊断，缺CP/PP通信、KV组装/梯度路由、投影/MLP/W，不能替代完整stage时间 |
| 29层现版端到端与owner记录 | 见本报告29层结果及`assets/cp/b-v1-20261106/model29/` | 吞吐/状态分类证据；无已完成的匹配F/B/D/H/W成本包，owner记录也不是61层调度显存预算 |

本次只读核对旧KV CP2 `p2p.json`：120原始记录、24汇总点，所有消息均为2 MiB，CP2、PP4、A800，含计算重叠与独立通信。61层目标H2048/global chunk2048/BF16的PP单lane消息为 `(2048/2)×2048×2=4 MiB`，旧表缺该点；缺口来自消息形状，不是层数本身。完整stage时间已包含任务内部CP通信，不要求额外CP带宽全扫描才能使用B。

## 2026-11-06 阶段 b：八卡现版基线

按用户最新授权启动首批16个配对在线性能窗：MHA/GQA各8，KV/QO/QO/KV/QO/KV/KV/QO，5预热＋20完整更新；数值仍未验收。控制器宿主310667/容器1037167，epoch1793930465.4569693，宿主只读归属审计310715；输出`/workspace/outputs/dspp-cp-v1-20261106/baseline-longmix-v1`，日志`/home/test/test01/tinacui/experiments/dspp-cp-v1-20261106/baseline-controller.log`。每组900秒上限、3次跨60秒空闲检查，审计必须在启动GPU前附加；共用归约/FA/p0设置与输入，计时不带数值快照/诊断，不在服务器并跑CPU分析。[当前性能合同](assets/cp/b-v1-20261106/baseline-protocol.json)。本批完成后继续b其余项，再进入c–f，不能将本批或CP完成当作全计划完成。


阶段a已冻结[chunk-v1](../../../docs/v5/design/chunk_size_selcetion/chunk-v1-20261106/report.md)，按总计划进入b。此节为新16层PP4×CP2目标，不能与旧4层数值或六卡吞吐混称。生产及本机CP源码均不覆盖；旧队列不恢复。

- 已验证归档`candidate-theoretical-validated-source.tar.gz`（SHA256 `abae3c76e26dd6e27fb51ee0d2ccddc3f8eee1feed8ce399f4a4c1c5503c807c`）提取为`/workspace/experiments/dspp-cp-v1-20261106/source-current`，9个关键源文件与旧最终`source-dh`一致。首版另保留`source-first`，只作后续代表性能窗。源清单、隔离入口与控制器见[本批材料](assets/cp/b-v1-20261106/)。
- 当前控制器宿主202371/容器1007591，epoch1793924938.4357908，`run_numeric.py`；宿主只读审计202660。输出`/workspace/outputs/dspp-cp-v1-20261106/numeric-v1`，每组900秒上限、3次跨60秒八卡UUID/显存/利用率/compute PID空闲确认，未知进程或失败则停止后续组。
- 数值先MHA与GQA(KV1)，各执行KV参考、Q/O fused、Q/O immediate D/H，3个逐轮对齐的完整更新。原合同不变：前向有效token .03/.03，其余 .004/.01，relative L2≤.02；保留完整梯度/dx/logits/loss，并增加完整FP32 master及实际更新量对照，避免只看更新后参数掩盖错误。直接记录真实非空H执行，不用计划标签代替。逐轮状态对齐不代表连续优化器轨迹。
- 公共p0精确残差修正已用；v1保持默认高性能FA/BF16归约。KV参考3轮/24 rank报告通过；Q/O fused首轮新增FP32实际更新量对照失败：已写出的rank2/3为3.250855%、rank4/5为2.855338%、rank6/7为2.682350%（2%门槛未改，rank0/1终止前未落盘，不推断其值或全模型聚合）。更新后参数本身allclose通过，不能用其替代实际更新量。原失败完整保留[本地摘要](assets/cp/b-v1-20261106/numeric-v1-failure.json)，未完成后续GQA/DH，正式吞吐未启动。
- v1已关闭、无未知GPU进程、生产dirty patch未改。有限控制`numeric-v2-full-reduction`双方关闭BF16低精度GEMM归约、FA保持默认，同一MHA失败输入3轮：原生完整梯度/dx/前向/loss检查全部通过，**实际FP32更新仍失败**。退出后CPU按每个CP复制对取1个rank，完整271,352,832个参数首轮聚合：梯度relative L2=0.257031%，111,392个梯度符号变化（0.04105%），实际master更新relative L2=2.696287%>2%。符号变化提示Adam首步敏感性，但不等于已证明内核/路由根因。[完整CPU归因](assets/cp/b-v1-20261106/numeric-v2-CPU-attribution/result.json)，CPU CUDA未初始化。
- `numeric-v3-deterministic`同输入、双方确定性FA2 backward＋完整BF16 GEMM归约：**MHA三轮原生完整数值及新增actual-master-update均通过**，共24个rank/轮报告和24个master报告。首轮完整271,352,832梯度及实际更新完全一致；后续各rank更新relative L2最高0.734453%<2%。[独立CPU复核](assets/cp/b-v1-20261106/numeric-v3-CPU-attribution/result.json)、[命令/源码清单](assets/cp/b-v1-20261106/evidence/numeric-v3-manifest.json)。GPU/CPU/宿主审计均闭合，无未知GPU，生产dirty patch未改；逐轮对齐不证明连续优化器轨迹。此证明绑定确定性FA设置，不能据此称默认非确定性路径已通过更新量门槛。
- 默认FA的KV自身复跑已完成原生24份报告且全部通过，但实际更新量再次失败，最大已观测rank relative L2=3.089496%；同后端/同输入/同初始权重与逐轮优化器状态也会超过2%。因此先前默认KV/QO差异不能直接归为Q/O错误。本轮b主比较统一采用已过MHA证明的完整GEMM归约＋确定性FA2 backward，明确标注这是共同数值设置，并非默认非确定性路径吞吐；不更改后端源码或阈值，其他目标与实际32K输入仍待验收。
- MHA Q/O immediate D/H目标三轮对已过Q/O fused完整更新通过：24份原生报告＋24份master报告，实际更新relative L2最大0.276590%；每个rank实际执行12个非空H parts。该证明仍是contracts/vocab32001域，不冒充32K/vocab512。后者CPU输入预检已完成：既有256文档，single loader首8篇训练长度32768/24576/16384/12288/800/224/600/424，共88,064有效token、vocab512，CUDA未初始化；[数据身份](assets/cp/b-v1-20261106/actual-domain-preflight.json)。目标GQA后先补该实际域一轮完整更新再计时，已准备有界`run_numeric_v5_long.py`，尚未启动。
- v4目标批次/宿主审计已闭合，除归因用默认FA自身复跑外，其余四组全部通过[独立复核](assets/cp/b-v1-20261106/evidence/target-review.json)。GQA QO fused对KV更新量最大rank误差0.480793%；GQA QO immediate对已过QO fused为0.147710%，各rank真实12个非空H parts，均为3个逐轮对齐完整更新。源码608个归档文件与实际隔离副本逐一相同，[全文件证据](assets/cp/b-v1-20261106/current-source-verification.json)，生产dirty patch未改。
- **v5实际32K域失败，正式性能门槛未通过。** KV参考完成；QO fused在同一88,064有效token、vocab512、单次完整更新对照失败并停止后续四组。即使v3设置，完整206,603,264参数梯度relative L2=0.263721%，实际master更新=4.123245%>2%；native dx最高2.761138%，logits还违反逐元素前向门槛（norm误差约1.95%不能代替逐元素检查）。235,112个梯度符号变化，与约91.1273%的可唯一映射更新误差关联；这不是路由/内核根因证明。[完整CPU归因](assets/cp/b-v1-20261106/numeric-v5-CPU-attribution/result.json)。所有GPU及审计已关闭，无未知GPU，生产patch未改；不修改阈值，不以短域证明替代实际32K。
- **v6独立CP1对照与CPU分析已完成，GPU/审计均关闭且无未知负载。** 同一32K/88,064-token失败输入、相同初始权重/优化器，CP1用PP4四卡参考，CP2 KV/QO各用八卡。按每个CP复制对取1个rank聚合全模型：KV梯度relative L2=0.103570%，实际更新=1.706392%；KV的outputs/logits/loss逐位相同，原生梯度/dx/参数检查全部通过（dx最大1.078725%）。新增逐stage实际更新guard在stage0为2.290572%而未过，全模型与逐stage口径必须区分，保留原工具失败，不自动改判。
- QO对同CP1：全模型梯度relative L2=0.263085%，实际更新=4.115378%；outputs norm最大1.855457%，dx最大2.763979%，logits逐元素门槛失败（norm最大1.948876%）；loss相对差约0.000586%。因此问题不只是新增master检查较严，QO在原有dx/logits检查也未通过。[完整独立参考汇总](assets/cp/b-v1-20261106/numeric-v6-CPU-attribution/result.json)。仍未证明底层根因，下一步应只采集实际首个差异attention的真实QKV/owner元数据，做定点高精度对照；不盲扫矩阵、不把损失接近当全部数值通过。
- 正式基线脚本与合同已准备，但`actual-domain-review`未通过、protocol仍provisional，`run_baseline.py`拒绝启动；不能把已有数值运行耗时作为性能。独立参考结果出来后按失败实际张量做定点对照，避免盲扫旧算子矩阵；需要变更主比较语义/数值合同才向用户提出具体选项，其他已验收的同阶段工作继续。

## 当前结论

| 项目 | 结论 |
| --- | --- |
| Q/O 后端（2026-11-05） | 首版省显存、吞吐更慢。优化后CP2无拆分/立即H/显式延后H完整更新均通过；CP3无拆分/立即H对同CP3原KV通过。CP2在线B用实测/理论时间均通过数值对照；理论模式实际选择5个D/H、3个H跨任务延后。优化后端到端吞吐未测，单卡局部长历史/短尾存在退步 |
| 原KV路线：CP2＋D/H＋在线 B | 已实现，在下述限定配置通过算子和完整更新对照；不代表所有输入已验收 |
| KV/dKV 通信合并 | 两次配对：受控输入有效吞吐 **+5.67%**，真实 arXiv 输入 **+6.11%** |
| 原KV路线：CP2在线B额外收益 | **未做出来**：已测长序列输入比同配置无拆分模板慢，有效吞吐 **−9.3%**；Q/O在线B收益未测 |
| 可选跨 bin `cp-aware` packing | 算子、梯度/dx/参数更新通过；CP1→CP2 部分前向超阈值，**原因未确认，未验收**，默认仍为 `baseline` |

## 原KV实现与适用范围

- 全局 chunk 容量2048，CP2 每 rank 容量1024。`paired` 固定切分；`hybrid` 把放得下的独立短序列整条分配，只交换跨 rank 共享序列的 KV。长序列及其 continuation 保留全局位置和历史归属。
- **D 发布 dx 前**：计算完整本地 dQ、当前 segment 的 dKV，合入已累积的历史贡献，SUM reduce-scatter 到 owner，然后完成其余 stage 输入反向。**H 只计算本地历史 dKV，无 collective**；必须在对应历史 segment 的 B/D 消费前完成，由消费者归约，贡献只消费一次。
- 一个协调者向整个 PP×CP 副本发布计划，统一任务与 collective 顺序。固定 profile 按进程加载、同轮 metadata 复用，每轮按实际输入求解；没有恢复每轮哈希、内容指纹或全量重复审计。
- PP×CP 曾出现等待环。当前以生产事件、独立收发 stream、接收 tensor 的 `record_stream` 和 CP 专用 FIFO 发送推进线程处理；四次各16更新的 PP2×CP2 无 watchdog 运行及后续 PP4×CP2 运行完成。
- deferred W 沿用现有机制；loss 使用全局有效 token 分母，权重梯度沿 CP 求和，不额外除以 CP。全局 loss 输出仍通过 gather 恢复。

11-01已测环境为 g37、8×A800 80GB、BF16、默认非确定性 FA2.7.4.post1，TP=1、样本 DP=1、CP=2，PP=2/4、VPP=2、均衡分层。当前仅验收 `recv-lookahead=0`、`send-inflight-depth=1`；CP>2、TP>1、多样本 DP、更深队列、其他 GPU 和长期收敛未验收。

## 与 SlimPipe 5.2 的关系

本仓库包含 SlimPipe 开源实现，DSPP 是其上的独立执行路径。[论文5.2](../../../related%20work/SPP/Li%20等%20-%202025%20-%20SlimPipe%20Memory-Thrifty%20and%20Efficient%20Pipeline%20Parallelism%20for%20Long-Context%20LLM%20Training.pdf)的 Variant Context Parallelism 对应 [CPQOAttnFunc](../megatron/core/kv_cache/attention.py)：KV 分片留在 owner，all-gather 当前 Q，各 rank 计算本地 KV 的部分输出，合并 softmax normalizer 后 reduce-scatter 输出；反向收集 Q/O/LSE/dO、归约 dQ，dKV 留在 KV owner。

当前 Q/O 是基于默认高性能FA的移植，不是直接调用SlimPipe原cuDNN入口；首版负收益不能代表论文方法的性能上限。当前 KV 路线采用不同取舍：只 gather 新 chunk 的共享 KV，在本地缓存供后续 chunk 使用，历史梯度在消费者处归约；**并没有每个后续 chunk 重传整个历史 KV**，但共享长序列的历史 KV 存在跨 rank 副本。KV/dKV 打包合并只是减少 collective 次数，与论文交换 Q/O 的方法不同。首版Q/O实测降低显存但吞吐更慢；调优后吞吐未测。GQA/MQA 下 Q/O 比 KV 宽，不能直接套用论文的等尺寸近似。

原实现不能直接开参数复用：[Transformer](../megatron/model/transformer.py)优先将 DSPP state 导向专用算子；原 `cp_qo_attn_func` 依赖旧 KVCache/ctx_pair、cuDNN SDPA 和分片保存钩子，未接 DSPP packed metadata 或 D/H。Q/O的H在KV owner本地计算历史dKV；Q/dO/全局O/LSE须提前具备，D的dQ跨卡归约须在发布dx前完成。新后端修复了“长链尾部Q只归一张卡，历史KV owner仍需参与”的通信判定；优化版按不重叠Q区间计算本地KV前缀，减少局部调用/合并，精简重复元数据和连续KV拷贝。这些是DSPP＋FA移植调整，并非直接复用SlimPipe优化代码。模板D/H及CP2在线B已开放；CP3在线B仍受限，不能直接删限制使用。

## 正确性：通过了什么，还缺什么

| 对照 | 结果与边界 |
| --- | --- |
| Q/O 新算子与更新 | CP2 paired/hybrid、CP3 hybrid各27例/rank通过FP64；原KV的CP3亦27例/rank通过。CPU因果/布局27通过。PP2×CP2复用旧CP1参数/optimizer参考，三轮84项通过；非连续轨迹或吞吐证明 |
| 优化Q/O算子/新内核 | fused/立即H/延后H对FP64：CP2 hybrid共81例/rank、CP2 paired及CP3 hybrid各27例/rank通过；H无通信、不改变已发布dx。新partial及原因果/设备缓存回归41通过；CPU布局/profile60通过 |
| 优化Q/O完整更新 | 4层/H512/FFN1024/MHA4/VPP2。PP2×CP2/chunk2048对CP1：无拆分、立即H各3轮84项通过；PP2×CP3/chunk2040对同CP3原KV：两模式各2轮84项通过。每轮对齐初始参数及optimizer，默认BF16 FA；不证明连续轨迹一致或长期收敛 |
| 优化Q/O显式延后H | 同上PP2×CP2小模型，实际owner审计生成匹配Q/O的显存profile，正常准入。三轮84项通过，第二轮4个H跨过其他任务后执行；CP3尚未做此完整模型模式 |
| 优化Q/O在线B与回退 | 同上PP2×CP2小模型，single采样24篇受控文档，对同CP2/QO/预取配置的无拆分模板逐轮对齐。真实异步B三轮84项通过，拆分数[0,1,0]，第二轮D/H在stage0/chunk1本地相邻；各rank完整计划一致，CPU规划进程未初始化CUDA。诊断注入第二轮结果缺失后统一回退template，另三轮84项通过。共享负载profile仅驱动数值验证，不作提速证据；CP3在线B与其他模型未验收 |
| 纯理论时间驱动在线B | 同一PP2×CP2/4层/H512模型与输入，完全移除计算/P2P时间profile参数，仍用真实FA/通信/异步规划及显存准入。假定1TFLOP/s、1GB/s时三轮84项通过（零拆分）；同假定注入未就绪回退另84项通过；带宽改为0.01GB/s后自然拆分[0,5,0]，其中3个H跨其他任务执行，再84项通过。三组各rank计划完全一致，CPU规划无CUDA。新理论模型CPU6项、默认profile路径回归12项通过；假定值不是A800性能测量 |
| 独立 FP64 attention/autograd | KV 合并前后，paired/hybrid 各72例/rank；可选 cp-aware 81例/rank通过。覆盖 MHA/GQA/MQA、历史累加、混装、padding、空 Q；检查 H 无通信及不再改变当前 dx |
| CP1→CP2 完整更新 | PP2、4层/H512，fused 与立即 H 各3轮、各84项通过；每轮对齐参数及 optimizer state，验证单次更新，不证明连续轨迹一致 |
| 显式延后 H | PP2×CP2 三轮84项通过，第二轮4个 H 跨过其他任务后执行；发送推进路径修改后同项复测通过 |
| 在线 B 实际选择 D/H | PP4×CP2、16层/H1024，两轮112项通过，第二轮实际拆分1个 B；参考是**同 CP2 的 fused 模板** |
| 显存 | 8 ranks 的 allocated 峰值均低于预测，最高9.58GiB；仅覆盖该诊断输入 |
| 回归 | CPU 58通过、8个旧契约测试排除；CP1内核数值/状态42通过，旧 tile grid 断言修订后单项通过；profile硬件匹配6通过，GPU等待2通过 |

人工边界包含 `800/224/600/424`、`1500/548`、`700/700/648`、`4608/768/768`、`6401/2305/97/31`，不是只用整齐长序列拼装。[数值合同](assets/cp/numerical-contract.json)：有效 token 前向 atol/rtol=.03/.03；梯度、dx、更新 atol/rtol=.004/.01；相对 L2≤.02。早期更严格前向判据的失败保留，后续合同通过不改写旧结果。

**可选cp-aware的缺口是同一失败输入上的基础CP控制。** 跨 bin 用例训练长度为 `2464/768/3520/3936/864/3328/4384/1`。CP1→CP2 三轮84项中77项通过，7项失败均为 outputs/logits；前向相对 L2 约0.28%–0.60%，最大绝对差约0.079。梯度、dx、更新后参数通过；同 CP2 fused 与立即 H 的84项也通过。

这次失败同时涉及 CP、hybrid 布局、KV 合并和新 packing，**没有在同一输入上逐项开关对照，不能确定是哪项引入，也不能直接归因于数据换序或 BF16**。较早另一组输入的 CP1→CP2 通过不能补上该缺口。早期基础 CP 诊断定位过“投影输入相同、GEMM 输出微差”，独立 FP64 投影误差约0.17%；它只解释已测局部现象。连续 BF16 更新 `update-v1` 的后续梯度失败也未改写为通过。

## 性能：分别比较了什么

**通信合并**指 K/V 从两次 all-gather 打包成一次、dK/dV 从两次 reduce-scatter 打包成一次；不是合并 D/H。端到端两边均用 hybrid＋无拆分 DSPP 模板，仅替换通信算子，包含打包、CPU准备、W和optimizer开销。

共同配置：8卡 PP4×CP2/VPP2，16层均衡、H1024、FFN2816、MHA8、global batch8、micro batch4、chunk2048；2预热＋8计时，取全副本最慢 rank 窗口，两次前后交替配对，关闭快照、审计和详细 profiler。

| 输入与比较 | 基线 ms/更新 | 改后 ms/更新 | 有效吞吐变化 |
| --- | ---: | ---: | ---: |
| 合成混合输入，max sequence8192，vocab32001；KV合并前→后 | 487.97 | 461.81 | **+5.67%** |
| 现有预分词 arXiv 变长数据，max sequence8192，vocab151936；KV合并前→后 | 1211.94 | 1142.20 | **+6.11%** |
| 独立长序列合成输入，max sequence32768，vocab512；无拆分模板→在线B | 1170.65 | 1291.04 | **−9.3%** |

arXiv 数据为 `/workspace/datasets/slimpajama_arxiv_50k/megatron/slimpajama_arxiv_50k_text_document`。这是读取既有真实文本数据；每次64篇计时样本，不代表测了整个文件，也不代表在线 B 有6.11%收益。

B 比较使用相同优化后 CP 运行时，2预热＋6计时，合成训练长度按以下16项循环、重复16次：`32768/24576/16384/12288/800/224/600/424/16384/768/768/12289/6401/97/31/548`。8轮拆分数为 `[0,1,2,1,0,0,0,0]`，第5、7轮 planner未就绪统一回退；负收益原因尚无充分分解证据。不能据此断言所有 CP2 配置下 B 都无收益。

局部 transport 的5次交替窗口中位耗时：共享0.607→0.379ms、混合0.611→0.320ms、全私有0.163→0.115ms。通信减少也不保证布局整体更快：一次8K配置对照中 hybrid 472.95ms，paired 436.75ms。原 CP1/PP8 的受控 +3.34% 与上述实验独立。

## 可选 packing 与 profile 注意事项

`cp-aware` 约40行 CPU 候选逻辑，最多评估64次可行移动：只把完整独立短序列移到其他既有 bin，固定长链尾部、不增加 physical task，不加 GPU 通信路径。它可能让短序列完整落在单 rank，减少共享 KV；也可能破坏负载均衡，因此保持非默认、未验收。

人工用例保持11个任务，单层 CP2/BF16/MHA8 通信量代理157,810,688→144,441,344 bytes（−8.47%），CPU准备中位数0.511→0.931ms；1000个人工随机布局中22个减少共享短序列 token。**没有该候选自己的端到端性能结果**，上述 +5.67%/+6.11% 均不包含其收益。

计算 profile 必须与 owner 显存审计分开采集：首批 immediate profile 被审计抬高 D 成本，最终 B 使用重新采集的 `profile-long-v512/compute-clean.json`，配套 `memory.json`、`p2p.json`。准备阶段检查模型、硬件、布局、CP实现及P2P lane数；换配置需确认适用性。显存取各 CP rank 最大值并保留解析下限、15%状态余量、至少256MiB workspace、2GiB reserve及10%增长余量；不能把全局历史 KV 简单除以 CP，也不是任意输入的容量保证。

正确性可用显式理论模式：`--dspp-cost-model theoretical --dspp-assumed-tflops 1 --dspp-assumed-bandwidth-gbps 0.01`。节点预测=主矩阵FLOPs/假定吞吐＋节点内CP通信量/假定带宽，PP边预测=实际wire字节/假定带宽；H通信量为0。按实际packing/owner统计因果QK对、GQA投影宽度与每stage层数，计入FA backward及D/H重计算；不含pointwise、tiling、启动/显存带宽开销或DAG外optimizer/梯度收尾。该模式不读取任何计算/P2P时间profile，仍保留原显存字节准入；实测P模式默认行为保持不变。验证命令、原始失败/报告及源码归档见 `candidate-theoretical-correctness-v1/`、`dspp-theoretical-correctness.py` 和 `candidate-theoretical-validated-source.tar.gz`。

排除的8个旧测试涉及已删除的内容指纹、逐轮解析下限/全图/checksum、旧 agreement 和重复显存审计合同；没有为通过旧断言恢复运行时重复工作。真实 CP 数值失败另行保留，不与旧契约混淆。

六卡首版对比：12层/H1024/FFN2816/MHA8，VPP2均衡分层，chunk2040、global batch8、micro batch4，max sequence32768、vocab512，使用既有 `longmix-v512`。2预热＋8计时，KV/QO/QO/KV交替，两次配对；关闭D/H、在线B和诊断。

| 输入/配置 | KV ms/更新 | 首版Q/O ms/更新 | Q/O吞吐变化 | 最大allocated，KV→Q/O |
| --- | ---: | ---: | ---: | --- |
| MHA，PP3×CP2 | 1257.24 | 1896.74 | −33.72% | 9.93→7.01GiB |
| MHA，PP2×CP3 | 1663.44 | 2407.65 | −30.91% | 12.89→7.29GiB |
| GQA（KV heads=1），PP3×CP2 | 1401.35 | 1778.09 | −21.19% | 4.52→4.10GiB |

独立诊断中，CP2历史KV逻辑峰值635,043,840→317,521,920 bytes，符合分片减半；Q/O仍需更多前反向collective。上述比较的KV已合并K/V及dK/dV通信，所有性能窗口均关闭D/H和在线B。首版节省显存、吞吐更慢；GQA的整进程allocated只减少约9%。GQA和arXiv队列尚未全部完成，优化版吞吐仍未测。更广CPU回归候选47通过/12失败/4旧契约排除，冻结基线46通过且失败集合完全相同，未新增失败；不能声称整套历史测试全绿。

**单卡Q/O局部profile已采完，隔离证据有限，作为初测保留**：120条owner记录、48组形状；BF16/默认FA，chunk2040、Hq8/head_dim128、KV heads=8/1。CP2/CP3分片形状在单卡逐个owner执行；F/B首版与优化版ABBA配对，5预热＋15重复，D/H单独采样。启动前GPU0显存0、利用率0、无compute进程；每组计时前后检查compute进程数量为1，否则脚本停止。但未连续记录PID、CPU负载及GPU频率/温度/功耗，不能保证窗口内短时干扰或共享主机影响已排除。非空owner两次窗口中位耗时比，中位1.002、最高1.086，只说明重复性，不证明无干扰。原始数据、命令和自动汇总在 `single-gpu-qo-local-v1/` 及同名 `.command.sh`，共享冒烟不计入结果。

代表长历史初测：当前Q=2040、此前history=16320。下表为局部毫秒数，每项取各owner最大值；不是实际多卡stage耗时，也不是已排除全部干扰的校准值。

| CP形状 / attention | F 首版→优化版 | B 首版→优化版 | 优化版D | 优化版H |
| --- | ---: | ---: | ---: | ---: |
| CP2 / MHA | 1.217→1.445 | 1.995→2.160 | 1.441 | 1.979 |
| CP3 / MHA | 1.081→1.620 | 1.729→1.972 | 1.270 | 1.884 |
| CP2 / GQA（KV=1） | 1.213→1.444 | 1.927→1.892 | 1.381 | 1.373 |
| CP3 / GQA（KV=1） | 1.074→1.613 | 1.711→1.684 | 1.210 | 1.168 |

初测观察：history=2040/Q=2040时，F耗时减少约48%–54%、B减少约24%–31%；history=32640/Q=128的短尾，F为首版2.21–2.67倍，B为1.50–2.55倍。代表长历史/短尾ABBA窗口中位数最大相差约1.8%/4.5%，退步信号有重复性，但倍率及原因需补隔离复测，不能直接归因于FA切分。完整stage的dx交付仍需其余反向及必要通信，额外H能否隐藏取决于真实流水线。该profile不含CP/PP通信、KV组装、owner梯度路由、投影/MLP/W或元数据准备，schema明确不兼容在线B完整stage成本，不能据此声称端到端提速。

CP3→CP1两轮84项中80项通过，4项失败为第二轮outputs/logits，KV和Q/O失败任务、元素数和最大差值相同；梯度/dx/参数更新通过。首层QKV投影小探针输入输出逐位一致，**未解释**这些前向outlier；同CP3后端对照84项通过不能替代跨CP严格前向验收。CP3性能仅作此数值边界下的后端比较。

Q/O独立目录 `/workspace/outputs/dspp-qo-20261105-exec01/` 保存冻结首版 `source/`、`baseline-source.tar.gz`、候选 `source-dh/` 及已验收源码 `candidate-small-validated-source.tar.gz`、`candidate-online-validated-source.tar.gz`。`candidate-small-*` 保存完整更新数值结果，`candidate-operators-*` 保存FP64对照，`candidate-native-small-v1` 保存内核回归；显存证据为 `candidate-small-audit-v1/owners` 和 `candidate-small-memory-profile.json`。在线B命令/计算与P2P profile/参考张量/注入回退脚本见 `candidate-online-correctness-v1/` 及根目录验证脚本，实际输入长度模式为 `800/224/600/424/1500/548/700/700/648/4608/768/768/6401/2305/97/31`，取前24篇。首轮冷启动耗尽规划预算而未拆分，后两轮预算内；不据此推断稳定性能。小模型使用GPU0–3/0–5共享运行，每进程PyTorch分配器限10GiB；rank0在线B峰值allocated约1.45GiB。未修改他人任务或GPU全局配置，原始日志/张量留服务器，共享耗时不计入性能。小模型正确性及单卡profile进程均已退出，六卡端到端队列停止，原定时任务保持暂停。

## 证据与复现入口

实验基点 `26adff346c2f142e98bdbe1f55cd87cb68ce17ea`，分支 `cuiym-dspp`，包含既有及本次未提交补丁；仅 HEAD 不能复现实验。容器 `tinacui-megatron`，Python `/workspace/venvs/megatron/bin/python`。最终独立目录 `/workspace/outputs/dspp-cp-longtask-20261101-1602/`：`source/` 为源码副本，`final-source.tar.gz` 为归档，各运行旁 `.command.sh`/`command.sh` 保存实际命令，原始日志、失败、timeline、张量快照留在服务器。

原阶段的本地 assets 保留9份材料（本次b阶段新增独立的`b-v1-20261106/`，不覆盖原材料）：[Q/O对照汇总](assets/cp/qo-comparison.json)、[验证汇总](assets/cp/validation-summary.json)、[性能窗口与配置](assets/cp/performance-summary.json)、[transport原始配对](assets/cp/transport-pairs.json)、[packing成本](assets/cp/cp-aware-cpu-benchmark.json)、[早期前向诊断](assets/cp/forward-diagnosis.json)、[数值合同](assets/cp/numerical-contract.json)、[源码身份](assets/cp/source-identity.json)、[合并前算子复现入口](assets/cp/unfused_baseline.py)。源码 hash 仅用于一次性归档；归档之后的本机改动不自动视为已测。

## 29层短序列打包基线：16窗口完成

同29层/H2048/FFN5504/Q16/词表32128、PP4CP2/VPP2、固定chunk2048，无D/H/在线B；每窗口5预热＋20完整计时更新，109920有效token，MHA/GQA各8窗。全部退出码0，原始输入/完整更新/实际层归属及干净60秒窗口已在本机复核，审计无未知GPU负载，生产补丁未变。MHA KV/QO为12440.34/9559.98 tokens/s（Q/O −23.15%），GQA1为12258.29/9695.71（−20.90%）；allocated峰值MHA为10162472960/9675720704 bytes，GQA1为8146766848/8058177536 bytes。当前含数值误差版本，数值未验收；不是c的61层对比。

证据：[完整复核](assets/cp/b-v1-20261106/model29/short-packed29-v1/timing-review.json)。已串行接续short-tail29-v1，宿主控制器710118、容器1175446、审计710150、监督710153；命令及日志见唯一检查点。b后续profile/容量/模板B仍待完成。

## 短尾资源中断及共享数值对照

短尾short-tail29-v1的11个窗口在外部GPU负载前完成并经本机复核；GQA04在首次确认外部PID时尚未获准，保守排除，GQA05–08未运行。只补这5窗，原输出和污染时间线保留。MHA完整8窗KV/QO为22710.67/16995.36 tokens/s（Q/O −25.17%）；GQA只有3窗，不发布完整比较。证据：[中断复核](assets/cp/b-v1-20261106/model29/short-tail29-v1/interruption-review.json)。

空闲不足期间复用旧16层失败输入/原KV参考，仅将Q/O的weightedO归约改成FP32求和后再转BF16。8卡完成一次完整更新，初始master全量逐元素一致；实际更新rL2从4.12325%变成4.07703%，dx最大2.74216%、logits仍失败，未解决旧误差，不采用此对照改动。共享时间不作性能。控制器收尾误要求所有rank提供valid_tokens（仅首stage提供），原KeyError保留；8份native及8份master报告完整，CPU复核修正读取方式，无GPU重跑。证据：[CPU复核](assets/cp/b-v1-20261106/numeric-v5-fp32-o-v1/CPU-review.json)。

## 旧数值失败的首attention差异边界

八卡不足期间，仅对旧16层失败输入补KV/QO各一次首任务探针，沿用原KV参考/初值、full GEMM/deterministic FA/exact p0及现有运行时；不是29层正式性能。两路均生成8份完整更新与探针报告，生产未改。实际首任务是32768序列的首2048片段、无历史；首层QKV线性层输入/权重/输出及postRoPE Q/K/V逐元素相同，但attention输出QO/KV rL2为0.1840614%。前512个query（因果前缀只有rank0的K）输出完全一致；从query512开始差异，恰好进入跨owner前缀，两后续区间rL2为0.31894%/0.31127%。这定位了差异出现的算子/边界，不等于证明根因或全模型正确。CPU FP64参考及简单BF16舍入模拟不足以精确重现native结果，下一步复用已捕获QKV做有限原生FA/分块softmax回放，不重复完整模型对照。证据：[首任务比较](assets/cp/b-v1-20261106/numeric-v5-first-attention-CPU-v1/result.json)、[owner边界](assets/cp/b-v1-20261106/numeric-v5-first-attention-CPU-v1/owner-range-review.json)。性能仍优先，待续短尾5窗。

## 首attention原生FA单卡回放与后续准备

复用已捕获的同一postRoPE QKV/真实owner/metadata，在一张显存足够的共享GPU上按原生KV和Q/O的FA2分块调用回放，并本地模拟两rank O/LSE合并；无模型、PP或真实collective。两路回放各自与原捕获输出逐元素一致，全量FA调用也与KV逐元素一致，QO/KV仍为0.1840614%；weightedO改FP32相加仅降至0.1524958%。因此该首片段差异可由Q/O分块FA及合并计算路径完整复现，无需通信或流水线参与；不等于全模型4.123%更新误差已完全归因/修复，不变更数值门槛或正式源码。allocated峰值仅81457664 bytes，是此回放的资源记录，不是目标训练容量/性能。[回放证据](assets/cp/b-v1-20261106/numeric-v5-first-attention-native-replay-v1/result.json)。

已在CUDA禁用的CPU窗口完成独立source-model29实体布局文件准备及原有split扩展缓存加载；只改隔离副本dspp_layout.py，原source-current/生产未改，后续正式profile无需在GPU窗口临时编译。[CPU准备凭据](assets/cp/b-v1-20261106/model29/model29-cpu-preparation-v1.json)。八卡恢复后先续短尾缺少的5窗。

## 29层GQA fused共享显存归属与通信量核对

KV/QO各6次完整更新、8rank有限值/实际29层归属/16份CP双lane owner文件均通过。6次更新共376050有效tokens，统计所有8rank的attention张量通信量折算：KV 10787665920 bytes，Q/O 260930956800 bytes，约24.19倍；这是按张量大小折算的ring-equivalent发送量，不是实测链路字节、带宽或耗时。F边界采样的去重历史KV storage最大值为159383552/79691776 bytes，Q/O减半；pending dKV采样最大值也为92274688/46137344 bytes。说明本GQA形状省缓存伴随很大的通信量代价，时间瓶颈贡献仍需独占profile核实。不同分项独立峰值不相加、不宣称容量上限或最终B内存合同。[小型汇总与限制](assets/cp/b-v1-20261106/model29/owner-fused29-gqa1-v1/owner-summary.json)。

原始owner JSON约9.4GB留服务器，CPU仅提取native头部和小rank摘要，未重扫/下载大文件。已接续原b3缺少的29层GQA Q/O非空H功能/owner单案例，仍为共享诊断、不开在线B，不新增全数值矩阵。独占资源恢复先补短尾5窗。

## 29层GQA Q/O目标非空H功能检查

原b3目标形状（29层/H2048/Q16/KVheads1/PP4CP2/globalchunk2048）共享运行6次完整更新，每rank实际完成H jobs为810/810/1296/1296/1296/1296/1296/1296，非空parts均大于0；8rank有限值、未跳过更新、实际全29层归属及16份owner文件通过，生产未改。allocated/reserved最高17930034176/18578669568 bytes仅为共享诊断资源记录。没有做此目标的fused/split完整梯度/更新对照，也不称25次正常稳定性或性能验收。[功能计数与范围](assets/cp/b-v1-20261106/model29/owner-split29-gqa1-qo-v1/H-summary.json)。原计划匹配KV单案例已接续，八卡独占恢复仍先补性能5窗。
