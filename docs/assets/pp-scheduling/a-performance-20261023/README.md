# 算法 A：有分布依据的 PP3 性能对照（2026-10-23）

**吞吐结论：当前 P1 同步实现的 A 在本机 PP3 配置中没有提升吞吐，相对旧 DSPP 下降1.80%。** 18次正常完成运行、保存计划复核及独立六次更新的确定性数值对照均已结束；另保留一次legacy停滞，不声称稳定性全部通过。原始材料位于 `/home/cuiym/tmp/dspp-a-performance-20261023/`，旧实验归档保持只读；不 commit/push。

| 方法 | 有效目标 tokens/s | 相对 legacy |
| --- | ---: | ---: |
| 原 DSPP legacy | 13163.98 | 基线 |
| 原模板通过新接口 template | 13033.99 | −0.99% |
| 严格 A，同步规划并执行 | 12927.43 | −1.80% |

每方法6次运行、72个计时optimizer step，处理2490752个相同有效目标token；汇总为token总和/各次block耗时总和，不取最快一次。A相对template为−0.82%。六组配对的A/legacy变化为−2.23%～−1.11%，六组均低；A/template为−3.34%～+1.41%，三胜三负。只有三组独立长度seed、每组两次重复，不给出普遍性能保证。一次legacy停滞单独保留，表中为正常完成运行吞吐，不包括失败的未完成block，也不表示稳定性全部通过。

72/72个计时A计划均接受且无fallback，每步14～37个真实W跨过F/B frontier；F/B和每chunk W次序保留，通信FIFO与模板相同。重复运行的同输入计划逐项相同。现有模型预测的合并吞吐改善约1.83%，没有兑现为本机实测收益；预测不替代GPU吞吐。详情见 [summary.json](summary.json)。

144份template/A计划通过生产依赖/身份验证及W保留范围检查，576次既有有限credit CPU重放排空，全部rank计划一致；正常完成运行均无跳步/NaN。A的三卡allocated峰值为30.102/26.074/26.224GiB；与legacy同组同卡的峰值变化为−0.0281%～+0.0300%，与template为−0.0331%～+0.0331%。reserved峰值见summary，各次分配器缓存可能不同。该实测范围内峰值基本相同，不替代完整owner合同、任意配置+5%或长期泄漏验收。依赖/协议复核见 [plan-checks.json](plan-checks.json)。

## 独立数值对照

相同24层配置、seed11输入、初始化seed1234、LR=1e-3和原cosine16步轨迹，三方法均在第6次optimizer更新后通过 `--exit-interval 6` 结束。统一采用既有FA的确定性归约，完整tensor字节SHA256比较，不抽样、不放宽容差。template和A各18个rank/step对legacy报告，七类tensor共7848次比较：各自882个更新前参数/882个完整梯度/882个更新后参数、636个stage output、530个dx、106个logits和6个loss，全部相同。A六次均接受且真实移动W，三rank计划一致；详见 [numeric-summary.json](numeric-summary.json)、[check_hashes.py](check_hashes.py)。检查的是训练模型参数，未另保存Adam的FP32 master/m/v状态作逐位断言。

该诊断增加CPU复制/哈希及确定性FA，不用于吞吐/显存验收，也不解决历史默认FA自重复数值门槛。附带的 `build_runtime_plan` CPU计时在第5/6步三rank中：template中位4.08ms、A中位26.97ms，额外约22.89ms；正式吞吐实验中A/template平均每步多21.88ms。量级接近提示同步规划值得进一步隔离，但这个函数计时不包含后续validation/plan hash/all-gather/publication，且诊断有复制/哈希干预，不能据此宣称已证明全部差异来源或异步预取能够实现多少收益。

## 模型、输入和环境

模型取自仓库 `examples/run_dspp_multisample_chunk_compare.sh` 和 `examples/run_dspp_infinipipe_compare.sh` 的长序列配置：24 层，hidden=2048，FFN=5504，MHA 16 heads/head dim=128，PP3/VPP2，各虚拟 stage 4 层，TP/CP/DP=1，BF16，FA，RoPE，SwiGLU，RMSNorm，零 dropout。max position/sequence=16384，固定 chunk=2048，microbatch=4/global batch=16。合成 token ID <32000；复用现有 Mistral tokenizer，实际 vocab=32005、padding 后32128；输入/output embedding 默认共享，rank0 有 embedding/head 参数副本，三 rank 日志参数计数为536381440/404783104/404783104。这是原实验缩放模型，不是完整 Mistral-7B。

使用原 `InfiniPipe-code/scripts/generate_dspp_compare_lengths.py`，其 sampler 为 `numerical-simulator/scripts/sample_infinipipe_seq_lens.py`：按 InfiniPipe Figure 2 的桶样本比例、桶 token 比例及 global_mean=4096，在桶内用 beta 分布采样。seeds=11/19/29 各抽1024个、min=512、对齐256；对齐后 >16K 的样本直接丢弃，保留其余顺序，不截成16K。它是论文桶分布模型的 ≤16K 条件分布，不是真实语料实测，也不代表完整长尾。三个 seed 保留978/987/978条；每方法运行只用前256条，其中前64条用于warmup，第65–256条计时。具体桶数、均值、最大长度和 SHA256 见 [input-summary.json](input-summary.json)。使用已有 dataset builder 的 `infinipipe` token 模式；next-token 目标比原始 document 少一个 token。

三张 L40（46068MiB/卡，约45GiB）；GPU0↔1 为 PIX，GPU2↔0/1 为跨 NUMA 的 SYS，没有 NVLink。临时隔离容器基于本地 `nvcr.io/nvidia/pytorch:26.01-py3`，torch2.10、FA2.7.4.post1，复用现有 venv 和只读依赖，不下载、不修改原容器。环境和起始工作树见 [host-environment.json](host-environment.json)、[baseline.json](baseline.json)。

## 比较和计时

三个方法分别为旧 DSPP `legacy`、原模板通过 P1 接口 `template`、运行时生成并执行的严格 A `a`；A 固定合并 F/B 和每 stage/chunk 的 W 次序，仅移动 W 相对 F/B 的位置。固定 chunk，无 dKV 拆分、无预取，不启用 B/C。

对三组相同 workload 先独立运行 legacy，用现有 task timeline 采第5–16步F/B/W，再用现有 `DsppExecutionCostModel.from_records` 冻结 profile。108个rank/step文件、12312个事件；全部比较开始前已冻结，见 [profile-provenance.json](profile-provenance.json)。校准和被测形状相同，因此这只是同 workload 的成本覆盖，不是 P2 或未知形状预测验收；P2是A/B/C共同的成本/显存基础设施，不是继续改进A的阶段。

每seed/方法两次重复，方法顺序轮换，共18个进程组，各16步，前4步warmup、第5–16步连续计时。三方法模型初始化seed=1234、输入和optimizer相同。主指标为计时block的有效目标token总数 / 三rank中最大的block wall time。开始前barrier及CUDA同步，最后同步；包含取数、GPU materialization、同步规划/验证/计划核对、通信、前后向、W、optimizer及区间内普通训练循环开销，不扣CPU规划耗时。不采 per-task event，不复制数值张量。block开始重置peak，记录allocated/reserved峰值；reserved包含warmup后已有缓存。NVML每秒采样仅作环境记录，不能认证瞬时峰值。

实际执行的原始入口副本为 [bench_entry.py](bench_entry.py)，独立数值诊断副本为 [hash_entry.py](hash_entry.py)，二者硬编码 `/workspace` 为临时容器挂载；精确命令、driver和全部日志保存在原始目录。数值诊断使用既有FA确定性归约并逐tensor保存完整字节的SHA256，包含各stage的output/dx、logits/loss、梯度和optimizer更新前后参数；不用于性能/显存结论，也不修改生产kernel。

## 复核与边界

[analyze.py](analyze.py) 对保存结果独立检查有效token与原始document、跨rank计划、A/模板输入绑定、任务集合、F/B次序、每chunk W次序、W在其B之后及最终通信FIFO，按完整18次运行汇总wall吞吐。应在原始目录运行；归档副本供审阅，不包含全部逐步原始plan。

本轮只认证测试范围内的PP3运行及吞吐现象；不作为PP4/8真机验收、长期训练收敛、完整owner显存合同或任意输入性能保证。历史默认FA自重复数值门槛仍单独保留；确定性诊断不替代默认FA验收。

## 成本接口复核发现

已有 `DsppExecutionCostModel.from_records` 按 `(physical stage, F/B/W, signature)` 聚合，`F0/F1` 等统一取首字母，运行时预测也没有virtual chunk字段。本轮rank0有66对同signature的分chunk耗时：F1/F0的比值中位数为1.348，W1/W0为1.289。同signature命中现有profile并不表示区分了embedding/head边界；这是源码和原始计时共同支持的限制，不是吞吐下降的因果归因。另未提供显式P2P profile，沿用25GB/s、20µs的fallback，不称为实测链路成本。见 [profile-boundary-diagnostic.json](profile-boundary-diagnostic.json) 及 [check_profile_boundaries.py](check_profile_boundaries.py)。本轮保持生产实现/profile冻结，没有中途调参追求正收益。

## 单独保留的停滞

第二轮seed19的legacy首次尝试在iteration12日志后不再更新，三GPU持续100%占用、功率约100W。保存三rank线程堆栈后手动SIGTERM终止，进程组总耗时242.12秒；原始目录 `failures/bench-repeat2-seed19-legacy-attempt1/`，含完整日志和失败退出码。失败没有覆盖、没有混入正常完成block的吞吐分母。只补跑一次相同配置/输入legacy，16步正常完成；不据此称为修复。

[stall-current-threads.json](stall-current-threads.json) 摘录最后顺序dump的当前线程，rank0在构造CUDA attention长度元数据的 `torch.tensor(..., device=cuda)`，rank1在 `wait_for_send_launch`，rank2在 `wait_for_sender_ready`。首次信号dump还包含继承worker环境的DataLoader子进程，输出交错；后续通过PPid筛出torchrun直接worker再逐rankdump，映射见 [stall-workers.json](stall-workers.json)。这只确认等待现场，没有捕获消息key/CUDA stream关系，根因未确定。本次legacy没有运行A规划器；不把该失败归因于A依赖图，也不将全部稳定性验收记为通过。

## 复跑

原始 `commands.json` 保存校准、失败尝试、全部比较及独立诊断的命令和环境；[driver.py](driver.py) 是执行driver的最终副本，包含一次失败后的续跑逻辑，`commands-at-stall.json` 保留失败时的原始命令表。driver使用历史memory配置作为参数起点，实际展开后的完整torchrun参数以commands为准。复跑应使用新的输出目录，复用冻结数据/profile时保持只读；不要让独立入口的output环境变量指向本次原始目录。归档脚本的 `/workspace` 路径对应本机 `/home/cuiym` 挂载。

[commands.json](commands.json) 为最终精确命令表，[raw-manifest.json](raw-manifest.json) 绑定原始目录422个文件的SHA256，[closing.json](closing.json) 记录起点六个生产/参考入口哈希与HEAD均不变。归档脚本副本语法编译和仓库 `git diff --check` 通过；没有为本轮重新运行未改动生产代码的全量CPU套件，实际保存计划的依赖/协议检查已单独执行。本轮只新增验证脚本/报告材料并维护实现计划，没有修改生产调度、分解算法或kernel，没有commit/push；临时容器已移除，三个原训练容器保持原样。

本轮增量diff位于原始目录 `performance-repository.patch`（仅本证据目录）及 `performance-plan.patch`（外部实现计划）；不把此前P1未提交改动混入本轮性能增量。
