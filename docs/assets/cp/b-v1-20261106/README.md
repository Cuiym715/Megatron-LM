# 阶段 b 目标 CP2 数值与基线材料

## 2026-11-06 用户最新执行授权（优先于下文旧门槛）

继续的是**完整 a→b→c→d→e→f 总计划**，不是只完成 CP/b；a 已冻结，不重启。用户明确要求先使用当前存在数值误差的版本做性能，误差根因由用户之后统一检查。因此保留 v5/v6 的数值失败和原阈值，但**不再以消除既有 CP 误差或取得数值通过凭据作为后续性能及阶段转移的前置条件**，暂停首层 attention 定位，不伪造正确性验收。相关结果统一标为“当前含数值误差版本，数值未验收”。

先完成 b 的现版性能/显存/容量和模板/B，再接 c 两组端到端矩阵，随后 d chunk 优化、e CP 优化、f DynaPipe 切分探索；b 收尾后立即继续 c，不等待用户再次启动。已有误差同样不阻塞后续总计划中的性能探索；新改动仍记录数值影响。公共设置仍是完整 BF16 GEMM 归约＋确定性 FA2 backward＋exact p0，双方一致；保留输入、模型、有效负载和完整更新计时、NaN/Inf/跳过更新、OOM、资源/PID 保护。DynaPipe 原算法、DistCA graph 关闭及阶段顺序不变。有空闲卡且有就绪任务则连续接续，不以一批/一次初测作为全任务终点。

自动化提示已同步此范围；按本次用户“继续推进总计划”的指令恢复 **ACTIVE，每 10 分钟**，从最新检查点续跑。不能把历史 PAUSED 或数值失败重新当作停止/性能阻塞，也不能重复启动正在运行的控制器。


16层/H1024/FFN2816/Q8、PP4×CP2/VPP2、chunk2048、hybrid/baseline；生产与本机后端源码未覆盖。当前运行状态以[总检查点](../../../../../../docs/v5/design/chunk_size_selcetion/selector-runtime-20261106/next-checkpoint.json)及[CP报告](../../../dspp-context-parallel-execution-report.md)为准。此目录不是全部阶段完成声明。

- [源归档身份](source-manifest.json)、[全部608文件核验](current-source-verification.json)：隔离源来自旧最终已验证归档，逐文件一致，不恢复旧队列。
- [默认路径首轮失败](numeric-v1-failure.json)：实际FP32更新量门槛揭示更新后参数allclose不能替代更新量。
- [关闭BF16低精度归约控制](numeric-v2-CPU-attribution/result.json)：首轮全模型梯度误差0.257031%，实际更新2.696287%仍失败。原生梯度/前向/dx/loss检查均通过。
- [确定性FA2控制](numeric-v3-CPU-attribution/result.json)、[实际命令](evidence/numeric-v3-manifest.json)：MHA三轮原生与实际更新门槛通过；首轮完整梯度/更新一致，后续更新最大rank误差0.734453%。不声称连续优化器轨迹或默认FA路径已通过。
- `run_numeric_v4.py`：一次默认FA的KV自身复跑归因，以及MHA/GQA完整D/H数值。归因控制若有完整报告的master-only失败，保留失败，不认证默认路径；其余目标作业全部门槛须通过。`review_target.py`在批次/审计闭合后逐项复核。
- [实际32K数据CPU预检](actual-domain-preflight.json)、`run_numeric_v5_long.py`：single loader首8篇共88,064有效token，最长32,768、vocab512。六个单次完整更新填补实际域缺口；CPU元数据就绪不是GPU通过。
- [待执行基线合同](baseline-protocol.json)、`benchmark_entry.py`：按最新授权可用含误差版本启动性能；数值未通过如实标注，明确共用完整GEMM归约＋确定性FA2 backward。5预热＋20更新、配对窗口，包含真实训练/优化器开销；数值快照、审计/诊断及服务器CPU研究不混入计时。默认FA自身复跑也超出更新量门槛的边界保留。

远程实验目录`/workspace/experiments/dspp-cp-v1-20261106`，原始输出`/workspace/outputs/dspp-cp-v1-20261106`。各数值入口的`.py`版本与原失败均保留；`numeric_entry_v4.py`同时保存完整FP32 master/实际更新及原生梯度快照，真实H计数来自执行函数。每组前后核查资源，开始前3次跨60秒确认完整八卡，宿主审计核对PID/启动时间/控制器祖先；作业900秒超时，不终止外部进程。

运行更新：首个无调试KV正式窗NCCL超时，未得到有效吞吐。两次运行探针及后续公共预热条件对照见[总检查点](../../../../../../docs/v5/design/chunk_size_selcetion/selector-runtime-20261106/next-checkpoint.json)；`baseline-v2-protocol.json`只在首个非计时更新后增加公共同步，不改算法/输入，原失败保留。三更新P2P调试成功不作为性能证据，也不声称原生运行路径已修复。
