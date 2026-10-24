# P2：拆分反向显存计数与 InfiniPipe 时间模型检查

本轮继续实现 P2，共用基础设施；没有实现 P3 的延后 H 执行器，没有替换在线时间模型，也没有改分解算法或 kernel。训练分支仍为 `cuiym-dspp`，HEAD `9550175c1d14ab309e96f090142e105b1a8501fd`。开始前的未提交工作保留，未 commit/push。

## 显存现在做到哪里

前轮只对不拆反向的 F/B/W 计算显存。本轮增加：

- **H 需要继续持有的数据**：Q/K/V、前向输出及 LSE 保留到 H；D 新建的 dO、每序列 LSE 副本及按128行对齐的 delta，从 D 保留到 H。按当前源码实际形状、物理/虚拟 stage 的层数和 MHA/GQA 头数计算。为防止 view 共享更大的 storage，Q 按完整 QKV backing 预留，相关项有意重复计费。
- **共享状态的最后使用者**：KV/pending 仍按 logical batch、stage、model chunk 只预留一份；最后一个 W 或 H 完成才结束计费，不能因为 W 完成就提前释放。
- **H 临时空间**：额外预留 query-head dKV partial、compact 输出、拼接及路由副本；沿用前轮保守 workspace。没有把 K/V 保留状态按 query head 错误展开。
- **依赖与容量**：接受显式 D/H 顺序作为纯 CPU 计数输入；检查任务完整性、D→H、所有后续 H→历史消费者，以及 F→B/D→W。多个未完成 H 的保留字节可以同时累计。最终仍使用既有 `PlanOwner` 和按任务顺序的差分计数，不遍历 tensor、allocator 或运行时计算图。

`dspp-counted-memory-v2` 显式绑定 `fa2-after-dx-v1` 和 after-dx 执行模式。v1 融合 B profile 保持兼容；不允许把旧 profile 无提示用于拆分。`--dspp-memory-profile` 可以用于现有 after-dx 路径，提交前容量检查和执行后峰值比较沿用原接口。

**真机仍是原来的 D 后立即执行 H。** 显式延后 H 和多个 H 同时待执行只完成 CPU 计数/依赖检查；执行这些新顺序属于 P3。实际 Plan IR 执行器仍只接受 F/B/W，没有绕过原计划验证器。显存增长上限按同输入融合 B 的模型基线计算，固定 workspace 对两者使用相同预留；不是“给拆分后的基线再免费加5%”。

## 验证

- CPU 相关回归 **110 passed**，覆盖 PP3/4/8 的峰值与独立慢算一致、多个 H 保留、W/H 释放、缺失/重复/过晚 H、MHA/GQA、packed delta 对齐及不足预算拒绝。PP8 小模板没有可直接移位的独立 H，因此多 H 测试使用明确的本地计数夹具；它不是已验证可执行的全局 PP8 调度。
- 一次 PP3、三步 after-dx 真机训练，6层 hidden512/FFN1024、MHA4/head_dim128、BF16、VPP2、capacity64，输入沿用前轮合成长度193/129/33/17，micro4/global8。9份逐卡记录及9份 optimizer 后整步峰值均未超过预留，无 skipped/NaN iteration；逐轮计划 hash 在三个 rank 一致。
- 暖态 rank0 预测约908MiB、实测最大750.10MiB；rank1/2 预测约217MiB、暖态实测最大130.02MiB。首轮 rank1/2 分别181.62/181.12MiB，也在预留内。首轮约62秒包含 split 扩展加载/编译，未当成性能结果。
- 没有新增 storage 分析工具；原 `dspp_memory.py` 内容校验和与本轮开始前一致。

保留限制：这些是限定小模型/输入覆盖的保守预留，不能证明任意模型形状或真实动态显存增量≤5%；没有重新进行完整逐 tensor 数值验收或 PP4/8 真机验证。

## InfiniPipe 原生时间模型的实测误差

复用了只读目录 `InfiniPipe-code/infiniPipe/profiler/logs/` 的8份历史1/3层标定文件，共864条原生验证点。重新执行现有 `CostModel.cost_comp`，复现归档中保存的预测值，最大差异不足 `3e-9` 秒。统计绝对/相对误差，不以训练集拟合 R²代替验证误差。

下表列时间最近的 `2026_9_5_7_4_llama-dspp-benchmark_comp/layer_3/result_3.json`。单序列拟合锚点256–4096 tokens；每类验证点数量有限，没有本轮重新测量。所有8份归档的逐点及逐组结果保存在原始目录的 `infinipipe-final/native-archive.json`。

| 原生验证类别 | 点数/方向 | F误差中位数 / P95 | B误差中位数 / P95 |
| --- | ---: | ---: | ---: |
| 无历史 batch/packing（含 batch=1） | 18 | 52.3% / 101.2% | 10.5% / 48.3% |
| 单序列切块（含无历史首块） | 30 | 6.6% / 48.6% | 28.9% / 48.3% |
| 带历史 packed | 16 | 9.4% / 24.7% | 6.8% / 19.3% |

这是**旧原生 profiler 的验证表现**，不能直接当作24层分区后的预测精度，也不代表 InfiniPipe 整体训练性能。归档没有完整版本身份，不能倒推当时所有 kernel/软件细节；可以确认的是当前时间公式与已保存预测一致。

## 在我们执行器上是否适用

另复用24层 hidden2048/FFN5504、MHA16、PP3/VPP2、capacity2048 的旧完整 stage 时间线，使用原性能实验的条件长度分布。seed11/19用于标定，seed29独立验证。标定5544条F/B样本，验证2664条；每个物理/虚拟 stage 单独拟合。只用无历史单序列锚点拟合系数，测试 packed/history 不参与拟合。B 与同 task 的 W 配对相加，和 InfiniPipe 的完整反向范围一致；不含通信或 optimizer。

| 拟合方式 | F误差中位数 / P95 | B+W误差中位数 / P95 | 非正预测 |
| --- | ---: | ---: | ---: |
| 原 `CompProfiler` 无约束二次拟合 | 86.7% / 1012.8% | 43.8% / 589.1% | F73/1332、B444/1332 |
| **本地原已存在**的非负二次拟合 | 22.2% / 72.4% | 5.1% / 19.9% | 0 |

非负拟合来自本地 `do_profile.py` 已有未提交改动，**不是原论文/原始作者流程**，本轮未修改该仓库。其余预测公式直接执行作者方法。负预测全部计入原版误差，没有过滤后再报告较好结果。原版模型在这组迁移数据上的失败不能泛化为论文结果错误：这里固定 dense capacity，但拟合变量是 valid token 数；锚点也只有767/1023/1279/1535/2048，和原生测量条件不同。

非负版本对训练集中完全没出现的形状，仅有每方向66条验证点：F中位24.6%、B+W中位9.8%。其余测试点虽然输入种子独立，形状可能已经出现；没有把全部2664点称为未知形状。该版本每次 `cost_comp` 同时预测F/B，CPU约24微秒（五遍中位数，含 NumPy 操作）。没有GPU求解开销或吞吐改进结论。

结论：InfiniPipe 的简单公式值得作为基线，但本轮数据不支持直接替换在线预测器。完整反向较好，前向及 packed/未知形状仍需校准；原公式没有独立 D/H/W 模型，也未验收64K外推。下一步应围绕实际执行路径补最少代表点，并检查调度排序是否正确，不扩大成通用预测工具或大规模实验矩阵。

## 检查入口

- 代码：`dspp_memory_counts.py`、`arguments.py`、`test_memory_counts.py`；评估脚本 `tools/evaluate_dspp_infinipipe_cost.py`。
- `summary.json` 汇总本轮验证；`memory-profile.json` 是实际运行的冻结profile；`infinipipe-native-summary.json` 保留8份原生归档的分组结果，迁移结果见 `infinipipe-transfer-summary.json`。
- 全部命令、原始输出、开始前快照和增量diff：`/home/cuiym/tmp/dspp-p2-split-memory-20261023/`。`implementation.patch` 相对本轮开始前工作树；`plan.patch` 相对本轮开始前计划。
- 开发期失败保留：首次PP8多H测试假定原模板必有可移位窗口，已改成明确的本地计数夹具；旧 `infinipipe-cost/` 是早期统计口径，不作为最终结果，原版最终误差以 `infinipipe-polyfit-final/` 为准。
