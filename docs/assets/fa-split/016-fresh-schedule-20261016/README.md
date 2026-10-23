# 016：当前实现的 F/B/W/H 与调度依赖采集

完成日期：2026-10-17（沿用启动时的20261016目录名）。**新采集已验收，可以开始条件性的CPU调度分析；本归档没有执行H重排或预测加速。** 研究结论与下一步统一见[外层plan §6](/home/cuiym/docs/v4/design/history-dkv/plan.md)，实现入口见[仓库文档](../../../dspp-history-dkv-split-implementation-plan.md)。

## 采集范围与结果

当前工作区 `9b9fc8f96` + 补丁，两项局部优化均开启；本轮生产kernel没有修改，不比较旧/新拆分实现。3×L40、PP3/VPP2、TP=CP=DP=1，6层、hidden4096、FFN8192、GQA32:16、BF16/d128，chunk2048。长度模式14336/4097/1537/257/2048，组内5条packing、每轮2组；seed1234、LR0、dropout0、无activation checkpoint。输入为合成token，分组按实际采样顺序，非每组强制包含五种长度。

| 验收结果 | 未拆分 | B后立即H |
| --- | ---: | ---: |
| 第6–25轮rank0 iteration均值 | 2019.230 ms | 2150.045 ms |
| 实际有效token / 总耗时 | 22443.06 token/s | 21077.56 token/s |
| 第26–30轮三rank F/B/W数量（各） | 594 | 594 |
| 非空H数量 | 0 | 360 |

立即H端到端慢6.48%；不代表合法延后H的结果。stage1/VPP0、Q2048/history12288的5个同形状样本，原B均值13.256 ms，拆分B10.680 ms、H6.314 ms。任务事件测完整计算流包络，不能视为内部kernel忙碌时间。样本保留iteration、stage、VPP和逐序列形状，不将变长batch之间的差异作为重复噪声。

两模式全部30轮的输入hash、各rank布局、loss配对通过。采样中覆盖90个E=2048无历史B，以及48个含历史的混装B。每个planned physical microbatch的F/B/W完整；所有非空H在对应B后立即执行。独立诊断确认198条历史贡献在消费者读取前完成，168个payload发送/接收配对，336个本地通信记录经launch correlation关联实际NCCL kernel。

计划有效轮次为30+30+3，共63轮。诊断初始化失败一次（0轮），再因嵌套profiler导出失败一次（执行3轮但排除），修复后有效3轮完成；实际执行66个训练轮次。正式60轮没有补跑。失败日志保留，不混入成本。

## 文件入口

| 文件 | 内容 |
| --- | --- |
| [manifest.json](manifest.json) | commit、正式/最终源码hash、软件环境、输入hash、全部raw文件校验和、失败与补跑记录 |
| [source/](source/) / [working-tree-final.patch](working-tree-final.patch) | 最终采集/解析入口和生产源码快照、当时工作区已跟踪改动；原始采集入口另存before-diagnostic-fix文件 |
| [summary.json](summary.json) | 验收、逐轮端到端、token吞吐、各rank设备区间和显存峰值 |
| [task-samples.json](task-samples.json) | 全部3924个正式F/B/W/H原始样本与序列边界 |
| [costs-by-shape.json](costs-by-shape.json) | mode/stage/VPP/kind/逐项Q-history-sequence_length分组统计 |
| `replay-{fused,after_dx}-{26..30}.json` | 10份CPU输入图：实测任务成本、数据依赖、原本地资源顺序、通信依赖和ordering信息 |
| [diagnostic-dependencies.json](diagnostic-dependencies.json) | 完整dx设备边界、梯度路由起止、host通信协议及任务包络 |
| [diagnostic-links.json](diagnostic-links.json) | H→实际消费者、slot→task、消息→CUDA launch→实际NCCL kernel关联 |
| [logs/](logs/) | 小体积训练日志、逐rank完成边界/输入hash/真实分组；失败日志单列 |
| [collector-tests.log](collector-tests.log) | 多轮选择、bh/all边界任务数、关闭采样不记录事件，5项轻量检查通过 |
| [cleanup.json](cleanup.json) | 新采集验收后删除的8份旧raw trace具体清单，约203 MB |

新原始材料保留于 `/home/cuiym/.cache/dspp-fa-split/collection016-20261016/runs/`：两模式timeline、保存的第26轮CPU输入、诊断trace/progress/路由事件。数据集位于同级 `data.{bin,idx,json}`。没有复制或删除其他项目数据，没有自动git提交。

正式任务样本不含profiler或任务内新增timer/同步/正确性钩子，事件在任务前初始化，轮末解析导出。端到端选取无任务采样的6–25轮；这20轮仍记录轻量的train-step设备边界、CPU输入引用、显存峰值，不能称为零观测开销。rank0 wall-clock、各rank设备区间、诊断耗时分别存储。

## CPU分析使用合同

- 保留一张卡一个计算资源，H不抢占；当前先固定F/B/W顺序，仅研究H移动。数据边与原资源顺序分开，不能把原H位置当成不可变数据依赖。
- B指完整stage输入反向，H包括历史attention重计算、GQA输出归约和历史累加。B→H；H必须在所有目标历史B开始前完成（保守整任务期限）。F连续历史、F→B、B→W、PP/VPP链均已导出。
- 当前VPP前向链为 `(0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(0,1)`，反向反转。相同物理rank的V转折是本地交付，其余有传输依赖。
- 依赖图的通信边**没有填零成本**。每个BF16 payload为2048×4096×2=16,777,216字节。每peer消息有固定顺序；send in-flight上限2、recv lookahead上限2；收到ready才能预取，credit与payload均可用才可发送，ack用于确认对端已发起；阻塞依赖期间仍由训练线程合作推进其他发送。相关生产逻辑在`schedules.py`，不能在重排时删除这些约束。
- 独立诊断使用已有progress线程，没有新增通信后台线程或调度同步。`progress/rank_*_iteration_0.jsonl`的0是诊断tracker局部编号，对应训练第3轮。记录Work检查/host等待，可描述推进约束，不能当作精确设备完成时刻。
- 消息通过同线程post顺序匹配CUDA launch，再用精确correlation id关联kernel；匹配数量、两端身份和本地CPU时钟范围均经过检查。本地CPU/Kineto对齐约有几十微秒差异，非精确跨GPU时钟校准。原始CUDA-event边界与Kineto时间不能直接拼成统一设备时轴。
- NCCL SendRecv驻留时间包括等待和争用，诊断又有profiler开销，不能作为固定通信服务成本。CPU入队也不是设备完成。未知host/通信成本应做条件假设及敏感性分析，不得默认为零或把空白视为可用H容量。
- 五个正式采样batch均没有模板padding。改变chunk/ordering后的padding通信、额外保留状态显存，以及并发改变后的任务成本均需重新处理。当前显存峰值只描述立即H。
- 图不含optimizer、数据准备等iteration外壳的完整成本模型；rank0 wall-clock不能直接与纯F/B/W/H图的makespan等同。重放时明确比较范围，分离固定外壳与未知项，不把误差静态贴到每个反事实结果。

## 复现入口（本轮已经执行完，无需再跑）

在准备好的同版本容器中，`/home/cuiym`挂载为`/workspace`，共用FA编译缓存，限定SM89。以新输出目录运行：

```bash
DSPP_COLLECTION_ROOT=/workspace/.cache/dspp-fa-split/collection-new/runs \
DSPP_COLLECTION_DATA_PATH=/workspace/.cache/dspp-fa-split/collection016-20261016/data \
  bash examples/collect_dspp_schedule_data.sh
```

默认正式两模式各30轮，诊断3轮；拒绝覆盖已有root。若仅诊断失败，补跑诊断部分，不执行整脚本覆盖/重复正式采集。诊断入口已经修复为只启动一个profiler，保留原通信函数身份。

离线验收和关联无需GPU：

```bash
python3 tools/summarize_dspp_schedule_collection.py \
  --raw-root /home/cuiym/.cache/dspp-fa-split/collection016-20261016/runs \
  --output docs/assets/fa-split/016-fresh-schedule-20261016
python3 tools/link_dspp_schedule_diagnostics.py \
  --raw-root /home/cuiym/.cache/dspp-fa-split/collection016-20261016/runs \
  --output docs/assets/fa-split/016-fresh-schedule-20261016
```

这两个命令只做校验、关联与输入导出，不计算H重排收益。下一步按外层plan §6开展CPU分析。
