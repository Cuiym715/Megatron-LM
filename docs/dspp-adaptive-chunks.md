# DSPP 端到端执行入口

历史版本保存在提交 `7b13bda9e`。当前使用普通 `pretrain_llama.py` 入口，不再叠加历史实验 entry/runtime monkeypatch。

- `--dspp-schedule template`：SPP packing、规则 ordering、VZB 模板。
- `--dspp-schedule b`：供给构造，然后最长序列尾部前沿 D/H 搜索；并列最长序列分别处理，每个 stage/VPP/序列接受一次后向前开放，最多接受32次。
- 固定容量使用 `--micro-seq-length`；B没有2048限制，成本和显存profile必须对应实际容量。
- CP1自动容量使用 `--dspp-prefetch --dspp-chunk-model FILE`。从模型域端点和当前输入的切片数变化点生成候选，用真实 packing 块数粗评。只对前两名做完整精评并复用其 packing，选中容量最多运行一次 B。`--dspp-chunk-fixed C` 可固定容量并复用同一执行路径。
- 供给构造仅保留C++实现，通过 `--dspp-planner-backend cpp`（默认）接入固定和自动路径。
- 删除旧 `--dspp-auto-chunk-size`、旧A/C命令入口、canonical attention适配器、逐任务时间线/显存审计/进度探针和一次性实验脚本。旧参数将由解析器拒绝，不会静默选择旧实现。

## 时间模型

执行使用原生BF16 attention；自动容量模型的execution为 `native-default-bf16-v1`。已复用原生2K/4K F/B/W、历史原生2K D/H、局部跨容量数据及现有P2P，导出 `profiles/dspp-native-pp8/model.json`。通过 `--dspp-chunk-model profiles/dspp-native-pp8/model.json` 加载。详细评估优先用全部944个实测键，未命中才使用拟合；局部数据只参与拟合，不冒充PP8实测。旧canonical数据未混入。原生模型已完成CPU接入，仍需目标PP8大容量成本、H延后显存与GPU验证，具体见 [补测清单](dspp-native-profile-plan.md) 和 `profiles/dspp-native-pp8/profile-plan.json`。

## 保留逻辑及原因

- 图依赖、FIFO、每chunk W累加顺序：定义合法执行次序，删除会改变训练结果或造成通信死锁。
- H插位后的显存可行性计算：这是调度目标约束，H延后会延长KV/梯度存活期；不是重复输入审计。
- 预算截止与未及时完成时的同轮模板回退：让训练不等待后台规划。
- 异步结果的轮次绑定、分布式失败传播：避免执行上一轮计划或让其他rank永远等待。
- 初始化时检查真实算子支持条件和模型执行模式：BF16/head-dim128、CP分片整除、TP/VPP支持范围不能通过删除断言获得实现。
- 单元测试及离线profile/模型导出/内核构建工具：用于验证修改和生成算法必需的成本；不在训练热路径运行。

仅保留完整更新时间、有效tokens、计划选择/回退等轻量性能信息；不记录逐任务CUDA事件或完整执行计划。

## 正式计时

`examples/run_dspp.sh` 是普通训练入口，模型、数据和profile参数由调用方传入。
需要连续窗口结果时，用torchrun运行 `tools/benchmark_dspp.py`，传入相同训练参数；设置 `DSPP_BENCH_OUTPUT`、`DSPP_BENCH_WARMUP=5`、`DSPP_BENCH_TIMED=20` 和 `--train-iters 25`。只在窗口边界同步；输出所有rank耗时、最慢rank有效tokens/s和显存峰值。不保留输入或完整计划，不做全参数比较。

仓库外旧a4–a6冻结目录不属于本次备份commit，未改写其历史证据；不能再用其entry/runtime注入脚本启动本版本。

固定容量B的静态配置、冻结profile和stage contexts随worker启动传入一次；每轮只传metadata、已构建ordering/counts及轮次时间，不再复制整份配置/profile到IPC。

## 本次验证边界

2026-11-08：纯CPU回归覆盖尾部候选、真实Plan依赖与回放、64/3058/4096容量、预算/显存拒绝、C++编译执行、H插位和常驻worker请求协议。运行 `python3 tests/unit_tests/dspp/test_algorithm_b.py`。Python编译、启动脚本语法和已删除模块引用检查同时通过。本机无PyTorch/CUDA，未执行完整PyTorch测试集或GPU训练；不得把这些CPU结果标为端到端验收。

## 小批次流水线空泡（2026-11-09）

真实任务不足以填满V流水线warmup时，空位只用于生成八阶段相位顺序，不产生F/B/W节点、占位张量或P2P消息。单个microbatch同样可调度；真实依赖未满足时自然等待。固定入口、预取worker和容量模型入口均传真实split counts，时间/显存模型与执行器使用同一份无占位任务计划。删除原 `_pad_dspp_v_split_counts` 及执行器的padding收发分支。

16条2K＋1条16K、固定4096的例子：12个真实块保持12个，每stage为72个F/B/W节点；全流水线数据消息由504条降为336条。不改变packing、真实F顺序、同序列反向顺序及每个模型chunk的W累加顺序。任务充足时，原有调度所有节点字段保持一致。

CPU检查：`python3 tests/unit_tests/dspp/test_slice_v_bubbles.py` 覆盖355组依赖图、120种有限收发缓冲组合、真实4096 packing及原调度一致性；尾部B回归同时通过。尚未进行此次修改的GPU正确性/吞吐复测。之前五组完成的实验快照和结果不改写。
