# P2：接通计数式显存约束，2026-10-23

本轮交付是 **固定 chunk、融合 B 路径的显存容量接线**，不是完整 P2 验收。复用已有 ready 构造器的占用字节、剩余用户和释放计数；未扩展通用 storage/allocator 分析工具，未修改拆分算法、kernel 或实现 P3–P6。分支 `cuiym-dspp`、HEAD `9550175c1d14ab309e96f090142e105b1a8501fd`；开始前保存全部已有工作树文件及 SHA256，未 commit/push。

## 实现与使用

新增 `dspp_memory_counts.py`，输入一个冻结的 `dspp-counted-memory-v1` JSON：模型/软件/设备和 capacity、输入覆盖范围、逐卡常驻/临时空间预留、各虚拟 chunk 每任务激活/W 字节及每 token KV 字节。例子见本目录 `memory-profile.json`，它只适用于下面的小配置。

- 激活及 attention 保存量：从 F 开始预约，对应 B 完成释放。
- W 所需数据：**从 F 起预约**，对应 W 完成释放；不假设这些数据到 B 才产生。
- 共享 KV/pending：每个 logical batch、每个 stage/chunk 只建一条记录；在首个 F 一次预约整组 token，在最后一个 W 后释放。这个保守规则同时避免重复按每个消费者计算整份缓存、也不依赖当前 segment-zero 的隐含释放假设。部分缓存与激活/W 可能共享 storage，系数允许保守重复预留，不声称逐 storage 精确预测。
- 通信：复用已有 lookahead/depth 计数，并额外保留所有 F/B 已发布 payload 的空间，含 padding 与本地 V 转折。`ready_sends` 不受 inflight depth 单独约束，不能只计算 inflight 池。
- 临时空间：每卡固定预留，与任务数据分开。物理容量减去常驻、保留量和固定预留后形成上限；动态占用同时受同输入模板预测值的 +5% 限制。

运行时使用 `--dspp-memory-profile PROFILE.json`。容量版本进入全 rank 核对的 PlanBinding；A 的 ready 构造收到真实的 owner/容量参数。最终以每卡任务位置做线性 `order_owner_peaks` 检查，不依赖预测毫秒，也不调用原审计版 `conservative_peak()` 的逐边界全记录扫描。旧无 profile 路径不变，P1 W retention envelope 继续保留。

`--dspp-memory-report-dir DIR` 可保存逐卡预测/实测报告。启用计数 profile 后，每轮重置一次 allocator 峰值；任务与通信全部排空后比较实际 allocated 峰值并全 rank 协调报错。提交前还核对设备、实际容量、常驻预留及当时可用容量。所有这些查询均在迭代边界，不遍历 tensor、不增加每任务同步。外部输入计划也经过相同容量检查。报告期间没有声称额外 all-gather/文件输出没有运行开销。

明确拒绝 profile 的布局/软件不符、未覆盖输入、负数/缺失大小、原模板本身超预算、自动 chunk 和 split-B 模式。后者要等 P3 的任务级 H 执行接口；本轮不会用融合 B 的释放规则冒充延后 H 支持。

## 系数来源与范围

复用上一轮 `c64-split-owners-fixed/owners` 的已修正记录，不重跑 owner 采集。F/W 单份预留来自暖态记录中相应任务新增可达 storage 的最大值，乘 1.25 并向上取整到 MiB；KV/额外 assembled-KV 按源码张量形状、真实 MHA 宽度计费。常驻预留来自暖态 allocator baseline 加 1 MiB 后向上取整，已包含优化器和输入材料化，不再逐 tensor 重建静态内存。

workspace 首版保留整轮 trace 中新增 allocation 的同时存活峰值，再乘 1.25 取整，三卡为 94/27/27 MiB。它包括部分已在任务系数中计入的状态，**刻意偏松**；不是把 owner 峰值差额当未来 workspace 硬上界。单份 F/W 的经验系数和这份临时空间预留仅在本次小配置做了运行核对，尚未认证任意形状/候选的硬上界。

覆盖范围从旧原始记录提取；packed K 总长度采用同一物理任务所有 item 的 K 长度之和，不能用最长单条序列长度替代。精确数值与证据见 `memory-profile.json`、`coefficient-evidence.json`，一次性构造脚本在原始目录 `prepare_profile.py`。

## 少量验证

- **CPU 102 passed**：新增 11 项计数验证及既有计划/标定/执行接口/W 队列回归。包含 PP3/4/8 的依赖、容量、独立慢算对照、W 延后导致容量拒绝、共享 KV 单份记录、profile 版本变化、packed 范围与未实现 split 模式拒绝。没有 GPU skip 冒充通过。
- **PP3 真机仅两次成功运行**：template/A 各三步，复用前轮 indexed 合成输入与成本/P2P profiles；6 层、hidden512、FFN1024、MHA4/head_dim128、BF16、VPP2、capacity64、microbatch4/global8，长度原始模式193/129/33/17。`num_query_groups=1` 是未开启 GQA 时被忽略的参数，实际按 MHA4 计费。此配置仅作工程核对，不是原24层模型吞吐实验。
- 18 份 stage/iteration 容量报告全部覆盖观察 allocated 峰值；另保存 optimizer 更新后的整步峰值，18 份也全部在预留内。各轮无 skip。暖态预测总量：rank0 约901 MiB，rank1/2 约209–210 MiB；观察峰值约749–750/129–130/129–130 MiB。估计明显偏保守。
- A 第1步因成本形状未覆盖而正常回退；第2/3步真正接受 A，分别移动全计划29/28个 W 位置。逐卡计划 hash 一致、与 template 输入/绑定一致、F/B 与各 chunk W/W 次序和任务完整性检查通过。没有把首步回退计成 A 优化通过。
- 独立 CPU 开销诊断：PP4 的768/6144个节点完整显存检查约1.37/11.31ms，PP8 的1536/12288个节点约2.71/22.39ms（各30次均值）。包括准备位置表和报告，排除模型初始化、调度求解和 makespan 图评价；不是端到端性能结论。显存检查仍有线性费用，不在每个 ready 尝试中全量扫描。

首次真机尝试在启动任务前因 profile 把 packed K 总长上限误写成193而被三 rank 一致拒绝；同时旧记录存在三个 item 的打包，初稿 item 上限2也不足。修正的是本轮 profile 的来源提取，未放宽生产校验。原 profile、命令和拒绝日志均留在新原始目录。另保留首次临时镜像版本不符导致依赖收集失败、测试 wire fixture 不完整、以及 NaN 在增长比例校验前就进入哈希计算的校验顺序问题；最终使用与前轮相同的26.01镜像通过。

## 未完成与下一步边界

这次接通的是 fused-B 的经验保守容量约束。模型动态 +5% 相对于**模型中的模板动态量**，不等于已证明真实动态显存 +5%；系数和 workspace 仍偏松。多 pending H 的大小、共享关系及最后用户需在 P3 接口下接入同一计数框架。未知形状成本误差、任意形状容量上界、长时间训练、PP4/8 真机均未验收。没有新增逐 tensor 数值对照或吞吐矩阵。

原始材料：`/home/cuiym/tmp/dspp-p2-memory-counts-20261023/`；含开始前快照、命令、完整日志、冻结 profile、18份容量报告、逐卡计划及整步峰值、CPU 开销、`summary.json`。本轮可检查增量为该目录 `implementation.patch` 和 `plan.patch`。旧实验目录只读。
