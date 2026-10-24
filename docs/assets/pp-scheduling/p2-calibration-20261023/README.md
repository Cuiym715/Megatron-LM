# P2 首批成本与显存基础设施，2026-10-23

P2 已开始实现，尚未整体验收。它是 A/B/C 共用基础设施；本次没有实现 P3 选择性 D/H、算法 B/C、新 kernel 或异步预取。已有 P1 工作树保留，未 commit/push。

前轮 A 性能简记：三卡 24 层模型、基于 InfiniPipe Figure2 的条件长度采样，legacy/template/A 为 13163.98/13033.99/12927.43 有效 tokens/s，A 相对 legacy −1.80%。完整结果保留在相邻 `a-performance-20261023/`，本次不重复该吞吐矩阵。

## 实现

- `dspp_calibration.py`：versioned 完整 stage 身份，区分物理/虚拟 stage、层数与 embedding/head、模型/硬件/软件/kernel、执行模式；F/B_fused/D/H/W 样本隔离，复用既有 exact→插值→回归接口。新增 capacity、Σq*h 特征；W 使用 dense capacity，缺 H 明确 unavailable。新版 profile 不能经旧聚合 API 读取。
- 完整 stage 的既有 after-dx/H 任务边界直接进入 timeline；没有改变执行顺序或拆分算法。A 可加载新版 profile，支持 CPU 调用方传入冻结的完整 stage contexts；新版路径的 P2P 缺方向/重叠/大小覆盖时回退同输入模板。旧 profile 维持原兼容语义，不能自动获得虚拟 stage 精度。
- `calibrate_dspp_stage.py`：冻结 profile，整档 capacity/history band/packing 留出，报告各 op 误差与覆盖率。非精确形状尚无认证 kernel 区域，统一低置信；不会据统计误差悄悄增加算法接受门槛。P2 扩展特征使用非负 ridge 拟合，避免直接截断无约束系数破坏抵消项；旧五特征拟合保持兼容。最终拟合器版本见 summary。
- `profile_dspp_p2p.py`：有向 NCCL CUDA dispatch-to-completion 实测，区分 isolated/指定 GEMM 并发；严格查询不将未测范围 clamp 为可信成本。
- `dspp_memory.py`：按完整底层 storage 去重并记录 offset/view/owner；遍历闭包、autograd saved tensors、KV/H 与通信引用。采集 allocator allocation/free history 和 compute-stream 完成事件；提供不依赖预测毫秒数的存活峰值及总容量/动态增量双层合同接口。测量不完整时拒绝 certified 合同。

## 少量验证

机器仍为 3×L40，仅 PP3 真机。工程标定使用 6 层 hidden512/FFN1024、MHA4/head_dim128、PP3/VPP2、BF16，输入长度 193/129/33/17；此小形状只检查接口与真实边界，不冒充真实分布性能实验或原 24 层模型的成本。

- capacity64/128 × fused/after-dx 四个三步运行；第 2、3 步标定。共 1332 个真实 stage 样本、30 个 stage/path 组：F408、B_fused204、D204、H108、W408。
- P2P 一次短探测：6 个有向链路 × 2 字节大小（65536/131072）× 2 条件，每点 2 次预热、5 次采样，共120样本/24点。测量包含 dispatch；compute 条件为独立 stream 上8次 BF16 1024² GEMM，不能推广为任意训练重叠的硬上界。
- owner 审计两个三步运行。首版审计器递归闭包暂存 tensor 至 cyclic GC，造成生命周期干预；已显式释放临时引用并增加 GC 关闭下的弱引用回归，只补跑一次。旧记录保留，owner 比较只用 `c64-split-owners-fixed/`。修正后三卡 allocated watermark 为750.097/130.144/129.916MiB、reserved796/144/144MiB；三卡九份 owner JSON/allocator trace 保留。
- CPU 回归101通过、10个GPU专属跳过；最终严格 profile/runtime 接线86通过（与前项重叠）；正确配置现有 pinned split cache 后计时/region 回归15通过（含重叠CPU项）。首次环境未指向缓存的8个GPU失败保留说明；不是 kernel 回归。CPU新测试曾因仅一个 logical batch 不满足 SliceV warmup 失败，已改为合法的两个 logical batch。未放宽生产校验。

## 明确未完成

- 未知 capacity 留出全部正确标为低置信。最终整档 capacity 留出 median 相对误差：F约22%、B_fused约4%、D约7–9%、H约27–32%、W约19%。单 item 整类留出仍出现 H median约857%、D约187%的大误差，不能接受为未知形状预测验收。没有据此运行新的分解计划。
- owner 快照与 allocator 峰值仍有差额：暖态 rank0约26MiB、rank1/2约1.8MiB；这只是峰值差额诊断，不能直接当 workspace 硬上界。optimizer/常驻量目前由 allocator baseline 覆盖；最后真实用户、跨 stream 可回收时刻和瞬态 copy/workspace 仍需完善。容量接口尚未接替 P1 retention envelope，未声称动态+5%已通过。
- 融合/拆分使用同数据/初始化/训练配置采样，未新增逐 tensor 数值验收；不解决历史默认 FA 数值门槛。makespan 排名/拆分收益反转需后续真实候选计划，当前只做 op 留出。
- 异构机器需调用方提供真实逐 stage 冻结 contexts；当前默认 runtime 推断只在本机同型号卡条件下检查。PP4/8 只有既有 CPU 计划/依赖回归，没有真机验收。

原始材料与命令：`/home/cuiym/tmp/dspp-p2-20261023/`。最终 profile 为 `stage-profile-final.json`，留出为 `holdout-final.json`；早期 `stage-profile.json/holdout.json` 保留为拟合修正前记录。摘要及 SHA256 在本目录 `summary.json`。本轮增量 diff：原始目录中的 `p2-repository.patch` 和 `p2-plan.patch`。
