# P6：下一轮CPU输入预取与后台规划

复用已有seed11分布输入及原24层模型，默认注意力模式。已完成同步B三步对照、后台B两步及完整checkpoint恢复第三步、正常后台C三步、主动终止worker的C三步检查。先各测一次当前融合/拆分任务耗时，冻结本轮成本，沿用原显存与P2P配置。

| 检查 | 结果 |
| --- | --- |
| CPU/GPU输入材料化 | 在真机逐张量比较新旧路径的tokens、labels、position IDs、loss mask，一致。 |
| 后台B | 两步均采用worker结果；worker没有初始化CUDA，第二步规划37.2ms。 |
| checkpoint恢复 | 第2步保存完整模型、优化器和RNG后停训；恢复仅执行第3步，样本位置16–23，最终消费24条。 |
| 正常后台C | 连续三步采用worker结果。后两步规划70.5/50.3ms，在上一轮任务执行期间完成，优化结果等待0ms。 |
| worker被终止 | 首步提交C后终止worker；后两步各rank一致回退同输入原模板，完成三步，无停滞。 |
| 通信、状态及显存 | 所有运行的rank计划相同，通信/KV/H/W排空，无跳步或日志记录的NaN；实测显存均低于冻结计数额度。 |

三步共24条序列、38,888个有效目标tokens。数据直接读取 `/home/cuiym/tmp/dspp-a-performance-20261023/data/seed11/data`，不重新采样；长度来自已有InfiniPipe图2分布近似，token为原合成数据。模型24层、hidden2048、FFN5504、MHA16、BF16、PP3/VPP2、固定chunk2048、micro4/global8、seed1234，LR1e-3沿原cosine三步轨迹。

正常后台B/C各卡最高allocated为21.722/20.239/20.552GiB；步末约9.051/6.837/6.837GiB。新路径不提前创建未来GPU输入或激活。正常C三轮紧凑CPU输入加序列化metadata payload为198649/123379/89887字节；这个额度不包括已有DataLoader缓存、Python对象开销或worker解释器RSS。

C后两步各rank最长边界取包/发布/核对32.2/42.4ms；rank0材料化后提交检查20.3/33.9ms。首轮取包到接收核对约6.2秒，含spawn导入。GPU计算时间用两次CUDA事件记录，CPU/IPC/提交为单调时钟记录。未做统一吞吐对照，不把短测中隐藏的规划时长换算成吞吐提升。

输入线程是协调rank的唯一训练iterator消费者；一个未来包、一个spawn planner。worker只收metadata及冻结配置，GPU不可见、CPU/BLAS线程数为1。使用独立Gloo组在边界发布，训练主线程执行全部collective；当前轮消息匹配、credit、buffer和执行状态生命周期保持原语义。未就绪、过期、失败或积压时保留该输入已验证回退，已提交计划不接受晚结果。

P6最初真机运行中，Megatron模块导入触发了worker的CUDA初始化，隔离检查拒绝结果并回退。已在创建worker时屏蔽GPU可见性修正；正常B/C结果均记录 `worker_cuda_initialized=false`。旧回退运行保留在 `async-b-cuda-import`、`resume-b-cuda-import`。无CUDA或镜像版本不匹配的两次启动记录也保留，没有纳入上表。

范围：固定chunk、single-pass文档数据、TP=CP=DP=1。未来microbatch数量由独立计算器预测；重启用checkpoint的已消费位置重新读取未训练包。修正了原保存调用中未定义的iterator参数，以及当前PyTorch读取完整训练checkpoint所需的 `weights_only=False`。

本批正常计划H=0，含H的异步组合及evaluation切换未做真机检查；没有新增完整梯度/更新参数逐张量对照。统一吞吐和边界开销优化留P7。算法定义与kernel均未改。

[逐组可核对结果](gpu-summary.json)。完整命令、当前成本标定、模型/优化器checkpoint、输入身份与状态、故障运行和失败日志：`/home/cuiym/tmp/dspp-p6-20261024/`；增量diff为该目录的 `incremental.patch`。未commit/push。
