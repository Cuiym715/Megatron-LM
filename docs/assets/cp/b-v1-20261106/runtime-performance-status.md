# 当前版本运行与性能边界

2026-11-06：用户授权暂缓已知数值误差，推进完整 a–f；a 已冻结，本文件属于 b。

| 批次 | 结果 | 性能用途 |
| --- | --- | --- |
| baseline-longmix-v1 | KV首窗NCCL600秒超时 | 无有效吞吐，余15窗未跑 |
| runtime-probe-v1 | 全8rank完成更新1，更新2等待PP发送启动；重复faulthandler后SIGSEGV | 仅运行定位，不能归为算法根因 |
| runtime-probe-v2 | P2P调试＋更新2前同步下，全部rank完成3更新且有限、退出0；控制器旧25记录断言失败保留 | 仅诊断；不能证明何项避免卡住 |
| baseline-longmix-v2 | 仅首更新共同预热同步，KV仍300秒超时 | 无有效吞吐，未声称修复 |
| qo-pilot-v1 | GPU0新外部任务进入，标记污染并只停止自身作业 | 排除；不能据此判QO自身持续运行失败或计算加速比 |
| nccl-order-pilot-v1 | 启动前资源保护拦下，0个GPU作业 | 尚未执行NCCL假设，不算GPU运行失败 |
| nccl-order-pilot-v2 | 同一未执行对照的新输出身份，已上传待命 | 等待干净八卡；具体下一组命令见next-launch.json |

旧5批原日志、源码/输入身份、资源/PID证据保留于 [归档](runtime-timing-evidence-20261106.tar) 及同名解包目录；SHA256 `94c5776b969384889f64df2bba1105a1a4bededde43a15d7bfe593d6dcc355ca`，8,519,680 bytes，111元数据文件；未更改原失败。

下一对照基于[NVIDIA多communicator约束](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)和[启动顺序开关](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-launch-order-implicit)提出主机启动顺序假设，**不是已定位根因**。CPU已核验加载NCCL2.29.2/CUDA13.1，未初始化CUDA；对照开启隐式启动顺序、保留race fatal检测，双方后续比较须相同设置。源码、算术和输入不变；首轮非计时预热同步也明确保留，不能混称默认原生路径吞吐。

成功时先审阅完整25次更新和实际timed token，再开展配对矩阵；失败则根据新证据定向处理运行问题，不机械重跑、不返回数值根因排查。b收尾后继续c–f。
