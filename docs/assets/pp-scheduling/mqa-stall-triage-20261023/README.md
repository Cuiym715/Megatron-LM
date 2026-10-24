# MQA 首次停滞：有限排查

结论：**没有定位根因，也没有足够证据认定是其他任务争用。**
保存的同输入计划没有发现静态依赖环或固定消息次序错误；真实通信等待与资源争用仍未排除。
按用户要求控制排查范围，不追加长时间 GPU 复现，不以两次重试通过声称修复。

## 已检查

1. 首次运行在 2026-10-23 本地时间 14:31:02 进入训练首步，14:33:33 手动
   SIGTERM 中断。日志最后的正常记录是输入 collate，没有首步比较报告；
   launcher 的 ChildFailedError 是中断的结果。distributed timeout=10 分钟，
   中断前未到该时限，所以“没有 NCCL timeout 日志”不能排除 NCCL 等待。
2. 首次与无包装重试的 torchrun 命令相同，环境仅输出目录不同。
   对成功保存的 template/A/无包装 A 各三步，共 9 份计划：三卡 hash 一致，
   生产验证器的任务/消息合同及无环检查全部通过。运行原有 CPU 协议模拟函数，
   lookahead/depth 为 0/1、1/1、2/2、4/1，36 次有限 credit 重放全部排空。
   首步 A 为 `unreliable-compute-cost` 模板回退，两次 A 三步计划完全一致。
   首次失败没有保存计划，这不是对首次现场计划的直接认证；CPU 模拟把 payload
   交接视为原子 rendezvous，不模拟 NCCL/CUDA stream 活性或真实线程时序。
3. 14:29–14:35 内核 journal 没有条目，未见 OOM/Xid/驱动错误。
   Docker 事件仅包含本轮容器的检查和中断，没有其他容器的 start/exec 事件。
   系统 journal 记录常规定时任务和 USB 网卡 DHCP 重试；本机测试容器使用
   `--network none`，没有证据把该网卡事件当作故障原因。
   Docker 事件并不记录所有既有进程或容器内部启动的子任务。
4. 当前三卡利用率 0%、各 14 MiB，无 GPU 计算进程。
   这是排查时的状态，不能反推停滞窗口。未找到 sysstat/atop/进程记账归档，
   首次也没有连续资源采样，因此不能确认或排除短时 GPU/CPU 资源争用。
5. 对照执行器等待路径：`wait_for_sender_ready` 等 Gloo ready，
   `flush_ready_sends` 受 credit/inflight 限制，`wait_for_send_launch` 等 ack，
   接着 `request.wait()` 建立 NCCL/CUDA 依赖。这些都可能表现为等待；
   首次未开启逐任务 progress/P2P debug/线程堆栈，无法确定卡在哪一个环节。
   初始化时的 device-ID/Gloo hostname 警告也见于通过运行，不能单凭它们归因。

## 停止条件与后续

本轮到此结束，没有新跑 GPU 训练，没有改生产代码或用降低通信并发绕开问题。
下次自然复现时，使用既有 `DSPP_GPU_PROGRESS_DIR`、执行器 P2P debug、
逐 rank faulthandler，以及带时间戳的 nvidia-smi/进程采样，保留具体任务与等待点，
再决定是否需要处理通信时序。当前保持“首步偶发停滞，原因未定位”的标记。

`saved-plan-checks.json` 和 `cpu-replay.log` 为本轮 CPU 检查；
`evidence-manifest.json` 绑定所用源文件、旧日志/计划与本轮只读系统检查。
原始脚本、系统日志在 `/home/cuiym/tmp/dspp-mqa-triage-20261023/`。
原失败及重试材料 `/home/cuiym/tmp/dspp-fixed-w-20261023/` 保持只读。
分支/HEAD 仍为 `cuiym-dspp` / `9550175c1d14ab309e96f090142e105b1a8501fd`，未 commit/push。

## 后续：按用户要求同配置再实跑一次

上文为有限排查阶段记录。用户随后要求实际重跑，已在运行前 GPU 空闲时，
用相同镜像/依赖、原 MQA torchrun 命令、同输入/profile/确定性 FA 设置运行，
只改输出目录，不加诊断包装。本次没有卡住，三步正常退出（code=0），
启动至退出约 45.5 秒；9 个 rank/step 报告七类比较均零差异，三步计划与此前
无包装重试完全一致。首步仍因成本置信度不足回退模板。
原始命令、日志、摘要及哈希在 `/home/cuiym/tmp/dspp-mqa-repeat-20261023/`。
临时容器已清理，没有生产代码修改。本次未复现，不将首次停滞改记为已修复。
