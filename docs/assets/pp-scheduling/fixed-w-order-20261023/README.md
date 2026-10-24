# 固定每个模型 chunk 的 W 顺序

按用户要求收紧配置 A：每卡每个模型 chunk 的 W 保留原模板 FIFO 顺序，
继续移动 W 相对 F/B 的位置填空泡。不同模型 chunk 没有额外的全局 W 顺序边。
这里的顺序是模板的 B→W 产生顺序，不能替换成 forward microbatch 编号排序。
这项策略修改已同步到设计文档 §2.2/§4.1、实施计划 §3.2/§4.2/§12.5。

基线仍为 `cuiym-dspp` / `9550175c1d14ab309e96f090142e105b1a8501fd`。
保留本轮前未提交的 P1/定位改动；没有 commit、push 或 reset。

## 实现

- `dspp_plan.py`：A 构造图增加同 stage/chunk 的参考 W/W 边。
  验证器同时约束 W 次序与对应 B 次序一致，独立反序列化也执行检查，
  避免外部计划绕过顺序约束。模板在同一 chunk 中的 W 与 B 顺序相同。
- `test_schedule_plan.py`：PP3/4/8 同 chunk 交换 W 拒绝（含序列化），
  跨 chunk 的独立 W 交换可接受，36 组异构成本保留 W 次序；
  原往返/依赖/协议测试同步检查每 chunk W 次序。
- `test_plan_task_execution.py`：实际 TaskBatch/WeightGradStore 的 CPU 三步
  legacy/template/A 对照收紧为 `atol=rtol=0`。
- 本轮没有修改执行循环、通信协议、任务 key、分解算法、optimizer 或 kernel。
  keyed store 继续用于任务身份核对，没有因 FIFO 约束而撤掉。

## 验证结果

| 检查 | 结果与范围 |
| --- | --- |
| 完整 DSPP CPU 套件 | 145 passed / 243 skipped；跳过项不计通过 |
| 冻结 027 八例，PP4/8 | 最终顺序、makespan、owner 峰值与归档一致；22 个 manifest 文件哈希未变 |
| PP3 MHA/GQA/MQA，原 LR=.001，三步 | legacy 自重复、template、A 对旧 legacy 快照共 81 个 rank/step 报告，七类比较全部零差异 |
| A 实际接受 | MHA/GQA 各 3/3 步，MQA 2/3 步；8 步均改变 W 相对 F/B 位置，保留 F/B 与每 chunk W 次序 |
| MQA fallback | 第 1 步 `unreliable-compute-cost`，执行模板，不计作优化接受 |
| 原失败计划恢复 W 顺序 | 保留 W 插入位置，仅恢复同 chunk W 次序，原 LR 三步 9 个报告全部零差异 |
| MHA 默认 FA 独立显存对照 | legacy/A 各三步，allocated/reserved 峰值及逐步结束 allocation 全相同 |

数值对照统一启用既有 FA `deterministic=True`，原容差 `.004/.01/.02`、
相同数据/seed/冻结 profile、原 Adam 和 BF16 更新；没有降 LR 或放宽门槛。
七类为 params_before/params_after/gradients/outputs/dx/logits/loss。
GPU 配置沿用旧 PP3/VPP2 六层、hidden512/FFN1024、Hq4、chunk64、capacity256、
microbatch4/global8，head KV 为 4/2/1。profile 与参考 tensor 快照只读使用旧目录。

针对原失败样例，`repair.json` 逐步列出了交换回来的两个 W key。
它仍是手工外部计划，经实际 `dspp_plan=` 输入执行，没有作为 A 构造器的性能接受。
该实验说明：**这份样例恢复累加顺序后，原第三步数值失败消失。**
旧同 chunk 换序计划的失败记录不改写，也不由三步结果推断长期收敛。

独立 memory pass 使用默认 FA，没有数值 tensor 复制钩子。三卡 allocated 峰值
为 749.662/130.086/130.112 MiB，reserved 峰值为 796/144/144 MiB。
此短配置不是完整 owner 合同、总显存 +5%、长期泄漏或吞吐验收。

## 异常与未完成项

首次 MQA A 首步长时间没有进展，未生成比较报告，约 150 秒后手动中断。
`hkv1/a-stalled-attempt1/train.log` 保留完整日志；加 preflight/堆栈记录的重试
三步通过，三卡首步 fallback hash 一致；再做一次原入口、无诊断包装复跑也通过
（另 9 个 rank/step 报告全部零差异）。两次重试通过不解释首次停滞，原因仍待查，
不能记为已修复。包装代码仅存在原始材料目录，未进入生产代码。

默认 FA 的既有 legacy 自重复数值失败没有解决，见前期报告；本轮显存 pass
没有做默认 FA 数值验收。长期多 seed 收敛、完整 owner/更长协议与泄漏检查仍待补。
PP4/8 仅 CPU 验证，没有真机验收；P2–P6 未实现。
最初两个新增 CPU 样例因测试候选位置选取错误而失败，修正后全套通过，初始日志保留。

## 可复核材料

本目录保存 `gpu-summary.json`、`repair.json`、`research-crosscheck.json`、
`archive-integrity.json`、`cpu-suite-final.log`、`commands.json` 和哈希清单。
完整日志、逐步计划与比较报告位于
`/home/cuiym/tmp/dspp-fixed-w-20261023/`；`evidence-manifest.json` 绑定原始文件内容。
原失败及定位材料分别保持在 `/home/cuiym/tmp/dspp-p1-20261023/` 和
`/home/cuiym/tmp/dspp-w-diag-20261023/`，本轮没有写入这些目录或历史实验归档。

`commands.json` 保存逐次 torchrun 参数和环境，包含被中断的第一次 MQA。
`run_gpu.py`、`resume_gpu.py`、`repeat_mqa.py` 是本轮顺序执行的 driver；
`probe_entry.py` 是第二次 MQA 的诊断包装；`summarize.py` 对结果执行断言。
它们均保存在原始目录并由 manifest 绑定。
CPU 在相同 Python3.12/torch2.10 依赖环境、隐藏 GPU 后执行
`python -m pytest -q tests/unit_tests/dspp`。
GPU 使用临时本地 26.01 镜像和现有依赖只读挂载，不下载或修改旧容器。

增量 diff：`/home/cuiym/tmp/dspp-fixed-w-20261023/fixed-w-only.patch`；
设计文档 diff：同目录 `fixed-w-design.patch`；P1 当前完整 diff：`p1-with-fixed-w.patch`。
