# P3：选择性拆分和延后 H

P3 已实现并通过下述手工计划验证。同一轮可混用完整反向 B 和 D/H；H 可在 D 后立即执行，也可移到依赖允许的后续位置。算法 B 的自动编排与拆分搜索尚未开始，属于 P4。

## 改动

- Plan v2 表达 F/B/D/H/W，记录 H 来源和所有历史消费者；缺失、重复、过晚 H、篡改依赖均拒绝。v1 模板/A 的序列化与执行接口保留。
- 各 physical task 的融合/拆分选择在 F 前冻结到各层 context；未选中的带历史任务仍走原融合 FA，D 复用原完整 dx/current-dKV 分解。
- H 从单槽改为按任务保存，可同时保留独立 H。每个 H 完成所有层的路由后才放行其消费者；失败状态不能重试累加。沿用原 pending/KV 路由，最后用户之后释放，迭代结束检查无残留。
- W 仍绑定原 task key、保持每个模型分块内的累加次序。D 用原 B 消息身份提交完整 dx，H 无额外 PP 消息；保留 FIFO、credit、ready/ack、buffer 和通信结束检查。
- 复用 P2 计数约束：v2 的 B 按融合路径计费，D/H 按实际所选集合与保留区间计费。timeline 和成本标签区分 B_fused/D/H。

实现入口：`dspp_plan.py`、`dspp_plan_runtime.py`、`schedules.py`、`dspp_packed_attention.py`。v2 外部计划通过现有 `dspp_plan` 参数提交；要求固定 capacity、原 split/after-dx 开关和 memory profile。`split_plan(reference, selected)` 只生成手工夹具，不执行搜索。未修改分解算法或 kernel。

## 验证规模与结果

新增三次训练，每次三个 optimizer step；另有两个 GPU 单元用例。没有做性能矩阵。

| 检查 | 结果 |
| --- | --- |
| 新增 CPU 合同用例 6 项 | 通过；PP3/4/8、Plan 往返、有限 lookahead/depth=2、H 身份/全部消费者、多任务释放、异常拒绝 |
| 已有相关 CPU 回归 97 项 | 通过；计划、任务执行、显存计数 |
| 定向成本标签 CPU 用例 2 项 | 通过；selective 下 B_fused/D/H 不混标 |
| 立即 H 三步训练 | 正常结束，作为数值参照 |
| H 跨 W 三步训练 | 输出、dx、梯度、logits/loss、更新前后参数相对参照最大差异均为 0 |
| H 跨独立 F 三步训练 | 同上，均为 0；三步各有一个 H 确实跨过 W 和 F |
| 三层 GQA、两个待执行 H | 同一 KV state 中同时持有两个 H，每个包含三层；延后/立即/融合反向输出、dx、所有参数梯度差异为 0；H 不改变已交付 dx |
| 原立即 H 路径单个 GPU 回归 | 通过；确认旧接口兼容 |

训练复用 P2 的功能配置：6层、hidden512/FFN1024、MHA4/head_dim128、PP3/VPP2、BF16、chunk64、micro4/global8、三步 Adam、LR 1e-3，合成长度193/129/33/17。精确命令见各运行目录的 `command.json`，原输入及标定沿用只读 `/home/cuiym/tmp/dspp-p2-20261023/` 与 `dspp-p2-split-memory-20261023/memory-profile.json`。此配置用于执行正确性，无性能采样。

三层用例使用 GQA(query heads=4，KV heads=2)、head_dim128、长度145/70/17/3、chunk64。训练与该数值用例沿用确定性 FA 诊断，比较门槛仍为 atol=0.004、rtol=0.01、relative-L2≤0.02；没有通过放宽阈值取得结果。

所有27份逐卡/逐步训练记录检查通过：同一步全 rank 计划摘要一致；三种计划的输入绑定、拆分集合、F/B 相对次序、W 次序、通信消息/FIFO 一致，只改变指定 H 位置。D/H 按任务一一对应，训练结束通信与 KV/H/W 状态排空，无 optimizer 跳步。

整步峰值包含 optimizer 更新。三个 rank 的最大 allocated 为749.08/181.62/181.12 MiB，对应最大计数预留907.62/216.81/216.81 MiB。暖态最大为749.08/142.26/141.40 MiB。

通信日志确认部分 D 的 NCCL 发送在 H 前提交：立即/跨 W/跨 F 分别2/4/4次，各有76个需跨卡发送的 D。其余仍受原 FIFO/credit 控制，不能据此声称全部发送已在 H 前完成或证明获得通信重叠收益。

## 记录与复核

原始目录：`/home/cuiym/tmp/dspp-p3-20261023/`。`summary.json` 保存逐场景指标、计划摘要、实际 H 移位和三层梯度指标。`analyze.py` 应对原始目录运行：`python3 analyze.py /home/cuiym/tmp/dspp-p3-20261023`。

- 训练日志及逐步全张量对照：`adjacent/`、`across-w/`、`across-fb/`；完整参照张量：`snapshots/`。
- CPU：`cpu-initial.log`、`cpu-p3-final.log`；三层与成本标签：`gpu-multilayer-final.log`；旧路径：`legacy-history-regression.log`。
- 三层测试最初两次在加载扩展前被 Git safe.directory 拦住，分别缺主源码、CUTLASS 子仓库的信任目录；失败日志 `gpu-multilayer.log`、`gpu-multilayer-fixed-env.log` 保留。命令级补齐两个只读源码目录后通过，未改测试容差或扩展源码。
- 起点 `cuiym-dspp@9550175c1d14ab309e96f090142e105b1a8501fd`，保留既有改动。`before/`、`tracked-before.patch`、`before-sha256.json` 保存开始前状态；`implementation.patch`、`plan.patch` 仅包含本轮增量，`closing.json` 记录复核。

训练结束后仅删除重复的相同映射、修正注释，并补齐诊断报告的 plan_hash 字段；未再运行训练。最后复核六项新 CPU 合同用例与旧路径/成本标签。历史实验归档和 kernel 保持原样，未 commit/push。
