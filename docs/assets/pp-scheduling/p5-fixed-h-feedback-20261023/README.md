# P5 固定H反馈：实现与真机记录

代码已接通，含H数值验收未完成。新增 `--dspp-schedule c`：先B，再固定其H集合做一次反馈；不增加或撤销拆分。反馈未改善、超容量、过截止或不符合部署W次序时保留整个B计划。

## 输入和模型

复用已有seed11分布输入：`/home/cuiym/tmp/dspp-a-performance-20261023/data/seed11/`。原采样入口为 `InfiniPipe-code/scripts/generate_dspp_compare_lengths.py`，调用已有论文图2分布采样器；没有新建采样器。

24层、hidden2048、FFN5504、MHA16、BF16、PP3/VPP2、chunk2048、micro batch4/global batch8，初始LR1e-3。三步的有效序列长度：

| 步 | 有效长度 |
| --- | --- |
| 1 | 767/767/13567/767/767/1279/767/767 |
| 2 | 767/3839/1023/767/767/2559/1023/767 |
| 3 | 1279/1279/767/1023/767/767/1279/767 |

共38,888个有效tokens。冻结profile复用 `/home/cuiym/tmp/dspp-b-performance-20261023/` 下的stage、P2P和memory文件。为了覆盖含H反馈，使用标记为diagnostic的构造耗时；没有改变输入、训练算子和显存额度。运行带逐任务快照，不用于吞吐比较。

## 真机结果

| 运行 | 结果 |
| --- | --- |
| 融合参考 | 完成三步，保存逐卡快照 |
| 初组C诊断 | 三步全部数值差异为0；H均为0，实际执行的是保留B的分支 |
| 含H的C，对融合参考 | 首步H=1、反馈被采用；执行和显存检查完成，更新参数数值对照失败 |
| 反馈前B，对融合参考 | 初始参数/前向/logits/loss相同；梯度最大相对差异0.81%，dx1.33%；更新参数最大2.94%，超过原2%门槛 |
| 含H的B参考 | 完成三步，H数1/1/0 |
| C，对上述B参考 | 首步反馈被采用，更新参数仍超差；未完成后两步对照 |
| 相同B计划首步自重复诊断 | 更新参数最大相对差异2.95%，15个权重张量超差；诊断收集器退出0只表示保存了报告，不表示数值通过 |

含H首步C的allocated峰值为21.716/19.809/20.543GiB，预测为36.037/32.147/32.995GiB。H任务身份为 `(stage0, chunk1, group3, split1)`；反馈保留该H，只修改stage1/2的任务编排。诊断预测34,300→34,150ms只是说明反馈接受条件被触发，不能解读为真机加速。

自重复还对同一拆分反向状态连续计算两次：四层dQ分别有442/475/467/468个BF16元素不同，最大绝对差1.164e-10；current dK/dV相同。现有split dQ采用FP32原子累加；测试中的确定性FA开关只控制融合路径。已经测到split dQ的重复性问题，尚未确认它解释全部更新参数超差。没有降低LR、修改kernel或放宽阈值。

## 部署决策和待办

a. B与反馈共享原未拆分预测时间60%的规划预算。新增反馈预留 `0.10ms ×（计算任务数＋可拆候选数）`，从B搜索额度扣除；系数沿用031尾部预留，C的分配尚未单独标定。

b. C继承部署版B的每模型chunk内W次序校验。违反时保留B；没有修改研究归档。

c. 先定位拆分反向自重复与参数更新超差，再完成含H三步验收。正常成本下的C吞吐留到P7。P6/P7未实施。

两项部署做法只记录在plan，`pp-scheduling.md`保持本轮开始时的内容。

## 可复核材料

[数值与显存汇总](gpu-summary.json)。完整原始目录：`/home/cuiym/tmp/dspp-p5-20261023/`。

- `run_training.py`、`training_entry-*.py`及各运行的`command.json`：命令和诊断入口版本。
- `reference/`、`fixture/`、`fixture-targeted/`：融合参考、无H路径和含H失败。
- `diagnostic-seed/`、`seed-reference/`、`c-vs-seed/`、`diagnostic-seed-repeat/`：B隔离、三步参考、C对B失败、自重复和dQ重复测量。
- `seed-reference-launch-failed/`：前组尚未退出导致端口占用，未进入训练，排除。
- `inspect_feedback.py`、`inspect-feedback.log`：从保存的B种子重建C，检查反馈的次序变化；`reconstructed-c-1.json`明确为重建计划。
- `before/`、`before-sha256.json`、`status-before.txt`、`p5-incremental.patch`：保留开始时工作树，隔离本轮增量。

分支`cuiym-dspp`，HEAD `9550175c1d14ab309e96f090142e105b1a8501fd`。未commit/push。
