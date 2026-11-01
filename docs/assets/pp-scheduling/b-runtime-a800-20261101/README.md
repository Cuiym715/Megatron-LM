# B运行时精简与A800/PP8证据索引（2026-11-01）

当前说明见[算法B实现文档](../../../dspp-algorithm-b-implementation.md)。本目录只收录小型原始汇总、8文件部署身份和增量补丁，不把大型checkpoint、profile和trace复制进代码仓库。

| 文件 | 内容 |
| --- | --- |
| stage12-benchmark.json | 精简前61层baseline＋P/C/F0/F2五项，及64层共同模型三方法，均5预热＋20计时 |
| runtime-pairs-benchmark.json | 精简后61层baseline/B三次配对，全部在线，汇总+2.82% |
| endpoint-benchmark.json | 同首尾减层规则61层三方法三组轮换，B相对baseline+3.34%；含逐rank报告身份与结束资源状态 |
| infini-endpoints-cpu.json | InfiniPipe7/8/8/8/8/8/8/6实际构建、成本/显存及层区间检查 |
| runtime-hashfree.patch | 相对部署前工作树的8文件精简补丁，不包含已有的其他未提交改动 |
| deployed-source.json | 同8文件的部署前/后SHA256；这些身份在计时外生成，不是runtime核验 |
| archive-identities.json | 三个完整证据归档的服务器路径、字节数及SHA256 |

完整证据留在g37：

- `/workspace/outputs/dspp-longrun-20261101-exec01/stage12-validated-v1/`
- `/workspace/outputs/dspp-longrun-20261101-exec01/final-validated-v1/`
- `/workspace/outputs/dspp-unified61-20261101-exec01/final-validated-v1/`

这些容器路径的宿主机前缀为`/home/test/test01/tinacui`，替换`/workspace`。对应本机交接目录为`dspp/longrun-20261101-exec01`与`dspp/unified61-20261101-exec01`，其完整报告、逐份rank报告/包身份核对记录均已保存。本机已核对前两批64份和80份、最后一批80份rank报告及完整压缩包字节身份。SHA仅用于这类离线归档，不重新引入训练热路径。

接受的验证范围：3856个CPU预测查询相同，真实spawn和D/H求解；首批3轮隔离入口及3轮部署入口、150配对更新的完整图验证；最近一批150个DSPP计划完整图验证。显式schema2 D/H计划额外进行旧容量检查，schema1融合B的baseline不冒称做了schema2容量检查。两批InfiniPipe61各2轮全层执行/梯度/参数更新验证，最近32个checkpoint分片头检查。默认高性能FA保持不变，跨框架同权重逐元素对照及真实变长数据收益未在本批补做。

失败记录保留在完整归档：初版InfiniPipe61探针的flat参数假设、GPU索引舍入越界已修复；最近三方法汇总器误向非rank0读取valid_tokens已修复，已完成训练结果复核复用，没有删除慢轮或重跑挑选结果。
