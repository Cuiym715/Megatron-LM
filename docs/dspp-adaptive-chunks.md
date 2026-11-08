# 可调的轻量 chunk 选择

2026-11-06：实现与 CPU 集成验证完成。GPU 数值、显存实测、预测排序和吞吐评估待空闲卡；该功能显式开启，现有固定 chunk 和 CP 路径保持默认行为。

## 使用入口

在原 CP1、BF16、head-dim=128、after-dx、prefetch 训练命令上添加：

```sh
--dspp-chunk-model /path/to/model.json \
--dspp-chunk-points 64 \
--dspp-chunk-step 64 \
--dspp-chunk-spacing log \
--dspp-chunk-score pipeline
```

`--dspp-schedule template` 只执行所选 base；`--dspp-schedule b` 对选中容量运行一次 B。使用正常训练入口，不叠加旧 chunk-v1 的实验 entry/monkeypatch，也不打开旧的 `--dspp-auto-chunk-size`。新 bundle 已含成本、P2P、显存估计，旧 compute/P2P/memory profile 参数不参与新路径，初始化也不读取它们。

`--dspp-chunk-fixed 2048` 是同执行方式、同模型下的固定容量对照；默认 0 表示自动选择。64 点、64 token 网格和 pipeline 评分是临时默认值，未认定最优。`step=1` 允许任意整数容量，但仍只采样至多 `points` 个值；不是遍历所有整数。

当前导出包的搜索域是 2048～8192，由同执行方式的 PP8 实测数据范围确定。代码没有这个区间的常量，也没有四候选名单。更换有效模型可以改变搜索域。3058、3072、3840 等容量的 metadata/base 已完成 CPU 检查，3058 另完成真实 worker 的一次 B 和物化检查；实际 GPU 行为还待验证。

## 在线工作

1. 序列长度直方图只统计一次；按点数、间隔、linear/log 参数直接生成有限采样点，不枚举整个整数区间。
2. 批量计算七特征及每 stage 预计工作量；`work` 用总工作量，`pipeline` 用瓶颈 stage 工作量乘以简化填充代价。后者是待评估的粗评分，非完整 VZB makespan。
3. 取前二，再补当前值，最多三个候选。实际重新 packing/ordering，构造 VZB、history/P2P 依赖，计算显存和完整 base 时间。
4. 直接复用选中候选的 metadata/base/compute/messages，只给它创建一个 B 请求。B 的现有总预算包含本轮前端准备时间。

模型和静态配置每个进程启动时加载一次。worker 初始化时收到静态 bundle，此后请求只含变化的计划、metadata 和成本。没有每轮 token SHA、静态 profile 传输、重复整图验证或完整 worker 结果日志；接收端保留轮次、输入归属和 wire shape 判断。依赖/容量计算及既有物化失败 rendezvous 仍保留。

当前 bundle 使用 `canonical-query-v2` 执行适配器：沿用原四容量实验的确定性 FA、完整 BF16 归约、FP32 pending 和零 dropout 路径。fused backward 在各物理 chunk 内按至多 2048 query 的 tile 逆序执行，D/H 保持原生算子。新容量会改变 tile 划分，不能由 CPU 计划合法推断 GPU 梯度/更新已通过。适配器仅由新功能初始化安装，CP1 之外不可开启。

## 模型与 CPU 验证

离线导出使用同配置/同执行方式的冻结 compact snapshots，按 stage、virtual chunk 和 F/B/W/D/H 分组拟合七特征非负模型。已测签名优先查小表；每组最多保留 32 个聚合点，不保存原始重复记录。运行时不拟合。当前包 145,839 bytes，80 组模型；初始化约 4.85 ms（g37 独立 CPU 测量）。

显存使用已有记录的最大 bytes/token 系数和静态项构造估计，再由实际任务 lifetimes 检查。它是推导的模型，尚不是新容量的实测保证。D/H 目前只有 2048 的同执行方式数据，其他容量可求解，但预测精度仍待 GPU 检查。

```sh
python tools/export_dspp_chunk_model.py --profiles-dir /path/to/frozen/profiles --output /path/to/model.json
CUDA_VISIBLE_DEVICES= python tools/validate_dspp_chunk_cpu.py \
  --model /path/to/model.json --output /tmp/chunk-cpu.json --unit-tests
```

CPU 验证工具独立加载真实 metadata 模块，避免初始化训练应用；不启动 GPU 或通信组。覆盖有限采样、形状统计、完整 base、真实 spawn B、迟到/错误回退、token/label/position/mask 重组及固定 prefetch 回归。可用 `--legacy-runtime`、`--legacy-profiles` 加入冻结 chunk-v1 的同输入对照。GPU 上的后续比较应使用新功能的 `--dspp-chunk-fixed 2048` 与自动模式配对，以保持执行方式和模型一致。
