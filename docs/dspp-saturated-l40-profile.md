# DSPP L40 饱和配置与 timeline 记录

日期：2026-09-03

## 1. 最终配置

为了保留约 13 个 physical microbatch，同时避免原来 `chunk=32` 的小 GEMM，最终采用：

```text
GPU:                     3 x NVIDIA L40
PP / VPP:                3 / 2
Transformer layers:      24（每个逻辑 stage 4 层）
hidden / FFN / heads:    2048 / 5504 / 16
BF16
chunk size:              256
training length pattern: 768, 512, 192, 128
logical microbatches:    2 x 4 sequences
activation P2P payload:  256 x 2048 x 2 bytes = 1 MiB
```

长度模式是原正确性数据 `96,64,24,16` 的 8 倍，chunk 也是 `32` 的 8 倍。因此长序列仍
分别切成 3 段或 2 段，短序列 packing 的相对关系保持不变。随机 sampler 会令不同 iteration
的组合略有变化；记录 timeline 的 iteration 2 为 13 个 physical microbatch，和旧图一致。

L40 的 1 秒采样在活跃计算区间观察到 99--100% SM busy。该结论表示 task 已足够大、活跃
stage 能占满 SM；它不是 Tensor Core roofline 百分比。pipeline bubble 仍然保留，因而三个
stage 不会在整个 iteration 同时保持 100%。

## 2. 数据来源

原 `/tmp/dspp_b1_varlen_20260902_text_document` 不是公开数据集，而是临时合成的 Megatron
indexed dataset：32 个文档，raw length 循环为 `97,65,25,17`，对应训练长度
`96,64,24,16`；token 是递增的合成 id。

现在增加 `tools/build_dspp_synthetic_dataset.py`，使数据可复现。饱和 preset 在默认数据不
存在时自动生成 32 个文档，训练长度循环为 `768,512,192,128`。它只用于验证调度、性能形态
和 timeline，不代表真实语料的 loss 或长度分布。

## 3. 三卡结果

最终 2-iteration timeline run：

```text
iteration 1: 3017.5 ms, loss 10.74022, grad norm 60.211
iteration 2: 1020.6 ms, loss  9.60648, grad norm 34.987
skipped iterations: 0
NaN iterations:     0
```

iteration 2：

```text
physical microbatches: 13
packing utilization:   0.9423
effective tokens/s:    3083.5
critical span:         934.8 ms
peak allocated memory: stage0 11.17 GiB, stage1/2 about 8.71 GiB

compute-lane utilization:
stage0 98.1%
stage1 61.6%
stage2 63.1%
```

F median 约为 6.3--7.7 ms，W median 约为 2.54--3.39 ms。与旧 hidden=64/chunk=32 配置中
约 0.02 ms 的 W 相比，当前 task 已不再是纯 launch-overhead 微基准。B 的 CUDA-event 宽度
仍会受队列位置影响，因此比较 intrinsic B/F/W cost 时仍应使用 Nsight 的非 NCCL kernel sum。

## 4. 产物与复现

- `results/dspp_saturated_c256_h2048_l24/timeline/iteration_000002.svg`
- `results/dspp_saturated_c256_h2048_l24/timeline/iteration_000002_microbatches.md`
- `results/dspp_saturated_c256_h2048_l24/timeline/iteration_000002_summary.json`
- `results/dspp_saturated_c256_h2048_l24/train_iteration2.log`

复现：

```bash
TRAIN_ITERS=2 \
DSPP_TIMELINE_ITERATION=2 \
DSPP_TIMELINE_DIR=/workspace/src/results/dspp_saturated/timeline \
bash examples/run_dspp_b23_l40_saturated.sh
```

## 5. 大消息限制

尝试 `hidden=4096, chunk=1024` 时 activation P2P payload 为 8 MiB，三张卡均进入低功耗
NCCL kernel 自旋，首个 iteration 无法完成；`hidden=2048, chunk=512` 的 2 MiB payload 也
触发相同问题。1 MiB payload 的两个配置均可完成训练。

这表明当前 stage-local P2P 任务顺序对大消息 rendezvous 仍有未解决的进度问题。最终 preset
选择 1 MiB payload 不是因为显存不足，而是为了先获得可运行的饱和计算 timeline。正式长上下文
实验前，需要单独修复这个大消息通信问题，不能把小 payload 的成功视为通信正确性的充分证明。
