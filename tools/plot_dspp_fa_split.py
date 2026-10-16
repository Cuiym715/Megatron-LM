#!/usr/bin/env python3
"""Plot verified GPU events and numerical evidence; never synthesize F/B/W."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


CASES = ("original", "dq_then_dkv", "dkv_then_dq", "dq_only", "dkv_only")
LABELS = ("原 FA", "拆分：先 dQ", "拆分：先 dKV", "仅计算 dQ", "仅计算 dKV")
COLORS = {"F_attn": "#2878B5", "prepare": "#737B85", "dQ": "#168C78",
          "dKV": "#D55E00", "fused_B_attn": "#9467A8"}
NAMES = {"F_attn": "Attention 前向", "prepare": "共享预处理",
         "dQ": "dQ（含清零/转换）", "dKV": "完整 dKV", "fused_B_attn": "原融合 backward"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def timeline_figure(data, output):
    fig = plt.figure(figsize=(14, 8.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.2], hspace=0.48)
    ax = fig.add_subplot(grid[0])
    config = data["config"]
    fig.suptitle("阶段 A：真实 GPU 执行，dQ 与 dKV 已分开启动", fontsize=20, fontweight="bold", y=1.025)
    fig.text(0.5, 0.978, f'L40 · BF16 · S={config["length"]}, H={config["heads"]}, d=128 · 单序列 MHA / causal / dropout=0',
             ha="center", fontsize=11, color="#555555")
    end = 0
    for y, case in enumerate(CASES):
        for event in data["timeline"][case]:
            start, duration = event["start_us"], event["duration_us"]
            phase = event["phase"]
            ax.broken_barh([(start, duration)], (y - 0.22, 0.44), facecolors=COLORS[phase], linewidth=0)
            if duration > 45:
                label = {"F_attn": "F_attn", "fused_B_attn": "融合 B_attn"}.get(phase, phase)
                ax.text(start + duration / 2, y, label, color="white", fontsize=11,
                        ha="center", va="center", fontweight="bold")
            end = max(end, start + duration)
    ax.set_yticks(range(5), LABELS)
    ax.set_ylim(4.6, -0.65)
    ax.set_xlim(0, np.ceil(end / 100) * 100)
    ax.set_xlabel("GPU 时间（μs；每个样例从其首个 GPU kernel 起点对齐）", labelpad=9)
    ax.grid(axis="x", color="#e1e4e8", linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(color=COLORS[p], label=NAMES[p]) for p in COLORS],
              loc="lower left", bbox_to_anchor=(0, 1.015), ncol=5, frameon=False, fontsize=10)

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    table_ax.set_title("主计算 kernel 启动次数：由实际 kernel 名称和 CUDA launch correlation 核对", loc="left", fontsize=12, pad=12)
    cells = [[LABELS[i], data["launch_checks"][c]["dq_main"],
              data["launch_checks"][c]["dkv_main"], data["launch_checks"][c]["original_fused_main"]]
             for i, c in enumerate(CASES)]
    table = table_ax.table(cellText=cells, colLabels=["执行样例", "独立 dQ", "独立 dKV", "原融合 backward"],
                           cellLoc="center", colWidths=[0.34, 0.22, 0.22, 0.22], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e3e6ea")
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#f8fafc")
    fig.text(0.125, -0.005,
             "色块是实际 GPU kernel，空白保留观测到的间隔。原融合路径也含预处理和转换。\n"
             "这是 attention 级执行证据；没有 stage W，橙色为完整 dKV，尚未选择历史区域或延后。\n"
             "单次 profiler 采样会扰动时间，不用此图宣称加速或估计可用调度空隙。",
             fontsize=10, color="#555555", linespacing=1.6)
    save(fig, output, "gpu_execution")


def accuracy_figure(validation, output):
    cases = [r for r in validation["numerical_cases"] if r["length"] > 1]
    labels = [f'S={r["length"]}\nH={r["heads"]}' if r["amplitude"] == 1 else
              f'压力样例\nseed={r["seed"]}' for r in cases]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("正确性：拆分没有增加可见的梯度误差", fontsize=20, fontweight="bold", y=1.01)
    fig.text(0.5, 0.963, "独立 FP64 数学参考；原融合 FA 与拆分实现使用相同 BF16 输入、相同前向输出", ha="center", fontsize=11, color="#555555")
    for ax, grad in zip(axes, ("dq", "dk", "dv")):
        original = [r["gradients"][grad]["fused_vs_fp64"]["relative_l2"] * 100 for r in cases]
        split = [r["gradients"][grad]["split_vs_fp64"]["relative_l2"] * 100 for r in cases]
        x = np.arange(len(cases))
        ax.plot(x, original, "o--", color="#747B86", markerfacecolor="white", markersize=9,
                linewidth=1.4, label="原融合 FA")
        ax.plot(x, split, ".", color="#168C78", markersize=8, label="拆分")
        ax.axhline(1.2, color="#B45243", linestyle=":", linewidth=1, label="相对 L2 验收上限 1.2%")
        ax.axvspan(len(cases) - 3.5, len(cases) - 0.5, color="#fcf0e8", alpha=0.8, zorder=-1)
        ax.set_ylabel(grad.replace("dq", "dQ").replace("dk", "dK").replace("dv", "dV") + " 相对 L2 误差（%）")
        ax.set_ylim(0, 1.32)
        ax.set_xlim(-0.5, len(cases) - 0.5)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, ncol=3, fontsize=10)
    axes[-1].set_xticks(range(len(cases)), labels, fontsize=9)
    fig.subplots_adjust(hspace=0.16, bottom=0.16)
    fig.text(0.125, 0.025,
             "绿点与灰圈基本重合，表示两种 BF16 backward 对 FP64 的误差相当；不是与 FP64 精确相等。\n"
             "浅橙区域为幅度加大的三个压力样例：要求与原 FA 逐位一致，并通过范数误差检查。\n"
             "S=1 的数学零 dQ/dK 单独检查绝对误差，不画无意义的相对误差。完整阈值与首轮失败记录见计划 §9.3。",
             fontsize=10, color="#555555", linespacing=1.5)
    save(fig, output, "gradient_accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "path"})
    timeline_figure(json.loads(args.evidence.read_text()), args.evidence.parent)
    accuracy_figure(json.loads(args.validation.read_text()), args.evidence.parent)


if __name__ == "__main__":
    main()
