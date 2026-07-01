#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CVPR-style FPN Ablation Figure.

输出两张独立图：
- F1 + mAP（双 Y 轴，通过不对称 ylim 使两曲线上下分区）
- Mean IoU
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

DEFAULT_EVAL_ROOT = "eval_result/stage1_fpn_ablation"
DEFAULT_LEVELS = [1, 2, 4, 6]
TYPE0_EXPECTED_COUNT = 1652
BEST_MARKER_SIZE = 300

C_F1 = "#E6862E"
C_MAP = "#9CCAE5"
C_IOU = "#E45756"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
})


def _metrics_path(eval_root: str, n_layers: int) -> str:
    return os.path.join(
        eval_root,
        f"fpn_{n_layers}layer_legacyAttn",
        "metrics.json",
    )


def _type0_complete(metrics: dict) -> bool:
    dist = metrics.get("task_distribution") or {}
    return int(dist.get("type0", 0)) == TYPE0_EXPECTED_COUNT


def load_type0_series(eval_root: str, levels: List[int]):
    series = {"f1": [], "map": [], "mean_iou": []}
    plot_levels = []

    for n in levels:
        path = _metrics_path(eval_root, n)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if not _type0_complete(metrics):
            continue
        plot_levels.append(n)
        series["f1"].append(float(metrics["type0_f1"]))
        series["map"].append(float(metrics["type0_map"]))
        series["mean_iou"].append(float(metrics["type0_mean_iou"]))

    return plot_levels, series


def to_percent(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def _fmt_one_decimal(v, _pos):
    return f"{v:.1f}"


def _dual_axis_ylim(values, band: str, span: float = 3.0):
    """通过不对称 ylim 把曲线推到图的上/下区域，避免双轴同形重叠。"""
    arr = np.asarray(values, dtype=float)
    center = (float(arr.min()) + float(arr.max())) / 2.0
    frac = 0.78 if band == "upper" else 0.22
    ymin = center - frac * span
    return ymin, ymin + span


def _four_y_ticks(ymin: float, ymax: float):
    """生成 4 个等距刻度（3 条网格线），与 IoU 子图风格一致。"""
    ticks = np.linspace(ymin, ymax, 4)
    return np.round(ticks, 1)


def _apply_four_y_ticks(ax, ymin: float, ymax: float):
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(_four_y_ticks(ymin, ymax))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_one_decimal))


def _style_axis(ax, tick_step: float, *, grid: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_one_decimal))
    if grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)


def _style_twin_axis(ax, tick_step: float):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_alpha(0.35)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_one_decimal))
    ax.tick_params(axis="y", labelsize=11)


def _plot_series(ax, x, y, label, color, marker):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=8,
        linewidth=2.5,
        label=label,
        zorder=3,
    )

    best_idx = int(np.argmax(y))
    ax.scatter(
        x[best_idx],
        y[best_idx],
        marker="*",
        s=BEST_MARKER_SIZE,
        color=color,
        edgecolor="black",
        linewidth=0.8,
        zorder=10,
    )


def _set_xaxis(ax, x):
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=11)
    ax.set_xlabel("FPN Depth", fontsize=12)


def _save_figure(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_f1_map_figure(x, f1_pct, map_pct, output_path: str):
    fig, ax1 = plt.subplots(figsize=(5.0, 4.0))
    ax2 = ax1.twinx()
    fig.subplots_adjust(top=0.95, bottom=0.14, left=0.14, right=0.88)

    _plot_series(ax1, x, f1_pct, "F1", C_F1, "^")
    _plot_series(ax2, x, map_pct, "mAP", C_MAP, "s")

    f1_lo, f1_hi = _dual_axis_ylim(f1_pct, band="upper", span=3.0)
    map_lo, map_hi = _dual_axis_ylim(map_pct, band="lower", span=3.0)

    _style_axis(ax1, tick_step=0.4, grid=True)
    _apply_four_y_ticks(ax1, f1_lo, f1_hi)
    _style_twin_axis(ax2, tick_step=0.4)
    _apply_four_y_ticks(ax2, map_lo, map_hi)

    ax1.set_ylabel("F1", fontsize=12)
    ax2.set_ylabel("mAP", fontsize=12)

    _set_xaxis(ax1, x)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        fontsize=10,
    )

    _save_figure(fig, output_path)


def plot_iou_figure(x, iou_pct, output_path: str):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.subplots_adjust(top=0.95, bottom=0.14, left=0.14, right=0.96)

    _plot_series(ax, x, iou_pct, "Mean IoU", C_IOU, "o")

    ax.set_ylim(77.6, 79.0)
    _style_axis(ax, tick_step=0.4)
    ax.set_ylabel("Mean IoU", fontsize=12)
    _set_xaxis(ax, x)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    _save_figure(fig, output_path)


def plot_fpn_ablation(
    eval_root: str,
    levels: List[int],
    output_loc: str,
    output_iou: str,
):
    x, series = load_type0_series(eval_root, levels)
    if len(x) == 0:
        raise RuntimeError("No valid results found.")

    x = np.asarray(x, dtype=float)
    f1_pct = to_percent(series["f1"])
    map_pct = to_percent(series["map"])
    iou_pct = to_percent(series["mean_iou"])

    plot_f1_map_figure(x, f1_pct, map_pct, output_loc)
    plot_iou_figure(x, iou_pct, output_iou)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--levels", default="1,2,4,6")
    parser.add_argument(
        "--output_loc",
        default=None,
        help="F1+mAP 图输出路径",
    )
    parser.add_argument(
        "--output_iou",
        default=None,
        help="Mean IoU 图输出路径",
    )
    args = parser.parse_args()

    levels = [int(v.strip()) for v in args.levels.split(",") if v.strip()]
    output_loc = args.output_loc or os.path.join(
        args.eval_root, "fpn_ablation_loc_f1_map.png"
    )
    output_iou = args.output_iou or os.path.join(
        args.eval_root, "fpn_ablation_loc_iou.png"
    )

    plot_fpn_ablation(args.eval_root, levels, output_loc, output_iou)


if __name__ == "__main__":
    main()
