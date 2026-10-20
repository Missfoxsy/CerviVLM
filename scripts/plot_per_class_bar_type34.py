#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 eval 日志 + val JSON 解析 Type3/Type4 预测，绘制各类别 P/R/F1 分组柱状图。

数据读取逻辑与 ``plot_confusion_matrix_type34.py`` 一致（log 正则 + val_json GT 对齐）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_confusion_matrix_type34 import (
    TYPE3_CLASSES,
    TYPE3_LABELS_EN,
    TYPE3_LOG_PATTERN,
    TYPE4_CLASSES,
    TYPE4_LABELS_EN,
    TYPE4_LOG_PATTERN,
    _load_pairs_from_log,
)


def compute_per_class_metrics(
    pred_gt_pairs: List[Tuple[Optional[str], Optional[str]]],
    valid_classes: List[str],
) -> Dict[str, Any]:
    total = len(pred_gt_pairs)
    correct = sum(1 for p, g in pred_gt_pairs if p is not None and g is not None and p == g)
    accuracy = correct / total if total > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in valid_classes:
        tp = sum(1 for p, g in pred_gt_pairs if p == cls and g == cls)
        fp = sum(1 for p, g in pred_gt_pairs if p == cls and g != cls)
        fn = sum(1 for p, g in pred_gt_pairs if p != cls and g == cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    precisions = [per_class[c]["precision"] for c in valid_classes if per_class[c]["support"] > 0]
    recalls = [per_class[c]["recall"] for c in valid_classes if per_class[c]["support"] > 0]
    f1s = [per_class[c]["f1"] for c in valid_classes if per_class[c]["support"] > 0]
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def _plot_grouped_bars(
    metrics: Dict[str, Any],
    classes: List[str],
    display_labels: List[str],
    title: str,
    output_path: str,
) -> None:
    per_class = metrics["per_class"]
    n = len(classes)
    x = np.arange(n)
    # width = 0.24

    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 6.5))
    width = 0.18
    offset = 0.22

    bars_p = ax.bar(
        x - offset,
        precision,
        width,
        label="Precision",
        color="#B7C3D9",
        edgecolor="none",
    )

    bars_r = ax.bar(
        x,
        recall,
        width,
        label="Recall",
        color="#D4E0E7",
        edgecolor="none",
    )

    bars_f = ax.bar(
        x + offset,
        f1,
        width,
        label="F1-score",
        color="#CFCFCF",
        edgecolor="none",
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.linspace(0, 1, 6))
    # ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=10)

    # subtitle = (
    #     f"Accuracy={metrics['accuracy']:.4f}  "
    #     f"Macro P/R/F1={metrics['macro_precision']:.4f}/"
    #     f"{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}"
    # )
    ax.set_title(f"{title}", fontsize=13)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if n > 6 else 0,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Type3/Type4 per-class P/R/F1 bar charts from eval log + val JSON"
    )
    parser.add_argument(
        "--log",
        default="logs/eval_stage20512.log",
        help="评估日志（解析 type3/type4 预测输出）",
    )
    parser.add_argument(
        "--val_json",
        default="/path/to/dataset/stage2_cot_val.json",
        help="验证集 JSON（提供 GT 标签，与评估顺序对齐）",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_result/stage2_taskAdaptive_0512",
    )
    parser.add_argument(
        "--prefix",
        default="per_class_bar",
        help="输出文件名前缀",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, Any] = {
        "log": args.log,
        "val_json": args.val_json,
        "output_dir": out_dir,
    }

    type3_pairs = _load_pairs_from_log(
        args.log, args.val_json, 3, TYPE3_LOG_PATTERN, TYPE3_CLASSES, "type3"
    )
    m3 = compute_per_class_metrics(type3_pairs, TYPE3_CLASSES)
    out3 = os.path.join(out_dir, f"{args.prefix}_type3.png")
    _plot_grouped_bars(
        m3,
        TYPE3_CLASSES,
        TYPE3_LABELS_EN,
        "Lesion Type Classification",
        out3,
    )
    summary["type3"] = m3
    print(
        f"Type3: n={len(type3_pairs)}, acc={m3['accuracy']:.4f}, "
        f"macro_f1={m3['macro_f1']:.4f} -> {out3}"
    )

    type4_pairs = _load_pairs_from_log(
        args.log, args.val_json, 4, TYPE4_LOG_PATTERN, TYPE4_CLASSES, "type4"
    )
    m4 = compute_per_class_metrics(type4_pairs, TYPE4_CLASSES)
    out4 = os.path.join(out_dir, f"{args.prefix}_type4.png")
    _plot_grouped_bars(
        m4,
        TYPE4_CLASSES,
        TYPE4_LABELS_EN,
        "Cell Origin Classification",
        out4,
    )
    summary["type4"] = m4
    print(
        f"Type4: n={len(type4_pairs)}, acc={m4['accuracy']:.4f}, "
        f"macro_f1={m4['macro_f1']:.4f} -> {out4}"
    )

    meta_path = os.path.join(out_dir, f"{args.prefix}_type34_metrics.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Metrics JSON: {meta_path}")


if __name__ == "__main__":
    main()
