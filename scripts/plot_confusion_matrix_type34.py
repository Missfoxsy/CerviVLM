#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot Type3 / Type4 confusion matrices from eval log + val JSON."""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# Internal keys (match dataset / eval extraction)
TYPE3_CLASSES = ["ASCUS", "ASC_H", "EpitheliumNormal", "HSIL", "LSIL"]
TYPE4_CLASSES = [
    "上皮成团",
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮底层",
    "宫颈鳞状上皮表层",
]

# English display labels (plot only)
TYPE3_LABELS_EN = ["ASCUS", "ASC_H", "Normal", "HSIL", "LSIL"]
TYPE4_LABELS_EN = [
    "epithelial cluster",
    "squamous mid layer",
    "squamous basal layer",
    "squamous superficial layer",
]

# Type3 constraint differs from type5 (CoT); use it to avoid double-counting.
TYPE3_LOG_PATTERN = (
    r"是什么类型的细胞.*?请从问题给出的候选类别中选择，并且只输出一个类别名称"
    r".*?assistant\s*\n\s*=================\n([^\n]+)"
)
TYPE4_LOG_PATTERN = (
    r"图中的细胞来源是什么.*?assistant\s*\n\s*=================\n([^\n]+)"
)


def _extract_class_from_text(text: str, valid_classes: list):
    if not text:
        return None
    t = str(text).strip()
    if t in valid_classes:
        return t
    for cls in valid_classes:
        if cls in t:
            return cls
    return None


def _gt_from_item(item: dict, valid_classes: list):
    for conv in item.get("conversations", []):
        if conv.get("from") == "assistant":
            return _extract_class_from_text(conv.get("value", ""), valid_classes)
    return None


def _parse_preds_from_log(log_path: str, pattern: str, valid_classes: list):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    preds = []
    for m in re.finditer(pattern, text, flags=re.DOTALL):
        preds.append(_extract_class_from_text(m.group(1).strip(), valid_classes))
    return preds


def _load_pairs_from_log(
    log_path: str,
    val_json: str,
    task_id: int,
    pattern: str,
    valid_classes: list,
    task_name: str,
):
    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    items = [item for item in val_data if int(item.get("task_id", -1)) == task_id]
    if not items:
        raise ValueError(f"验证集中未找到 task_id={task_id} 样本")

    preds = _parse_preds_from_log(log_path, pattern, valid_classes)
    if len(preds) != len(items):
        raise ValueError(
            f"{task_name} 预测数({len(preds)})与验证集样本数({len(items)})不一致"
        )

    return [(pred, _gt_from_item(item, valid_classes)) for pred, item in zip(preds, items)]


def _build_confusion_matrix(pairs, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for pred, gt in pairs:
        if gt not in idx:
            continue
        row = idx[gt]
        col = idx[pred] if pred in idx else -1
        if col >= 0:
            cm[row, col] += 1
    return cm


def _plot_confusion_matrix(cm, display_labels, title, accuracy, output_path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{title}\nAccuracy = {accuracy:.4f}", fontsize=13)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0:
                continue
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Type3/Type4 confusion matrices")
    parser.add_argument(
        "--log",
        default="logs/eval_stage20512.log",
        help="评估日志（解析 type3/type4 预测输出）",
    )
    parser.add_argument(
        "--val_json",
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
        help="验证集 JSON（提供 GT 标签，与评估顺序对齐）",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_result/stage2_taskAdaptive_0512",
    )
    args = parser.parse_args()

    out_dir = args.output_dir

    type3_pairs = _load_pairs_from_log(
        args.log, args.val_json, 3, TYPE3_LOG_PATTERN, TYPE3_CLASSES, "type3"
    )
    cm3 = _build_confusion_matrix(type3_pairs, TYPE3_CLASSES)
    acc3 = sum(p == g and p is not None for p, g in type3_pairs) / len(type3_pairs)
    out3 = os.path.join(out_dir, "confusion_matrix_type3.png")
    _plot_confusion_matrix(
        cm3, TYPE3_LABELS_EN, "Lesion Type Classification", acc3, out3
    )
    print(f"Lesion Type: n={len(type3_pairs)}, acc={acc3:.4f} -> {out3}")

    type4_pairs = _load_pairs_from_log(
        args.log, args.val_json, 4, TYPE4_LOG_PATTERN, TYPE4_CLASSES, "type4"
    )
    cm4 = _build_confusion_matrix(type4_pairs, TYPE4_CLASSES)
    acc4 = sum(p == g and p is not None for p, g in type4_pairs) / len(type4_pairs)
    out4 = os.path.join(out_dir, "confusion_matrix_type4.png")
    _plot_confusion_matrix(
        cm4, TYPE4_LABELS_EN, "Cell Origin Classification", acc4, out4
    )
    print(f"Cell Origin: n={len(type4_pairs)}, acc={acc4:.4f} -> {out4}")


if __name__ == "__main__":
    main()
