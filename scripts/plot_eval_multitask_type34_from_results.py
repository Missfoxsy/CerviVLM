#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
支持两种输入：

1) **metrics.json**（推荐）：直接读取 `eval_stage2_cot.py` 输出的 `metrics.json` 中的
   `type3_per_class` / `type4_per_class` 及宏平均指标，生成竖向分组柱状图（纯 SVG）。

2) **results.json**：从逐条样本重新按 `compute_per_class_metrics` 计算（可配合排除某类 gt）。

默认优先使用 `--metrics_json` 指向的文件；若不存在则回退到 `--results_json`。

results 模式默认会：
- Type3：去掉 gt 为 GlandTumor、HPV 的样本，且图中不画这两类；
- Type4：去掉 gt 为「腺上皮成团」的样本，且图中不画该类。

metrics 模式下指标已聚合，**不再应用**上述排除规则（需与 eval 时语言/任务一致）。

可用 --exclude_type3_gt / --exclude_type4_gt 传空字符串关闭对应过滤（仅 results 模式）。
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 与 eval_stage2_cot.py 中文评估一致的全量类别顺序
TYPE3_CELL_TYPE_CLASSES_ZH_FULL = [
    "ASCUS",
    "ASC_H",
    "EpitheliumNormal",
    "GlandTumor",
    "HPV",
    "HSIL",
    "LSIL",
]

TYPE4_SOURCE_CLASSES_ZH_FULL = [
    "上皮成团",
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮底层",
    "宫颈鳞状上皮表层",
    "腺上皮成团",
]

def _parse_exclude_csv(s: Optional[str]) -> frozenset:
    if s is None or not str(s).strip():
        return frozenset()
    return frozenset(x.strip() for x in str(s).split(",") if x.strip())


def _filtered_classes(full: List[str], exclude_gt: frozenset) -> List[str]:
    return [c for c in full if c not in exclude_gt]


def _filter_pairs_by_gt(
    pairs: List[Tuple[Optional[str], Optional[str]]], exclude_gt: frozenset
) -> List[Tuple[Optional[str], Optional[str]]]:
    if not exclude_gt:
        return pairs
    return [(p, g) for p, g in pairs if g not in exclude_gt]


def compute_per_class_metrics(
    pred_gt_pairs: List[Tuple[Optional[str], Optional[str]]], valid_classes: List[str]
) -> Dict[str, Any]:
    """与 eval_stage2_cot.compute_per_class_metrics 一致（不依赖 numpy）。"""
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


@dataclass
class _PlotTask:
    task_name: str
    accuracy: Optional[float]
    macro_p: Optional[float]
    macro_r: Optional[float]
    macro_f1: Optional[float]
    per_class: List[Tuple[str, float, float, float, int]]


def _plot_vertical_grouped_svg(task: _PlotTask, out_path: str) -> None:
    """与 plot_eval_stage2_cot_metrics.plot_task 相同的竖向 P/R/F1 + 柱顶数值。"""
    if not task.per_class:
        raise ValueError(f"No per-class data for {task.task_name}")

    rows = [
        {
            "cls": cls,
            "p": float(p),
            "r": float(r),
            "f1": float(f1),
            "support": int(support),
        }
        for cls, p, r, f1, support in task.per_class
    ]

    left_margin = 80
    right_margin = 30
    top_margin = 95
    bottom_margin = 120
    chart_h = 520
    group_w = 90
    group_gap = 26
    bar_w = 18
    bar_gap = 6
    chart_w = len(rows) * group_w + max(0, len(rows) - 1) * group_gap
    width = left_margin + chart_w + right_margin
    height = top_margin + chart_h + bottom_margin

    col_p = "#2F5D9B"
    col_r = "#2E8B57"
    col_f1 = "#B34747"
    grid_col = "#00000020"
    text_col = "#111827"
    muted_col = "#374151"

    def clamp01(v: float) -> float:
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    def y_of(v: float) -> float:
        v = clamp01(v)
        return top_margin + (1.0 - v) * chart_h

    title = task.task_name
    subtitle_parts = []
    if task.accuracy is not None:
        subtitle_parts.append(f"Acc={task.accuracy:.4f}")
    if task.macro_p is not None and task.macro_r is not None and task.macro_f1 is not None:
        subtitle_parts.append(
            f"Macro(P/R/F1)={task.macro_p:.4f}/{task.macro_r:.4f}/{task.macro_f1:.4f}"
        )
    subtitle = "  ".join(subtitle_parts)

    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    out.append(
        f'<text x="{left_margin}" y="34" font-size="20" font-family="DejaVu Sans, Arial" fill="{text_col}">{html.escape(title)}</text>'
    )
    if subtitle:
        out.append(
            f'<text x="{left_margin}" y="60" font-size="13" font-family="DejaVu Sans, Arial" fill="{muted_col}">{html.escape(subtitle)}</text>'
        )

    legend_x = left_margin + chart_w - 260
    legend_y = 30
    out.append(
        f'<rect x="{legend_x}" y="{legend_y-10}" width="10" height="10" fill="{col_p}"/><text x="{legend_x+16}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">Precision</text>'
    )
    out.append(
        f'<rect x="{legend_x+92}" y="{legend_y-10}" width="10" height="10" fill="{col_r}"/><text x="{legend_x+108}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">Recall</text>'
    )
    out.append(
        f'<rect x="{legend_x+160}" y="{legend_y-10}" width="10" height="10" fill="{col_f1}"/><text x="{legend_x+176}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">F1</text>'
    )

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_of(t)
        out.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{left_margin + chart_w}" y2="{y}" stroke="{grid_col}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left_margin-10}" y="{y+4}" text-anchor="end" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">{t:.2f}</text>'
        )
    out.append(
        f'<text x="{left_margin + chart_w/2}" y="{top_margin + chart_h + 70}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">Per-class metrics</text>'
    )

    for idx, row in enumerate(rows):
        gx = left_margin + idx * (group_w + group_gap)
        group_total_w = 3 * bar_w + 2 * bar_gap
        bx0 = gx + (group_w - group_total_w) / 2
        for j, (name, val, col) in enumerate(
            [("P", row["p"], col_p), ("R", row["r"], col_r), ("F1", row["f1"], col_f1)]
        ):
            x = bx0 + j * (bar_w + bar_gap)
            y = y_of(val)
            h = top_margin + chart_h - y
            out.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{max(0, h)}" fill="{col}"/>')
            val_text = f"{clamp01(float(val)):.4f}"
            out.append(
                f'<text x="{x + bar_w/2}" y="{y - 6}" text-anchor="middle" font-size="11" font-family="DejaVu Sans, Arial" fill="{text_col}">{val_text}</text>'
            )
        cls = str(row["cls"])
        cls_x = gx + group_w / 2
        out.append(
            f'<text x="{cls_x}" y="{top_margin + chart_h + 28}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, Arial" fill="{text_col}">{html.escape(cls)}</text>'
        )
        out.append(
            f'<text x="{cls_x}" y="{top_margin + chart_h + 46}" text-anchor="middle" font-size="11" font-family="DejaVu Sans, Arial" fill="{muted_col}">n={row["support"]}</text>'
        )

    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def _block_from_metrics_json(metrics: Dict[str, Any], prefix: str) -> Optional[Dict[str, Any]]:
    """从 eval metrics.json 顶层字段组装与 compute_per_class_metrics 输出兼容的结构。"""
    per_key = f"{prefix}_per_class"
    if per_key not in metrics or not isinstance(metrics[per_key], dict):
        return None
    return {
        "accuracy": float(metrics.get(f"{prefix}_accuracy", 0.0)),
        "macro_precision": float(metrics.get(f"{prefix}_macro_precision", 0.0)),
        "macro_recall": float(metrics.get(f"{prefix}_macro_recall", 0.0)),
        "macro_f1": float(metrics.get(f"{prefix}_macro_f1", 0.0)),
        "per_class": metrics[per_key],
    }


def _metrics_to_plot_task(task_title: str, m: Dict[str, Any], class_order: List[str]) -> _PlotTask:
    rows: List[Tuple[str, float, float, float, int]] = []
    pc = m["per_class"]
    for cls in class_order:
        d = pc.get(cls, {})
        rows.append(
            (
                cls,
                float(d.get("precision", 0.0)),
                float(d.get("recall", 0.0)),
                float(d.get("f1", 0.0)),
                int(d.get("support", 0)),
            )
        )
    return _PlotTask(
        task_name=task_title,
        accuracy=float(m["accuracy"]),
        macro_p=float(m["macro_precision"]),
        macro_r=float(m["macro_recall"]),
        macro_f1=float(m["macro_f1"]),
        per_class=rows,
    )


def _plot_from_metrics_file(
    metrics_path: str,
    out_dir: str,
    prefix: str,
) -> None:
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json 应为 JSON 对象")

    os.makedirs(out_dir, exist_ok=True)
    out_meta: Dict[str, Any] = {"source_metrics": metrics_path}

    m3 = _block_from_metrics_json(metrics, "type3")
    if m3 and m3["per_class"]:
        class_order = sorted(m3["per_class"].keys())
        plot3 = _metrics_to_plot_task("cell type (type3)", m3, class_order)
        svg3 = os.path.join(out_dir, f"{prefix}_type3_per_class.svg")
        _plot_vertical_grouped_svg(plot3, svg3)
        out_meta["type3"] = {"class_order": class_order, "svg": svg3}
        print(
            f"Type3 (from metrics): Acc={m3['accuracy']:.4f}  Macro F1={m3['macro_f1']:.4f}  -> {svg3}"
        )
    else:
        print("Warning: metrics.json 中缺少 type3_per_class，跳过 Type3 图")

    m4 = _block_from_metrics_json(metrics, "type4")
    if m4 and m4["per_class"]:
        class_order = sorted(m4["per_class"].keys())
        plot4 = _metrics_to_plot_task("cell origin (type4)", m4, class_order)
        svg4 = os.path.join(out_dir, f"{prefix}_type4_per_class.svg")
        _plot_vertical_grouped_svg(plot4, svg4)
        out_meta["type4"] = {"class_order": class_order, "svg": svg4}
        print(
            f"Type4 (from metrics): Acc={m4['accuracy']:.4f}  Macro F1={m4['macro_f1']:.4f}  -> {svg4}"
        )
    else:
        print("Warning: metrics.json 中缺少 type4_per_class，跳过 Type4 图")

    meta_path = os.path.join(out_dir, f"{prefix}_type34_plot_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)
    print(f"Plot meta: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="eval_result/stage2_layerGated/metrics.json",
        help="eval 输出的 metrics.json（含 type3_per_class / type4_per_class）；若文件存在则优先用此路径绘图",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="eval_result/eval_multitask/results.json",
        help="eval 输出的 results.json（列表，每项含 task_type / pred_class / gt_class）；在 metrics 文件不存在时使用",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_result/stage2_layerGated/plots",
        help="输出 SVG 与汇总 meta 的目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="stage2_layerGated",
        help="输出文件名前缀，如 <prefix>_type3_per_class.svg",
    )
    parser.add_argument(
        "--force_results",
        action="store_true",
        help="即使存在 metrics.json 也强制从 results.json 逐条重算",
    )
    parser.add_argument(
        "--exclude_type3_gt",
        type=str,
        default="GlandTumor,HPV",
        help="Type3：按 gt 排除的类别（逗号分隔）；传空字符串表示不排除、使用全量 7 类",
    )
    parser.add_argument(
        "--exclude_type4_gt",
        type=str,
        default="腺上皮成团",
        help="Type4：按 gt 排除的类别（逗号分隔）；传空字符串表示不排除、使用全量 5 类",
    )
    args = parser.parse_args()

    if not args.force_results and args.metrics_json and os.path.isfile(args.metrics_json):
        _plot_from_metrics_file(args.metrics_json, args.out_dir, args.prefix)
        return

    exclude3 = _parse_exclude_csv(args.exclude_type3_gt)
    exclude4 = _parse_exclude_csv(args.exclude_type4_gt)
    # 若未在命令行显式传空，且用户希望与历史「默认」一致：空串已解析为 frozenset()
    # 当两个参数都未给出时 argparse 会给默认串；只有传 "" 才得到全量
    type3_classes = _filtered_classes(TYPE3_CELL_TYPE_CLASSES_ZH_FULL, exclude3)
    type4_classes = _filtered_classes(TYPE4_SOURCE_CLASSES_ZH_FULL, exclude4)

    with open(args.results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results.json 应为 JSON 数组")

    type3_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    type4_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tt = item.get("task_type")
        if tt == "type3":
            type3_pairs.append((item.get("pred_class"), item.get("gt_class")))
        elif tt == "type4":
            type4_pairs.append((item.get("pred_class"), item.get("gt_class")))

    type3_pairs_raw_n = len(type3_pairs)
    type4_pairs_raw_n = len(type4_pairs)
    type3_pairs_f = _filter_pairs_by_gt(type3_pairs, exclude3)
    type4_pairs_f = _filter_pairs_by_gt(type4_pairs, exclude4)

    os.makedirs(args.out_dir, exist_ok=True)

    out_metrics: Dict[str, Any] = {
        "source": args.results_json,
        "exclude_type3_gt": sorted(exclude3),
        "exclude_type4_gt": sorted(exclude4),
        "type3_count_raw": type3_pairs_raw_n,
        "type4_count_raw": type4_pairs_raw_n,
        "type3_count_filtered": len(type3_pairs_f),
        "type4_count_filtered": len(type4_pairs_f),
        "type3_class_order": type3_classes,
        "type4_class_order": type4_classes,
    }

    if type3_pairs_f:
        m3 = compute_per_class_metrics(type3_pairs_f, type3_classes)
        out_metrics["type3"] = m3
        plot3 = _metrics_to_plot_task(
            "cell type",
            m3,
            type3_classes,
        )
        svg3 = os.path.join(args.out_dir, f"{args.prefix}_type3_per_class.svg")
        _plot_vertical_grouped_svg(plot3, svg3)
        print(
            f"Type3: raw={type3_pairs_raw_n} filtered={len(type3_pairs_f)} "
            f"exclude_gt={sorted(exclude3) or 'none'}  Acc={m3['accuracy']:.4f}  -> {svg3}"
        )
    else:
        print("Warning: Type3 过滤后无样本或未找到 type3")

    if type4_pairs_f:
        m4 = compute_per_class_metrics(type4_pairs_f, type4_classes)
        out_metrics["type4"] = m4
        plot4 = _metrics_to_plot_task(
            "cell origin",
            m4,
            type4_classes,
        )
        svg4 = os.path.join(args.out_dir, f"{args.prefix}_type4_per_class.svg")
        _plot_vertical_grouped_svg(plot4, svg4)
        print(
            f"Type4: raw={type4_pairs_raw_n} filtered={len(type4_pairs_f)} "
            f"exclude_gt={sorted(exclude4) or 'none'}  Acc={m4['accuracy']:.4f}  -> {svg4}"
        )
    else:
        print("Warning: Type4 过滤后无样本或未找到 type4")

    metrics_path = os.path.join(args.out_dir, f"{args.prefix}_type34_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
