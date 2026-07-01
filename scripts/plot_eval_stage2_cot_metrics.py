#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import html
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TaskMetrics:
    task_name: str
    accuracy: Optional[float]
    macro_p: Optional[float]
    macro_r: Optional[float]
    macro_f1: Optional[float]
    per_class: List[Tuple[str, float, float, float, int]]  # (cls, p, r, f1, support)


TYPE_HEADER_RE = re.compile(r"^Type(?P<tid>\d+)\s*\((?P<title>.*?)\)\s*指标:\s*$")
ACC_RE = re.compile(r"^\s*Accuracy:\s*(?P<acc>[0-9.]+)\s*\(")
MACRO_RE = re.compile(r"^\s*Macro P/R/F1:\s*P=(?P<p>[0-9.]+)\s+R=(?P<r>[0-9.]+)\s+F1=(?P<f>[0-9.]+)\s*$")
PER_CLASS_LINE_RE = re.compile(
    r"^\s*(?P<cls>.+?):\s*P=(?P<p>[0-9.]+)\s+R=(?P<r>[0-9.]+)\s+F1=(?P<f>[0-9.]+)\s+support=(?P<s>\d+)\s*$"
)


def _svg_plot_title(task_id: int, title_from_log: str) -> str:
    """与 plot_eval_multitask_type34_from_results.py 一致：Type3/4 用简短英文标题。"""
    if task_id == 3:
        return "cell type"
    if task_id == 4:
        return "cell origin"
    return f"Type{task_id} ({title_from_log})"


def parse_tasks_from_log(text: str, task_ids: List[int]) -> Dict[int, TaskMetrics]:
    lines = text.splitlines()
    results: Dict[int, TaskMetrics] = {}
    i = 0
    while i < len(lines):
        m = TYPE_HEADER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        tid = int(m.group("tid"))
        if tid not in task_ids:
            i += 1
            continue
        title = m.group("title").strip()

        accuracy = None
        macro_p = None
        macro_r = None
        macro_f1 = None
        per_class: List[Tuple[str, float, float, float, int]] = []

        j = i + 1
        while j < len(lines):
            line = lines[j].rstrip("\n")
            if TYPE_HEADER_RE.match(line.strip()) or line.strip().startswith("====") or line.strip().startswith("评估完成"):
                break
            ma = ACC_RE.match(line)
            if ma:
                accuracy = float(ma.group("acc"))
            mmc = MACRO_RE.match(line)
            if mmc:
                macro_p = float(mmc.group("p"))
                macro_r = float(mmc.group("r"))
                macro_f1 = float(mmc.group("f"))
            mp = PER_CLASS_LINE_RE.match(line)
            if mp:
                per_class.append(
                    (
                        mp.group("cls").strip(),
                        float(mp.group("p")),
                        float(mp.group("r")),
                        float(mp.group("f")),
                        int(mp.group("s")),
                    )
                )
            j += 1

        results[tid] = TaskMetrics(
            task_name=_svg_plot_title(tid, title),
            accuracy=accuracy,
            macro_p=macro_p,
            macro_r=macro_r,
            macro_f1=macro_f1,
            per_class=per_class,
        )
        i = j
    return results


def plot_task(task: TaskMetrics, out_path: str) -> None:
    if not task.per_class:
        raise ValueError(f"No per-class lines parsed for {task.task_name}")

    # --------- 纯 SVG（避免 numpy/matplotlib 在受限环境触发 OpenMP SHM 权限问题）---------
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

    # layout（竖向分组柱状图）
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

    # colors
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
        subtitle_parts.append(f"Macro(P/R/F1)={task.macro_p:.4f}/{task.macro_r:.4f}/{task.macro_f1:.4f}")
    subtitle = "  ".join(subtitle_parts)

    # SVG header
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # title
    out.append(f'<text x="{left_margin}" y="34" font-size="20" font-family="DejaVu Sans, Arial" fill="{text_col}">{html.escape(title)}</text>')
    if subtitle:
        out.append(f'<text x="{left_margin}" y="60" font-size="13" font-family="DejaVu Sans, Arial" fill="{muted_col}">{html.escape(subtitle)}</text>')

    # legend
    legend_x = left_margin + chart_w - 260
    legend_y = 30
    out.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="10" height="10" fill="{col_p}"/><text x="{legend_x+16}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">Precision</text>')
    out.append(f'<rect x="{legend_x+92}" y="{legend_y-10}" width="10" height="10" fill="{col_r}"/><text x="{legend_x+108}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">Recall</text>')
    out.append(f'<rect x="{legend_x+160}" y="{legend_y-10}" width="10" height="10" fill="{col_f1}"/><text x="{legend_x+176}" y="{legend_y}" font-size="12" font-family="DejaVu Sans, Arial" fill="{muted_col}">F1</text>')

    # grid + y-axis ticks (0~1)
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

    # bars (grouped vertical)
    for idx, row in enumerate(rows):
        gx = left_margin + idx * (group_w + group_gap)
        # center the group within group_w
        group_total_w = 3 * bar_w + 2 * bar_gap
        bx0 = gx + (group_w - group_total_w) / 2

        items = [
            ("P", row["p"], col_p),
            ("R", row["r"], col_r),
            ("F1", row["f1"], col_f1),
        ]
        for j, (name, val, col) in enumerate(items):
            x = bx0 + j * (bar_w + bar_gap)
            y = y_of(val)
            h = top_margin + chart_h - y
            out.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{max(0, h)}" fill="{col}"/>')
            # value label on top
            val_text = f"{clamp01(float(val)):.4f}"
            out.append(
                f'<text x="{x + bar_w/2}" y="{y - 6}" text-anchor="middle" font-size="11" font-family="DejaVu Sans, Arial" fill="{text_col}">{val_text}</text>'
            )

        # class label and support below
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


def main():
    parser = argparse.ArgumentParser(description="Plot per-class metrics from eval_stage2_cot.log.")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval_result/plots")
    parser.add_argument("--tasks", type=str, default="4,5", help="Comma-separated task ids to plot, e.g. 4,5")
    args = parser.parse_args()

    with open(args.log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    task_ids = []
    for t in args.tasks.split(","):
        t = t.strip()
        if not t:
            continue
        task_ids.append(int(t))

    parsed = parse_tasks_from_log(text, task_ids=task_ids)
    os.makedirs(args.out_dir, exist_ok=True)

    for tid in task_ids:
        if tid not in parsed:
            raise ValueError(f"Task {tid} not found in log: {args.log_path}")
        out_path = os.path.join(args.out_dir, f"stage2_cot_type{tid}_per_class.svg")
        plot_task(parsed[tid], out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

