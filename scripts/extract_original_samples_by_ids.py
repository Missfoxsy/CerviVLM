#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据采样结果中的 id，从原始数据集中提取对应的全部任务样本。"
    )
    parser.add_argument(
        "--sampled_results",
        type=str,
        default="eval_result/stage2_layerGated/results_sampled10.json",
        help="采样后的结果 JSON（含 id 字段）",
    )
    parser.add_argument(
        "--origin_dataset",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
        help="原始数据集 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_result/stage2_layerGated/stage2_cot_val_by_sampled10_ids_all_tasks.json",
        help="输出 JSON 路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sampled_path = Path(args.sampled_results)
    origin_path = Path(args.origin_dataset)
    output_path = Path(args.output)

    if not sampled_path.exists():
        raise FileNotFoundError(f"采样结果不存在: {sampled_path}")
    if not origin_path.exists():
        raise FileNotFoundError(f"原始数据集不存在: {origin_path}")

    with sampled_path.open("r", encoding="utf-8") as f:
        sampled_data = json.load(f)
    with origin_path.open("r", encoding="utf-8") as f:
        origin_data = json.load(f)

    if not isinstance(sampled_data, list):
        raise ValueError("sampled_results 顶层必须是 list")
    if not isinstance(origin_data, list):
        raise ValueError("origin_dataset 顶层必须是 list")

    sampled_ids = {item.get("id") for item in sampled_data if isinstance(item, dict) and item.get("id")}
    if not sampled_ids:
        raise ValueError("在 sampled_results 中没有找到有效 id")

    filtered = [
        item
        for item in origin_data
        if isinstance(item, dict) and item.get("id") in sampled_ids
    ]

    # 统计每个 id 包含的 task_id，方便校验“一个 id 对应多个任务”是否完整
    id_to_task_ids = defaultdict(list)
    for item in filtered:
        sid = item.get("id")
        tid = item.get("task_id")
        id_to_task_ids[sid].append(tid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    task_counter = Counter(item.get("task_id") for item in filtered if isinstance(item, dict))
    print(f"采样结果 id 数量: {len(sampled_ids)}")
    print(f"匹配到的原始样本数: {len(filtered)}")
    print("按 task_id 统计:")
    for tid in sorted(task_counter.keys()):
        print(f"  task_id={tid}: {task_counter[tid]}")
    incomplete = [
        sid for sid, tids in id_to_task_ids.items()
        if len(set(tids)) < 6
    ]
    print(f"已保存: {output_path}")
    if incomplete:
        print(f"Warning: 有 {len(incomplete)} 个 id 在原始集中不足 6 个任务。")


if __name__ == "__main__":
    main()
