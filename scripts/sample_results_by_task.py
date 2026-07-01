#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 results.json 按 task_type 随机采样并另存。"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eval_result/stage2_layerGated/results.json",
        help="输入 results.json 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_result/stage2_layerGated/results_sampled_10_per_task.json",
        help="输出采样后的 json 路径",
    )
    parser.add_argument(
        "--per_task",
        type=int,
        default=10,
        help="每个 task_type 采样数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if args.per_task <= 0:
        raise ValueError("--per_task 必须 > 0")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list")

    groups = defaultdict(list)
    for item in data:
        if not isinstance(item, dict):
            continue
        task_type = item.get("task_type", "unknown")
        groups[task_type].append(item)

    rng = random.Random(args.seed)
    sampled = []
    sampled_stat = {}
    for task_type in sorted(groups.keys()):
        items = groups[task_type]
        k = min(args.per_task, len(items))
        selected = rng.sample(items, k) if k > 0 else []
        sampled.extend(selected)
        sampled_stat[task_type] = {
            "available": len(items),
            "sampled": len(selected),
        }

    # 打乱合并后的样本顺序，避免同 task_type 连在一起
    rng.shuffle(sampled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"输入样本总数: {len(data)}")
    print(f"输出样本总数: {len(sampled)}")
    print("各任务采样统计:")
    for task_type in sorted(sampled_stat.keys()):
        stat = sampled_stat[task_type]
        print(f"  {task_type}: {stat['sampled']}/{stat['available']}")
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
