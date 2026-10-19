#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段1（对齐训练）评估脚本。

与 ``eval_stage2_cot.py`` 共用 ``evaluate_model``，但默认只评估 **task_id 0,1,2**
（与 stage1 训练数据一致：定位 / 描述定位 / 区域描述）。

在启用 ``--use_hierarchical_fusion`` 时，同样会汇总 ``last_gates`` 并写入
``metrics.json`` / ``gate_stats.json``，便于查看 Stage1 训练后门控状态。
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from scripts.eval_stage2_cot import (
    load_checkpoint_model,
    maybe_enable_hierarchical_fusion,
    evaluate_model,
    _build_origin_gt_map,
    _is_official_vlm_fo1_model_dir,
    DEFAULT_VLM_FO1_BASE,
)


def _parse_stage1_task_ids(s: str):
    """仅保留 0,1,2；Stage1 数据不含 3/4/5。"""
    allowed = {0, 1, 2}
    ids = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            v = int(p)
        except Exception:
            continue
        if v in allowed:
            ids.append(v)
        elif v in (3, 4, 5):
            print(
                f"Warning: task_id={v} 不属于 Stage1 训练任务，已忽略。"
                " Stage1 评估仅支持 0,1,2。"
            )
    if not ids:
        return [0, 1, 2]
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser(
        description="阶段1 对齐评估（默认任务 0,1,2；门控统计与 eval_stage2_cot 一致）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/stage1_layerGated_open",
        help="通常为 Stage1 输出目录（LoRA adapter）；与 checkpoint 可相同",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/stage1_layerGated_open",
        help="与 base 相同时仅加载 Stage1 adapter",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json",
        help="Stage1 验证集 JSON（与 train_config_stage1_lora.yaml 中 val_json 对齐）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result/stage1_layerGated_open",
        help="评估输出目录（results.json, metrics.json, gate_stats.json）",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_feature_root", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument(
        "--use_hierarchical_fusion",
        action="store_true",
        help="启用层级融合并加载/统计门控（与 Stage1 训练一致时请加此选项）",
    )
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="legacy",
        choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"],
        help="层级融合模式；FPN 消融请使用 legacy（保留注意力、无门控）",
    )
    parser.add_argument("--save_hierarchical_attn_dir", type=str, default=None)
    parser.add_argument("--save_hierarchical_attn_every", type=int, default=0)
    parser.add_argument("--save_hierarchical_attn_per_task_max", type=int, default=10)
    parser.add_argument("--save_hierarchical_attn_seed", type=int, default=42)
    parser.add_argument(
        "--task_ids",
        type=str,
        default="0,1,2",
        help="仅支持 0,1,2",
    )
    parser.add_argument("--origin_json", type=str, default=None)
    parser.add_argument("--save_every_n_samples", type=int, default=1000)
    parser.add_argument(
        "--fpn_num_levels",
        type=int,
        default=None,
        help="VT FPN 层数消融（1/2/4/6）；默认从 checkpoint 下 fpn_ablation.json 读取",
    )
    args = parser.parse_args()

    selected_task_ids = _parse_stage1_task_ids(args.task_ids)

    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        prefetch_hierarchical_fusion=args.use_hierarchical_fusion,
        hierarchical_num_heads=args.hierarchical_num_heads,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
        fpn_num_levels=args.fpn_num_levels,
    )
    maybe_enable_hierarchical_fusion(
        model,
        enabled=args.use_hierarchical_fusion,
        num_heads=args.hierarchical_num_heads,
        fusion_mode=args.hierarchical_fusion_mode,
    )
    model_name_for_template = (
        args.base_model
        if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model)
        else DEFAULT_VLM_FO1_BASE
    )

    print(f"\n[Stage1 Eval] 加载验证集: {args.test_json}")
    if not os.path.exists(args.test_json):
        print(f"Error: 验证集不存在: {args.test_json}")
        return

    test_dataset = CervicalDataset(
        json_path=args.test_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=model_name_for_template,
        max_regions=100,
        image_size=896,
        is_train=False,
        verbose=False,
        path_feature_root=args.path_feature_root,
        language=args.language,
    )
    print(f"[Stage1 Eval] 样本数: {len(test_dataset)}  评估任务: {selected_task_ids}")

    origin_by_id, origin_by_basename = {}, {}
    if args.origin_json:
        if os.path.exists(args.origin_json):
            origin_by_id, origin_by_basename = _build_origin_gt_map(args.origin_json)
            print(
                f"[Stage1 Eval] 已加载 origin GT bbox: by_id={len(origin_by_id)}, "
                f"by_basename={len(origin_by_basename)}"
            )
        else:
            print(
                f"Warning: origin_json 不存在，task0/1 将回退为 test_json 的 regions: {args.origin_json}"
            )

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        dataset=test_dataset,
        device=args.device,
        output_dir=args.output_dir,
        model_name=model_name_for_template,
        path_feature_root=args.path_feature_root,
        visualize=args.visualize,
        language=args.language,
        save_hierarchical_attn_dir=args.save_hierarchical_attn_dir,
        save_hierarchical_attn_every=args.save_hierarchical_attn_every,
        save_hierarchical_attn_per_task_max=args.save_hierarchical_attn_per_task_max,
        save_hierarchical_attn_seed=args.save_hierarchical_attn_seed,
        task_ids_to_eval=selected_task_ids,
        origin_by_id=origin_by_id,
        origin_by_basename=origin_by_basename,
        save_every_n_samples=args.save_every_n_samples,
        run_title="阶段1 对齐评估汇总（任务 0,1,2）",
    )
    print(f"\n[Stage1 Eval] 完成，结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
