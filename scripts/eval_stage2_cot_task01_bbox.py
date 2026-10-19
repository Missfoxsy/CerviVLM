#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅评估 Stage2 的 task0/task1，按 bbox 坐标计算定位指标（IoU/mAP）。

关键逻辑：
1) 预测输出中的 <regionX> -> 从待评估数据集 regions 中取对应 bbox
2) 原始数据集每图一个 GT bbox（不分 task_id），用于 task0/1 共用 GT
3) 若模型输出多个 region，选择与 GT IoU 最大的预测框参与该样本评估
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.mm_utils import prepare_inputs
from utils.data_utils import parse_grounding_output
from scripts.eval_stage2_cot import (
    load_checkpoint_model,
    maybe_enable_hierarchical_fusion,
    add_task_constraints_to_user_text,
    _is_official_vlm_fo1_model_dir,
    DEFAULT_VLM_FO1_BASE,
)


def _compute_iou_xyxy(box1, box2) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    a1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    a2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _map_from_best_iou(best_iou: float) -> float:
    iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    flags = [1.0 if best_iou >= t else 0.0 for t in iou_thresholds]
    return float(np.mean(flags))


def _extract_origin_bbox_from_text(text: str):
    """
    解析形如:
    <ref>阳性</ref><box>(525,577),(787,899)</box>
    """
    if not text:
        return None
    m = re.search(
        r"<box>\s*\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*,\s*\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*</box>",
        str(text),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    x1, y1, x2, y2 = map(float, m.groups())
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _extract_origin_image_basename(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") != "user":
            continue
        val = str(conv.get("value", ""))
        m = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", val)
        if m:
            return os.path.basename(m.group(1).strip())
    return ""


def _build_origin_gt_map(origin_json_path: str):
    with open(origin_json_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)

    by_id = {}
    by_basename = {}
    for item in origin_data:
        if not isinstance(item, dict):
            continue
        gt_bbox = None
        for conv in item.get("conversations", []):
            if conv.get("from") == "assistant":
                gt_bbox = _extract_origin_bbox_from_text(conv.get("value", ""))
                if gt_bbox is not None:
                    break
        if gt_bbox is None:
            continue
        sid = item.get("id")
        if sid:
            by_id[str(sid)] = gt_bbox
        bname = _extract_origin_image_basename(item)
        if bname:
            by_basename[bname] = gt_bbox
    return by_id, by_basename


def _pick_gt_bbox_for_eval_item(eval_item: dict, by_id: dict, by_basename: dict):
    sid = str(eval_item.get("id", ""))
    if sid in by_id:
        return by_id[sid]
    bname = os.path.basename(str(eval_item.get("image", "")))
    return by_basename.get(bname)


def _extract_user_text(eval_item: dict) -> str:
    for conv in eval_item.get("conversations", []):
        if conv.get("from") == "user":
            t = str(conv.get("value", "")).replace("<image>", "").strip()
            if t.startswith("\n"):
                t = t[1:]
            return t
    return ""


def evaluate_task01_bbox(
    model,
    tokenizer,
    image_processors,
    eval_dataset: CervicalDataset,
    eval_data_raw: list,
    origin_by_id: dict,
    origin_by_basename: dict,
    model_name: str,
    output_dir: str,
    device: str = "cuda",
    language: str = "zh",
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    counters = {
        "type0": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "maps": [], "n": 0},
        "type1": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "maps": [], "n": 0},
    }
    results = []

    for idx in tqdm(range(len(eval_data_raw)), desc="Eval task0/1 bbox", ncols=100):
        item = eval_data_raw[idx]
        task_id = item.get("task_id", None)
        if task_id not in (0, 1):
            continue
        task_type = f"type{task_id}"

        gt_bbox = _pick_gt_bbox_for_eval_item(item, origin_by_id, origin_by_basename)
        if gt_bbox is None:
            continue

        user_text = _extract_user_text(item)
        user_text = add_task_constraints_to_user_text(user_text, int(task_id), language=language)
        image_path = item["image"]
        _ = Image.open(image_path).convert("RGB")

        eval_sample = eval_dataset[idx]
        bbox_list_for_model = [r["bbox"] for r in item.get("regions", []) if isinstance(r.get("bbox"), list) and len(r["bbox"]) == 4]
        if not bbox_list_for_model:
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": user_text},
            ],
            "bbox_list": bbox_list_for_model,
        }]

        generation_kwargs = prepare_inputs(
            model_name=model_name,
            model=model,
            image_processors=image_processors,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_tokens=512,
            temperature=0.0,
            do_sample=False,
            path_feature_root=None,
            task_ids=[int(task_id)],
        )

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)
        output_text = tokenizer.decode(
            output_ids[0, generation_kwargs["inputs"].shape[1]:],
            skip_special_tokens=False,
        ).strip()

        pred_dict = parse_grounding_output(output_text)
        pred_indices = sorted({i for _, idxs in pred_dict.items() for i in idxs if isinstance(i, int)})
        pred_bboxes = [bbox_list_for_model[i] for i in pred_indices if 0 <= i < len(bbox_list_for_model)]

        best_iou = 0.0
        best_bbox = None
        for pb in pred_bboxes:
            iou = _compute_iou_xyxy(pb, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_bbox = pb

        ap = _map_from_best_iou(best_iou)
        if not pred_bboxes:
            tp, fp, fn = 0, 0, 1
        elif best_iou >= 0.5:
            tp, fp, fn = 1, 0, 0
        else:
            tp, fp, fn = 0, 1, 1

        counters[task_type]["tp"] += tp
        counters[task_type]["fp"] += fp
        counters[task_type]["fn"] += fn
        counters[task_type]["ious"].append(float(best_iou))
        counters[task_type]["maps"].append(float(ap))
        counters[task_type]["n"] += 1

        results.append({
            "id": item.get("id", f"sample_{idx}"),
            "task_type": task_type,
            "image": image_path,
            "prediction": output_text,
            "pred_indices": pred_indices,
            "pred_bboxes": pred_bboxes,
            "best_pred_bbox": best_bbox,
            "gt_bbox": gt_bbox,
            "best_iou": float(best_iou),
            "map_50_95": float(ap),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    metrics = {}
    for t in ("type0", "type1"):
        tp = counters[t]["tp"]
        fp = counters[t]["fp"]
        fn = counters[t]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(counters[t]["ious"])) if counters[t]["ious"] else 0.0
        map_50_95 = float(np.mean(counters[t]["maps"])) if counters[t]["maps"] else 0.0

        metrics[f"{t}_samples"] = counters[t]["n"]
        metrics[f"{t}_precision@0.5"] = precision
        metrics[f"{t}_recall@0.5"] = recall
        metrics[f"{t}_f1@0.5"] = f1
        metrics[f"{t}_mean_iou"] = mean_iou
        metrics[f"{t}_map_50_95"] = map_50_95
        metrics[f"{t}_tp"] = tp
        metrics[f"{t}_fp"] = fp
        metrics[f"{t}_fn"] = fn

    metrics["total_samples_evaluated"] = len(results)
    metrics["note"] = (
        "GT bbox 来自 origin_data；预测 bbox 来自 stage2 数据集 regions 与模型输出 <regionX> 对应。"
    )

    with open(os.path.join(output_dir, "results_task01_bbox.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "metrics_task01_bbox.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Task0/1 bbox 坐标评估结果")
    print("=" * 60)
    for t in ("type0", "type1"):
        print(f"\n{t}:")
        print(f"  Samples:      {metrics.get(f'{t}_samples', 0)}")
        print(f"  Precision@.5: {metrics.get(f'{t}_precision@0.5', 0):.4f}")
        print(f"  Recall@.5:    {metrics.get(f'{t}_recall@0.5', 0):.4f}")
        print(f"  F1@.5:        {metrics.get(f'{t}_f1@0.5', 0):.4f}")
        print(f"  Mean IoU:     {metrics.get(f'{t}_mean_iou', 0):.4f}")
        print(f"  mAP@.5:.95:   {metrics.get(f'{t}_map_50_95', 0):.4f}")
    print("\n保存文件：")
    print(f"  {os.path.join(output_dir, 'results_task01_bbox.json')}")
    print(f"  {os.path.join(output_dir, 'metrics_task01_bbox.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Task0/1 bbox 坐标评估（使用 origin_data GT）")
    parser.add_argument("--base_model", type=str, default="outputs/stage1_layerGated")
    parser.add_argument("--checkpoint", type=str, default="outputs/stage2_layerGated")
    parser.add_argument("--eval_json", type=str, default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json")
    parser.add_argument("--origin_json", type=str, default="/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json")
    parser.add_argument("--output_dir", type=str, default="eval_result/stage2_layerGated_task01")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--use_hierarchical_fusion", action="store_true")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="legacy",
        choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"],
        help="与 Stage1/Stage2 训练一致；加载含 modules_to_save 的 adapter 前需挂载同名 fusion 模块",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.eval_json, "r", encoding="utf-8") as f:
        eval_data_raw = json.load(f)
    origin_by_id, origin_by_basename = _build_origin_gt_map(args.origin_json)

    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        prefetch_hierarchical_fusion=args.use_hierarchical_fusion,
        hierarchical_num_heads=args.hierarchical_num_heads,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
    )
    maybe_enable_hierarchical_fusion(
        model=model,
        enabled=args.use_hierarchical_fusion,
        num_heads=args.hierarchical_num_heads,
        fusion_mode=args.hierarchical_fusion_mode,
    )

    model_name_for_template = (
        args.base_model if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model) else DEFAULT_VLM_FO1_BASE
    )
    eval_dataset = CervicalDataset(
        json_path=args.eval_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=model_name_for_template,
        max_regions=100,
        image_size=896,
        is_train=False,
        verbose=False,
        path_feature_root=None,
        language=args.language,
    )

    evaluate_task01_bbox(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        eval_dataset=eval_dataset,
        eval_data_raw=eval_data_raw,
        origin_by_id=origin_by_id,
        origin_by_basename=origin_by_basename,
        model_name=model_name_for_template,
        output_dir=args.output_dir,
        device=args.device,
        language=args.language,
    )


if __name__ == "__main__":
    main()

