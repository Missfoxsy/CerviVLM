#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PS3C 二分类评估：阳性 / 阴性。

指标：Accuracy、Macro Precision/Recall/F1、每类 P/R/F1/Support、混淆矩阵。
"""

import argparse
import gc
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.ps3c_classification_dataset import (
    PS3C_CLASSES,
    PS3C_TASK_ID,
    PS3CClassificationDataset,
    extract_label_from_item,
    parse_ps3c_label,
)
from vlm_fo1.mm_utils import prepare_inputs
from scripts.eval_stage2_cot import (
    DEFAULT_VLM_FO1_BASE,
    _is_official_vlm_fo1_model_dir,
    _unwrap_hierarchical_fusion_module,
    compute_per_class_metrics,
    load_checkpoint_model,
    maybe_enable_hierarchical_fusion,
)


def _extract_user_text(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            t = str(conv.get("value", "")).replace("<image>", "").strip()
            if t.startswith("\n"):
                t = t[1:]
            return t
    return ""


def _clear_hf_gate_cache(model):
    try:
        core = model.get_model() if hasattr(model, "get_model") else model
        extractor = getattr(core, "object_vp_extractor", None)
        hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
        hf_inner = _unwrap_hierarchical_fusion_module(hf)
        if hf_inner is not None:
            hf_inner.last_gates = None
            hf_inner.last_gates_raw = None
    except Exception:
        pass


def _save_eval_progress(output_dir, results, pred_gt_pairs, metrics_extra=None):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if pred_gt_pairs:
        partial_metrics = compute_per_class_metrics(pred_gt_pairs, list(PS3C_CLASSES))
        partial_metrics["total"] = len(pred_gt_pairs)
        partial_metrics["correct"] = sum(1 for p, g in pred_gt_pairs if p == g and p is not None)
        if metrics_extra:
            partial_metrics.update(metrics_extra)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(partial_metrics, f, ensure_ascii=False, indent=2)


def evaluate_ps3c(
    model,
    tokenizer,
    image_processors,
    test_json: str,
    output_dir: str,
    base_model_path: str,
    device: str = "cuda",
    max_samples: int = 0,
    start_idx: int = 0,
    image_size: int = 896,
    save_every_n: int = 200,
    use_hierarchical_fusion: bool = True,
    hierarchical_num_heads: int = 8,
    hierarchical_fusion_mode: str = "task_adaptive",
    enrich_cache_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    model_name = (
        base_model_path
        if base_model_path and _is_official_vlm_fo1_model_dir(base_model_path)
        else DEFAULT_VLM_FO1_BASE
    )

    test_dataset = PS3CClassificationDataset(
        json_path=test_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=model_name,
        max_regions=1,
        image_size=896,
        is_train=False,
        verbose=False,
        path_feature_root=None,
        language="zh",
        enrich_cache_dir=enrich_cache_dir,
    )

    n_total = len(test_dataset.data)
    n_iter = n_total if int(max_samples) <= 0 else min(n_total, int(max_samples))
    start_idx = max(0, int(start_idx))
    print(f"[PS3C Eval] 测试样本: {n_iter}/{n_total}, start_idx={start_idx}, image_size={image_size}")

    results = []
    pred_gt_pairs = []
    results_path = os.path.join(output_dir, "results.json")
    if start_idx > 0 and os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        pred_gt_pairs = [(r.get("pred_label"), r.get("ground_truth")) for r in results]
        print(f"[PS3C Eval] 续评: 已加载 {len(results)} 条历史结果")

    for idx in tqdm(range(start_idx, n_iter), desc="PS3C classification", ncols=100, initial=start_idx, total=n_iter):
        item = test_dataset.data[idx]
        gt_label = extract_label_from_item(item)
        if gt_label is None:
            continue

        user_text = _extract_user_text(item)
        image_path = item["image"]
        bbox_list = [r["bbox"] for r in item.get("regions", []) if isinstance(r.get("bbox"), list)]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": user_text},
            ],
            "bbox_list": bbox_list,
        }]

        _clear_hf_gate_cache(model)

        output_ids = None
        generation_kwargs = prepare_inputs(
            model_name=model_name,
            model=model,
            image_processors=image_processors,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_tokens=64,
            temperature=0.0,
            do_sample=False,
            image_size=image_size,
            path_feature_root=None,
            task_ids=[PS3C_TASK_ID],
            verbose=False,
        )

        try:
            with torch.inference_mode():
                output_ids = model.generate(**generation_kwargs)
            output_text = tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
                skip_special_tokens=False,
            ).strip()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[PS3C Eval] OOM at idx={idx}, image={image_path}; 尝试更小 image_size={max(448, image_size // 2)}")
            generation_kwargs = prepare_inputs(
                model_name=model_name,
                model=model,
                image_processors=image_processors,
                tokenizer=tokenizer,
                messages=messages,
                device=device,
                max_tokens=64,
                temperature=0.0,
                do_sample=False,
                image_size=max(448, image_size // 2),
                path_feature_root=None,
                task_ids=[PS3C_TASK_ID],
                verbose=False,
            )
            with torch.inference_mode():
                output_ids = model.generate(**generation_kwargs)
            output_text = tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
                skip_special_tokens=False,
            ).strip()
        finally:
            del generation_kwargs
            if output_ids is not None:
                del output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pred_label = parse_ps3c_label(output_text)
        pred_gt_pairs.append((pred_label, gt_label))

        results.append({
            "id": item.get("id", f"sample_{idx}"),
            "image": image_path,
            "user_text": user_text,
            "ground_truth": gt_label,
            "prediction": output_text,
            "pred_label": pred_label,
            "correct": pred_label == gt_label and pred_label is not None,
        })

        if save_every_n > 0 and len(results) % int(save_every_n) == 0:
            _save_eval_progress(
                output_dir,
                results,
                pred_gt_pairs,
                metrics_extra={"dataset": os.path.basename(test_json), "partial": True},
            )

    metrics = compute_per_class_metrics(pred_gt_pairs, list(PS3C_CLASSES))
    metrics["total"] = len(pred_gt_pairs)
    metrics["correct"] = sum(1 for p, g in pred_gt_pairs if p == g and p is not None)
    metrics["dataset"] = os.path.basename(test_json)

    confusion = {a: {b: 0 for b in PS3C_CLASSES} for a in PS3C_CLASSES}
    for p, g in pred_gt_pairs:
        if g in confusion:
            row = confusion[g]
            key = p if p in row else "_invalid_"
            if key not in row:
                row[key] = 0
            row[key] = row.get(key, 0) + 1
    metrics["confusion"] = confusion
    metrics["invalid_predictions"] = sum(1 for p, _ in pred_gt_pairs if p not in PS3C_CLASSES)

    results_path = os.path.join(output_dir, "results.json")
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("PS3C 二分类评估结果")
    print("=" * 60)
    print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro P/R/F1:   {metrics['macro_precision']:.4f} / "
          f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"  Invalid pred:   {metrics['invalid_predictions']}")
    for cls in PS3C_CLASSES:
        pc = metrics.get("per_class", {}).get(cls, {})
        print(
            f"  [{cls}] P={pc.get('precision', 0):.4f} R={pc.get('recall', 0):.4f} "
            f"F1={pc.get('f1', 0):.4f} support={pc.get('support', 0)}"
        )
    print(f"\n  混淆矩阵 (行=GT, 列=Pred): {confusion}")
    print(f"\n  结果: {results_path}")
    print(f"  指标: {metrics_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PS3C 二分类评估")
    parser.add_argument("--base_model", type=str, default="outputs/stage2_layerGated_taskAdaptive")
    parser.add_argument("--checkpoint", type=str, required=True, help="few-shot 微调后的 checkpoint 目录")
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=0, help="0=全量测试集")
    parser.add_argument("--start_idx", type=int, default=0, help="从该下标续评（配合已有 results.json）")
    parser.add_argument("--image_size", type=int, default=896, help="推理短边缩放尺寸，与训练一致")
    parser.add_argument("--save_every_n", type=int, default=200, help="每 N 条保存一次中间结果")
    parser.add_argument("--use_hierarchical_fusion", action="store_true")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument("--hierarchical_fusion_mode", type=str, default="task_adaptive")
    parser.add_argument("--enrich_cache_dir", type=str, default="outputs/ps3c_fewshot_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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

    evaluate_ps3c(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        test_json=args.test_json,
        output_dir=args.output_dir,
        base_model_path=args.base_model,
        device=args.device,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        image_size=args.image_size,
        save_every_n=args.save_every_n,
        use_hierarchical_fusion=args.use_hierarchical_fusion,
        hierarchical_num_heads=args.hierarchical_num_heads,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
        enrich_cache_dir=args.enrich_cache_dir,
    )


if __name__ == "__main__":
    main()
