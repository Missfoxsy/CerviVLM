#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 SIPakMed 外部 JSON 上评估细胞来源二分类泛化性能。

数据格式：ShareGPT 风格 conversations，user 含 <image>，assistant 为来源类别之一。
模型若输出训练集 type4 细粒度标签「宫颈鳞状上皮表层」「宫颈鳞状上皮中层」，会归并为「宫颈鳞状上皮表层或中层」。
推理流程对齐 scripts/eval_stage2_cot.py（load_checkpoint_model、prepare_inputs、整图 bbox 作为单 ROI）。

默认将最短边缩放到 896（与 dataset/cervical_dataset 训练一致）；若 image_size 为 None/0 则仅按长边 2048 缩放，高分辨率整图易触发视觉编码器 OOM。

指标：accuracy、macro precision/recall/f1、各类别 precision/recall/f1/support。
"""

import argparse
import gc
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from eval_stage2_cot import (  # noqa: E402
    DEFAULT_VLM_FO1_BASE,
    _extract_class_from_text,
    _is_official_vlm_fo1_model_dir,
    add_task_constraints_to_user_text,
    compute_per_class_metrics,
    load_checkpoint_model,
    maybe_enable_hierarchical_fusion,
)
from vlm_fo1.mm_utils import prepare_inputs  # noqa: E402

# 与 sipakmed_source_dialog.json 候选标签一致；较长类别放前面，避免子串误匹配
SIPAKMED_SUPERFICIAL_MID = "宫颈鳞状上皮表层或中层"
SIPAKMED_SOURCE_CLASSES = [
    SIPAKMED_SUPERFICIAL_MID,
    "宫颈鳞状上皮底层",
]
# 训练集 type4 的表层/中层细粒度标签，评估时归并到 SIPAKMED_SUPERFICIAL_MID
SIPAKMED_SOURCE_ALIASES = [
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮表层",
]

DEFAULT_DATA_JSON = (
    "/path/to/dataset/sipakmed_source_dialog.json"
)


def _extract_source_class(text: str):
    """从模型输出中解析细胞来源类别；无法解析时返回 None。"""
    if not text:
        return None
    t = str(text).strip()
    cls = _extract_class_from_text(t, SIPAKMED_SOURCE_CLASSES)
    if cls is not None:
        return cls
    for alias in SIPAKMED_SOURCE_ALIASES:
        if t == alias or alias in t:
            return SIPAKMED_SUPERFICIAL_MID
    return None


def _load_dialog_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组，得到: {type(data)}")
    return data


def evaluate_sipakmed(
    model,
    tokenizer,
    image_processors,
    data: list,
    device: str,
    model_name: str,
    output_dir: str,
    task_id: int,
    language: str,
    path_feature_root,
    image_size,
    max_tokens: int,
    max_samples: int = None,
    save_every: int = 200,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    pairs = []
    results = []

    n = len(data) if max_samples is None else min(len(data), max_samples)
    print(f"[SIPakMed eval] image_size={image_size}（最短边目标；None 表示不按短边固定，易 OOM）")
    for idx in tqdm(range(n), desc="SIPakMed source", ncols=100):
        item = data[idx]
        image_path = item.get("image")
        if not image_path or not os.path.isfile(image_path):
            tqdm.write(f"[skip] 图像不存在: {image_path}")
            continue

        user_text = None
        assistant_text = None
        for conv in item.get("conversations", []):
            if conv.get("from") == "user":
                user_text = conv.get("value", "").replace("<image>", "").strip()
                if user_text.startswith("\n"):
                    user_text = user_text[1:]
            elif conv.get("from") == "assistant":
                assistant_text = conv.get("value", "")

        if user_text is None or assistant_text is None:
            tqdm.write(f"[skip] 缺 user/assistant: id={item.get('id', idx)}")
            continue

        user_text = add_task_constraints_to_user_text(user_text, task_id, language=language)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"[skip] 无法打开图像 {image_path}: {e}")
            continue

        w, h = image.size
        bbox_list_for_model = [[0.0, 0.0, float(w), float(h)]]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": user_text},
                ],
                "bbox_list": bbox_list_for_model,
            }
        ]

        try:
            generation_kwargs = prepare_inputs(
                model_name=model_name,
                model=model,
                image_processors=image_processors,
                tokenizer=tokenizer,
                messages=messages,
                device=device,
                max_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                path_feature_root=path_feature_root,
                task_ids=[int(task_id)],
                image_size=image_size,
            )

            with torch.inference_mode():
                output_ids = model.generate(**generation_kwargs)
            in_len = int(generation_kwargs["inputs"].shape[1])
            output_text = tokenizer.decode(
                output_ids[0, in_len:],
                skip_special_tokens=False,
            ).strip()
        except torch.cuda.OutOfMemoryError as e:
            tqdm.write(f"[OOM] skip id={item.get('id', idx)} image={image_path} ({e})")
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            output_text = ""
            pred_cls = None
            gt_cls = _extract_source_class(assistant_text)
            pairs.append((pred_cls, gt_cls))
            results.append(
                {
                    "id": item.get("id", f"sample_{idx}"),
                    "image": image_path,
                    "user_text": user_text,
                    "prediction_raw": "",
                    "pred_class": None,
                    "gt_class": gt_cls,
                    "ground_truth_raw": assistant_text,
                    "correct": False,
                    "error": "cuda_oom",
                }
            )
            if save_every > 0 and len(results) % save_every == 0:
                metrics = compute_per_class_metrics(pairs, SIPAKMED_SOURCE_CLASSES)
                _write_outputs(output_dir, results, metrics)
            continue

        pred_cls = _extract_source_class(output_text)
        gt_cls = _extract_source_class(assistant_text)
        pairs.append((pred_cls, gt_cls))

        results.append(
            {
                "id": item.get("id", f"sample_{idx}"),
                "image": image_path,
                "user_text": user_text,
                "prediction_raw": output_text,
                "pred_class": pred_cls,
                "gt_class": gt_cls,
                "ground_truth_raw": assistant_text,
                "correct": pred_cls == gt_cls and pred_cls is not None and gt_cls is not None,
            }
        )

        if save_every > 0 and len(results) % save_every == 0:
            metrics = compute_per_class_metrics(pairs, SIPAKMED_SOURCE_CLASSES)
            _write_outputs(output_dir, results, metrics)

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = compute_per_class_metrics(pairs, SIPAKMED_SOURCE_CLASSES)
    _write_outputs(output_dir, results, metrics)

    print("\n" + "=" * 60)
    print("SIPakMed 细胞来源 分类")
    print("=" * 60)
    print(f"  有效样本数: {metrics['total']}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro P/R/F1: P={metrics['macro_precision']:.4f} R={metrics['macro_recall']:.4f} F1={metrics['macro_f1']:.4f}")
    for cls in SIPAKMED_SOURCE_CLASSES:
        pc = metrics["per_class"].get(cls, {})
        print(
            f"    [{cls}] P={pc.get('precision', 0):.4f} R={pc.get('recall', 0):.4f} "
            f"F1={pc.get('f1', 0):.4f} support={pc.get('support', 0)}"
        )
    print("=" * 60)
    return metrics


def _write_outputs(output_dir: str, results: list, metrics: dict):
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    out_metrics = {
        "task": "sipakmed_cell_source",
        "classes": SIPAKMED_SOURCE_CLASSES,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SIPakMed 细胞来源 外部集评估")
    parser.add_argument(
        "--data_json",
        type=str,
        default=DEFAULT_DATA_JSON,
        help="对话格式 JSON 路径",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/stage1_layerGated_taskAdaptive",
        help="基座或 Stage1 输出（与 eval_stage2_cot 一致）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/stage2_layerGated_taskAdaptive",
        help="Stage2 checkpoint（LoRA 或完整权重目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result/sipakmed_source_stage2_layerGated_taskAdaptive",
        help="输出 results.json / metrics.json",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_feature_root", type=str, default=None)
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument(
        "--task_id",
        type=int,
        default=4,
        help="传给模型的 task_id（细胞来源分类对应 type4）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--image_size",
        type=int,
        default=896,
        help="传给 prepare_inputs 的最短边像素（与训练 CervicalDataset image_size=896 一致，显著降低显存）。设为 0 则关闭短边固定（等同 None，高分辨率易 OOM）",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="仅跑前 N 条，调试用")
    parser.add_argument("--save_every_n", type=int, default=200)
    parser.add_argument(
        "--no_hierarchical_fusion",
        action="store_true",
        help="关闭层级融合（默认开启，与 stage2_layerGated_taskAdaptive 评估一致）",
    )
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="task_adaptive",
        choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"],
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data_json):
        print(f"Error: 数据文件不存在: {args.data_json}")
        sys.exit(1)

    data = _load_dialog_json(args.data_json)

    use_hierarchical_fusion = not args.no_hierarchical_fusion
    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        prefetch_hierarchical_fusion=use_hierarchical_fusion,
        hierarchical_num_heads=args.hierarchical_num_heads,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
    )
    maybe_enable_hierarchical_fusion(
        model,
        enabled=use_hierarchical_fusion,
        num_heads=args.hierarchical_num_heads,
        fusion_mode=args.hierarchical_fusion_mode,
    )

    model_name_for_template = (
        args.base_model
        if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model)
        else DEFAULT_VLM_FO1_BASE
    )

    image_size = None if args.image_size == 0 else args.image_size

    evaluate_sipakmed(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        data=data,
        device=args.device,
        model_name=model_name_for_template,
        output_dir=args.output_dir,
        task_id=args.task_id,
        language=args.language,
        path_feature_root=args.path_feature_root,
        image_size=image_size,
        max_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        save_every=args.save_every_n,
    )
    print(f"\n完成，结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
