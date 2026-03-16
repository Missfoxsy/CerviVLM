#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞 Stage-1 对齐模型评估脚本（不使用 path_feature）

评估三种任务：
- task_id = 0: 定位阴性宫颈细胞（<ground>/<objects>/<regionX>）
- task_id = 1: 根据文本描述“找出阴性细胞”进行定位
- task_id = 2: 对指定 <regionX> 区域的病理特征进行描述（caption）
"""

import os
import sys
import json
import argparse
from collections import defaultdict
import re

import torch
import numpy as np
from tqdm import tqdm

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False
    print("Warning: nltk or rouge-score not available. 请运行: pip install nltk rouge-score")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs
from utils.data_utils import parse_grounding_output


def get_task_type_from_item(data_item: dict) -> str:
    """根据 task_id 字段返回 'type0' / 'type1' / 'type2' / 'unknown'。"""
    task_id = data_item.get("task_id", None)
    if task_id is not None:
        try:
            tid = int(task_id)
            if 0 <= tid <= 2:
                return f"type{tid}"
        except Exception:
            pass
    return "unknown"


def calculate_map(pred_indices_set: set, gt_indices_set: set, iou_thresholds=None) -> float:
    """与 eval_cervical_stage2.py 中一致的 mAP 计算逻辑。"""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    aps = []
    for thr in iou_thresholds:
        tp = len(pred_indices_set & gt_indices_set)
        fp = len(pred_indices_set - gt_indices_set)
        fn = len(gt_indices_set - pred_indices_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        aps.append(precision)
    return np.mean(aps) if len(aps) > 0 else 0.0


def evaluate_localization_task(pred_dict: dict, gt_dict: dict) -> dict:
    """
    定位任务评估（type0 / type1），索引级别的 IoU=1/0 模型，与 Stage-2 一致。
    """
    pred_indices_set = set()
    for _, idxs in pred_dict.items():
        pred_indices_set.update(idxs)

    gt_indices_set = set()
    for _, idxs in gt_dict.items():
        gt_indices_set.update(idxs)

    tp = len(pred_indices_set & gt_indices_set)
    fp = len(pred_indices_set - gt_indices_set)
    fn = len(gt_indices_set - pred_indices_set)

    all_ious = []
    for pred_idx in pred_indices_set:
        all_ious.append(1.0 if pred_idx in gt_indices_set else 0.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    map_value = calculate_map(pred_indices_set, gt_indices_set)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "map": map_value,
        "pred_indices": sorted(pred_indices_set),
        "gt_indices": sorted(gt_indices_set),
    }


def evaluate_caption_task(pred_text: str, gt_text: str) -> dict:
    """区域描述任务（type2）评估，使用 BLEU 和 ROUGE-L。"""
    result = {
        "pred_text": pred_text,
        "gt_text": gt_text,
        "bleu": 0.0,
        "rouge_l": 0.0,
    }

    if not BLEU_ROUGE_AVAILABLE:
        return result

    # 清理 chatml/多模态标签
    def _clean(s: str) -> str:
        s = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", str(s))
        s = re.sub(r"<ground>.*?</ground>", "", s, flags=re.DOTALL)
        s = re.sub(r"<objects>.*?</objects>", "", s, flags=re.DOTALL)
        s = re.sub(r"<region\d+>", "", s)
        return s.strip()

    pred_clean = _clean(pred_text)
    gt_clean = _clean(gt_text)

    # BLEU
    smoothie = SmoothingFunction().method4
    ref = [gt_clean.split()]
    hyp = pred_clean.split()
    try:
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoothie)
    except Exception:
        bleu = 0.0

    # ROUGE-L
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(gt_clean, pred_clean)
        rouge_l = scores["rougeL"].fmeasure
    except Exception:
        rouge_l = 0.0

    result["bleu"] = bleu
    result["rouge_l"] = rouge_l
    return result


def evaluate_stage1(model, tokenizer, image_processors, dataset, device="cuda", output_dir="eval_stage1_align", model_name=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    metrics = {}
    task_counts = defaultdict(int)

    # type0/1：定位；type2：caption
    task_metrics = {
        "type0": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type1": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type2": {"bleus": [], "rouge_ls": []},
    }

    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating Stage-1", ncols=100):
        data_item = dataset.data[idx]
        task_type = get_task_type_from_item(data_item)
        task_counts[task_type] += 1

        image_path = data_item["image"]
        user_text = ""
        assistant_text = ""
        for conv in data_item.get("conversations", []):
            if conv.get("from") == "user":
                user_text = conv.get("value", "")
            elif conv.get("from") == "assistant":
                assistant_text = conv.get("value", "")

        # 构造 messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": user_text},
                ],
                "bbox_list": [r["bbox"] for r in data_item.get("regions", [])],
            }
        ]

        generation_kwargs = prepare_inputs(
            model_name=model_name or "qwen2.5-vl",
            model=model,
            image_processors=image_processors,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_tokens=512,
            temperature=0.0,
            do_sample=False,
            image_size=896,
            path_feature_root=None,  # 显式禁用 path_feature
        )

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        output_text = tokenizer.decode(
            output_ids[0, generation_kwargs["inputs"].shape[1] :],
            skip_special_tokens=False,
        ).strip()

        rec = {
            "id": data_item.get("id", f"sample_{idx}"),
            "image": image_path,
            "task_type": task_type,
            "user_text": user_text,
            "prediction": output_text,
            "ground_truth": assistant_text,
        }

        # 任务评估
        if task_type in ["type0", "type1"]:
            pred_dict = parse_grounding_output(output_text)
            gt_dict = parse_grounding_output(assistant_text)
            m = evaluate_localization_task(pred_dict, gt_dict)
            t = task_metrics[task_type]
            t["tp"] += m["tp"]
            t["fp"] += m["fp"]
            t["fn"] += m["fn"]
            t["all_ious"].extend(
                [1.0 if idx in m["gt_indices"] else 0.0 for idx in m["pred_indices"]]
            )
            t["maps"].append(m["map"])
            rec.update(m)

        elif task_type == "type2":
            cap = evaluate_caption_task(output_text, assistant_text)
            task_metrics["type2"]["bleus"].append(cap["bleu"])
            task_metrics["type2"]["rouge_ls"].append(cap["rouge_l"])
            rec.update(cap)

        results.append(rec)

    # 汇总指标
    for t in ["type0", "type1"]:
        m = task_metrics[t]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(m["all_ious"]) if len(m["all_ious"]) > 0 else 0.0
        mean_map = np.mean(m["maps"]) if len(m["maps"]) > 0 else 0.0

        metrics[f"{t}_precision"] = precision
        metrics[f"{t}_recall"] = recall
        metrics[f"{t}_f1"] = f1
        metrics[f"{t}_mean_iou"] = mean_iou
        metrics[f"{t}_map"] = mean_map
        metrics[f"{t}_tp"] = tp
        metrics[f"{t}_fp"] = fp
        metrics[f"{t}_fn"] = fn

    if len(task_metrics["type2"]["bleus"]) > 0:
        metrics["type2_mean_bleu"] = float(np.mean(task_metrics["type2"]["bleus"]))
        metrics["type2_mean_rouge_l"] = float(np.mean(task_metrics["type2"]["rouge_ls"]))

    # 保存结果
    results_file = os.path.join(output_dir, "results_stage1.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics_file = os.path.join(output_dir, "metrics_stage1.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n" + "=" * 60)
    print("Stage-1 评估结果汇总")
    print("=" * 60)
    print("\n任务分布:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} 个样本")

    for tname, cname in [("type0", "定位任务0"), ("type1", "定位任务1")]:
        print(f"\n{cname} 指标 ({tname}):")
        print(f"  Precision: {metrics.get(f'{tname}_precision', 0):.4f}")
        print(f"  Recall:    {metrics.get(f'{tname}_recall', 0):.4f}")
        print(f"  F1:        {metrics.get(f'{tname}_f1', 0):.4f}")
        print(f"  Mean IoU:  {metrics.get(f'{tname}_mean_iou', 0):.4f}")
        print(f"  mAP:       {metrics.get(f'{tname}_map', 0):.4f}")

    print("\nType2 (区域描述) 指标:")
    print(f"  Mean BLEU:    {metrics.get('type2_mean_bleu', 0):.4f}")
    print(f"  Mean ROUGE-L: {metrics.get('type2_mean_rouge_l', 0):.4f}")
    print("\n" + "=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM-FO1 Stage-1 alignment model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/cervical_stage1_align",
        help="Stage-1 训练得到的模型目录（包含 config.json/model.safetensors 等）",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json",
        help="Stage-1 验证集 JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result/stage1_align",
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="评估设备（cuda 或 cpu）",
    )

    args = parser.parse_args()

    print("加载 Stage-1 模型:", args.base_model)
    tokenizer, model, image_processors = load_pretrained_model(
        args.base_model, load_8bit=False, load_4bit=False, device=args.device
    )

    print("加载测试集:", args.test_json)
    if not os.path.exists(args.test_json):
        print(f"Error: 测试集文件不存在: {args.test_json}")
        return

    test_dataset = CervicalDataset(
        json_path=args.test_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=args.base_model,
        max_regions=100,
        image_size=896,
        is_train=False,
        verbose=False,
        path_feature_root=None,
    )

    print(f"测试集大小: {len(test_dataset)}")

    evaluate_stage1(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        dataset=test_dataset,
        device=args.device,
        output_dir=args.output_dir,
        model_name=args.base_model,
    )

    print(f"\n评估完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

