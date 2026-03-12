#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 阶段A验证集评估脚本（task_id=3/4/5/6）。

任务定义：
  - task_id=3: 7类细胞类型分类（准确率）
  - task_id=4: 5类来源分类（准确率）
  - task_id=5: 阳性/阴性二分类（准确率）
  - task_id=6: 细胞形态描述（关键词多标签 + 可选 BLEU/ROUGE）
"""

import os
import sys
import json
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


TASK3_CELL_TYPE_CLASSES = [
    "ASCUS",
    "ASC_H",
    "EpitheliumNormal",
    "GlandTumor",
    "HPV",
    "HSIL",
    "LSIL",
]

TASK4_SOURCE_CLASSES = [
    "上皮成团",
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮底层",
    "宫颈鳞状上皮表层",
    "腺上皮成团",
]

TASK5_BINARY_CLASSES = ["阳性", "阴性"]

TASK6_MORPH_KEYWORDS = ["核深染", "核增大", "核浆比高", "核不规则", "挖空", "核拥挤"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-A validation set for task_id=3/4/5/6.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="resources/VLM-FO1_Qwen2.5-VL-3B-v01",
        help="Path to base model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/cervical_PAfull_A",
        help="Path to stage-A checkpoint (full model or LoRA dir)",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/stage/stageA_val.json",
        help="Path to stage-A validation json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result/PAfull_A",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--path_feature_root",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/path_features",
        help="Pathology feature cache directory (*.pt)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap of evaluated samples (0 means all).",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save per-sample predictions json.",
    )
    return parser.parse_args()


def get_task_id(item: dict) -> int:
    tid = item.get("task_id", item.get("task_type", -1))
    try:
        return int(tid)
    except Exception:
        return -1


def extract_assistant_text(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "assistant":
            return str(conv.get("value", ""))
    return ""


def extract_user_text(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            return str(conv.get("value", ""))
    return ""


def clean_text_for_metric(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"<\|im_end\|>|<\|im_start\|>", "", s)
    s = re.sub(r"<objects>.*?</objects>", "", s, flags=re.DOTALL)
    s = re.sub(r"</?ground>|</?objects>", "", s)
    s = re.sub(r"<region\d+>", "", s)
    return s.strip()


def extract_class_from_text(text: str, valid_classes: List[str]) -> Optional[str]:
    if text is None:
        return None
    t = clean_text_for_metric(text)
    if t in valid_classes:
        return t
    for cls in valid_classes:
        if cls in t:
            return cls
    return None


def keywords_from_text(text: str, keywords: List[str]) -> set:
    t = clean_text_for_metric(text)
    return {kw for kw in keywords if kw in t}


def keyword_prf(pred_set: set, gt_set: set) -> Tuple[float, float, float]:
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def caption_metrics(pred_text: str, gt_text: str) -> Dict[str, float]:
    out = {"bleu": 0.0, "rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    if not BLEU_ROUGE_AVAILABLE:
        return out

    pred = clean_text_for_metric(pred_text)
    gt = clean_text_for_metric(gt_text)

    def needs_char_tokenize(s: str) -> bool:
        return (" " not in s) and (re.search(r"[\u4e00-\u9fff]", s) is not None)

    if needs_char_tokenize(pred) or needs_char_tokenize(gt):
        pred_tokens = list(pred)
        gt_tokens = list(gt)
        pred_for_rouge = " ".join(pred_tokens)
        gt_for_rouge = " ".join(gt_tokens)
    else:
        pred_tokens = pred.split()
        gt_tokens = gt.split()
        pred_for_rouge = pred
        gt_for_rouge = gt

    try:
        smoothing = SmoothingFunction().method1
        out["bleu"] = sentence_bleu([gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    except Exception:
        pass

    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(gt_for_rouge, pred_for_rouge)
        out["rouge_1"] = scores["rouge1"].fmeasure
        out["rouge_2"] = scores["rouge2"].fmeasure
        out["rouge_l"] = scores["rougeL"].fmeasure
    except Exception:
        pass
    return out


def load_checkpoint_model(base_model_path: str, checkpoint_path: str, device: str = "cuda"):
    print(f"\n加载基础模型: {base_model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        base_model_path, load_8bit=False, load_4bit=False, device=device
    )

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT 不可用，无法加载 LoRA，回退基础模型。")
            return tokenizer, model, image_processors
        print(f"加载 LoRA 权重: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        return tokenizer, model, image_processors

    # 非 LoRA 情况：兼容“阶段A完整模型目录”。
    # 注意：builder.load_pretrained_model 对 model_path 名称有校验（需包含 vlm-fo1），
    # 直接传 outputs/cervical_PAfull_A 可能报错，因此这里改为“基座模型 + 手动加载权重”。
    has_full_ckpt = (
        os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
        or os.path.exists(os.path.join(checkpoint_path, "model.safetensors.index.json"))
    )
    if has_full_ckpt:
        print(f"从 checkpoint 加载完整权重到基础模型: {checkpoint_path}")
        checkpoint_state = None
        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            checkpoint_state = torch.load(bin_path, map_location="cpu")
        else:
            index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
            try:
                from safetensors.torch import load_file
            except Exception:
                load_file = None
            if load_file is not None and os.path.exists(index_file):
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                checkpoint_state = {}
                weight_files = set(index.get("weight_map", {}).values())
                for weight_file in weight_files:
                    weight_path = os.path.join(checkpoint_path, weight_file)
                    if os.path.exists(weight_path):
                        checkpoint_state.update(load_file(weight_path))

        if checkpoint_state:
            model_state = {}
            for k, v in checkpoint_state.items():
                if not k.startswith("trainer_state") and not k.startswith("optimizer"):
                    model_state[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            print(
                f"完整权重加载完成: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )
        else:
            print("Warning: 未能读取完整 checkpoint 权重，回退基础模型。")
    else:
        print("Warning: 未检测到完整 checkpoint 或 LoRA 权重，使用基础模型。")
    return tokenizer, model, image_processors


def evaluate_stageA(
    model,
    tokenizer,
    image_processors,
    dataset: CervicalDataset,
    base_model: str,
    device: str,
    path_feature_root: Optional[str],
    max_samples: int = 0,
):
    model.eval()
    results = []

    # 聚合统计
    task_counts = defaultdict(int)
    task3 = {"correct": 0, "total": 0}
    task4 = {"correct": 0, "total": 0}
    task5 = {"correct": 0, "total": 0}
    task6 = {
        "precision": [],
        "recall": [],
        "f1": [],
        "exact_match": 0,
        "total": 0,
        "bleu": [],
        "rouge_1": [],
        "rouge_2": [],
        "rouge_l": [],
    }

    data_iter = dataset.data
    total = len(data_iter) if max_samples <= 0 else min(len(data_iter), max_samples)
    pbar = tqdm(range(total), desc="Evaluating Stage-A(3/4/5/6)")

    for idx in pbar:
        item = data_iter[idx]
        tid = get_task_id(item)
        if tid not in (3, 4, 5, 6):
            continue
        task_counts[str(tid)] += 1

        image_path = item.get("image", "")
        if not image_path or not os.path.exists(image_path):
            continue
        assistant_text = extract_assistant_text(item)
        user_text = extract_user_text(item).replace("<image>", "").strip()
        gt_regions = item.get("regions", [])
        bbox_list_for_model = [rg["bbox"] for rg in gt_regions if isinstance(rg.get("bbox", None), list) and len(rg["bbox"]) == 4]
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

        try:
            generation_kwargs = prepare_inputs(
                model_name=base_model,
                model=model,
                image_processors=image_processors,
                tokenizer=tokenizer,
                messages=messages,
                device=device,
                max_tokens=512,
                temperature=0.0,
                do_sample=False,
                path_feature_root=path_feature_root,
            )
            with torch.inference_mode():
                output_ids = model.generate(**generation_kwargs)
            output_text = tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
                skip_special_tokens=False,
            ).strip()
        except Exception as e:
            output_text = f"[GEN_ERROR] {type(e).__name__}: {e}"

        rec = {
            "id": item.get("id", f"sample_{idx}"),
            "task_id": tid,
            "prediction": output_text,
            "ground_truth": assistant_text,
        }

        if tid == 3:
            pred = extract_class_from_text(output_text, TASK3_CELL_TYPE_CLASSES)
            gt = extract_class_from_text(assistant_text, TASK3_CELL_TYPE_CLASSES)
            correct = pred is not None and gt is not None and pred == gt
            task3["total"] += 1
            task3["correct"] += int(correct)
            rec.update({"pred_class": pred, "gt_class": gt, "correct": bool(correct)})

        elif tid == 4:
            pred = extract_class_from_text(output_text, TASK4_SOURCE_CLASSES)
            gt = extract_class_from_text(assistant_text, TASK4_SOURCE_CLASSES)
            correct = pred is not None and gt is not None and pred == gt
            task4["total"] += 1
            task4["correct"] += int(correct)
            rec.update({"pred_class": pred, "gt_class": gt, "correct": bool(correct)})

        elif tid == 5:
            pred = extract_class_from_text(output_text, TASK5_BINARY_CLASSES)
            gt = extract_class_from_text(assistant_text, TASK5_BINARY_CLASSES)
            correct = pred is not None and gt is not None and pred == gt
            task5["total"] += 1
            task5["correct"] += int(correct)
            rec.update({"pred_class": pred, "gt_class": gt, "correct": bool(correct)})

        elif tid == 6:
            pred_set = keywords_from_text(output_text, TASK6_MORPH_KEYWORDS)
            gt_set = keywords_from_text(assistant_text, TASK6_MORPH_KEYWORDS)
            p, r, f1 = keyword_prf(pred_set, gt_set)
            em = pred_set == gt_set
            task6["precision"].append(p)
            task6["recall"].append(r)
            task6["f1"].append(f1)
            task6["exact_match"] += int(em)
            task6["total"] += 1

            cap = caption_metrics(output_text, assistant_text)
            task6["bleu"].append(cap["bleu"])
            task6["rouge_1"].append(cap["rouge_1"])
            task6["rouge_2"].append(cap["rouge_2"])
            task6["rouge_l"].append(cap["rouge_l"])
            rec.update({
                "pred_keywords": sorted(list(pred_set)),
                "gt_keywords": sorted(list(gt_set)),
                "kw_precision": p,
                "kw_recall": r,
                "kw_f1": f1,
                "kw_exact_match": bool(em),
                "bleu": cap["bleu"],
                "rouge_1": cap["rouge_1"],
                "rouge_2": cap["rouge_2"],
                "rouge_l": cap["rouge_l"],
            })

        results.append(rec)

    summary = {
        "task_distribution": dict(task_counts),
        "task3_accuracy": (task3["correct"] / task3["total"]) if task3["total"] > 0 else 0.0,
        "task3_correct": task3["correct"],
        "task3_total": task3["total"],
        "task4_accuracy": (task4["correct"] / task4["total"]) if task4["total"] > 0 else 0.0,
        "task4_correct": task4["correct"],
        "task4_total": task4["total"],
        "task5_accuracy": (task5["correct"] / task5["total"]) if task5["total"] > 0 else 0.0,
        "task5_correct": task5["correct"],
        "task5_total": task5["total"],
        "task6_kw_precision": float(np.mean(task6["precision"])) if len(task6["precision"]) > 0 else 0.0,
        "task6_kw_recall": float(np.mean(task6["recall"])) if len(task6["recall"]) > 0 else 0.0,
        "task6_kw_f1": float(np.mean(task6["f1"])) if len(task6["f1"]) > 0 else 0.0,
        "task6_kw_exact_match": (task6["exact_match"] / task6["total"]) if task6["total"] > 0 else 0.0,
        "task6_total": task6["total"],
        "task6_bleu": float(np.mean(task6["bleu"])) if len(task6["bleu"]) > 0 else 0.0,
        "task6_rouge_1": float(np.mean(task6["rouge_1"])) if len(task6["rouge_1"]) > 0 else 0.0,
        "task6_rouge_2": float(np.mean(task6["rouge_2"])) if len(task6["rouge_2"]) > 0 else 0.0,
        "task6_rouge_l": float(np.mean(task6["rouge_l"])) if len(task6["rouge_l"]) > 0 else 0.0,
        "bleu_rouge_available": BLEU_ROUGE_AVAILABLE,
    }
    return summary, results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.test_json):
        raise FileNotFoundError(f"测试集不存在: {args.test_json}")

    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model, args.checkpoint, device=args.device
    )

    print(f"\n加载验证集: {args.test_json}")
    dataset = CervicalDataset(
        json_path=args.test_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=args.base_model,
        max_regions=100,
        image_size=896,
        is_train=False,
        verbose=False,
        path_feature_root=args.path_feature_root,
    )
    print(f"验证集大小: {len(dataset)}")

    summary, results = evaluate_stageA(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        dataset=dataset,
        base_model=args.base_model,
        device=args.device,
        path_feature_root=args.path_feature_root,
        max_samples=args.max_samples,
    )

    summary_path = os.path.join(args.output_dir, "summary_stageA_346.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("Stage-A(任务3/4/5/6)评估结果")
    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 70)
    print(f"Summary 已保存: {summary_path}")

    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, "predictions_stageA_346.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"逐样本预测已保存: {pred_path}")


if __name__ == "__main__":
    main()

