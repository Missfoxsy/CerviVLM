#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已有 results.json 重新计算正确的 type0/1/2 指标。

修复点：
  - type0/1: GT bbox 改为从 origin_data 加载（而非 regions[0].bbox）
  - type2: 重新计算 BLEU/ROUGE（nltk 可用时使用 smoothing_method1）

依赖：
    pip install rouge-score   # ROUGE 计算；若未装则跳过 ROUGE，仅重算 BLEU
    pip install nltk
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed, BLEU will be set to 0")

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed, ROUGE will be set to 0")


# ──────────────────────────────────────────────────────────────
# 1.  从 origin_data 提取 GT bbox
# ──────────────────────────────────────────────────────────────

def _extract_origin_bbox_from_text(text: str):
    """解析 <box>(x1,y1),(x2,y2)</box> 返回 [x1,y1,x2,y2] 或 None"""
    if not text:
        return None
    m = re.search(
        r"<box>\s*\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*,"
        r"\s*\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*</box>",
        str(text), flags=re.IGNORECASE,
    )
    if not m:
        return None
    coords = list(map(float, m.groups()))
    if coords[2] < coords[0]:
        coords[0], coords[2] = coords[2], coords[0]
    if coords[3] < coords[1]:
        coords[1], coords[3] = coords[3], coords[1]
    return coords


def _extract_origin_image_basename(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            val = str(conv.get("value", ""))
            m = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", val)
            if m:
                return os.path.basename(m.group(1).strip())
    return ""


def build_origin_gt_map(origin_json_path: str):
    """返回 (by_id, by_basename)"""
    by_id = {}
    by_basename = {}
    with open(origin_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id", ""))
        gt_bbox = None
        for conv in item.get("conversations", []):
            if conv.get("from") == "assistant":
                gt_bbox = _extract_origin_bbox_from_text(conv.get("value", ""))
                if gt_bbox is not None:
                    break
        if gt_bbox is None:
            continue
        if sid:
            by_id[sid] = gt_bbox
        bname = _extract_origin_image_basename(item)
        if bname:
            by_basename[bname] = gt_bbox
    return by_id, by_basename


def lookup_gt_bbox(sample_id: str, image_path: str, by_id: dict, by_basename: dict):
    """从 origin_data GT map 中查找 GT bbox。优先按 id 匹配，按 basename 回退。"""
    if sample_id in by_id:
        return by_id[sample_id]
    bname = os.path.basename(str(image_path))
    if bname in by_basename:
        return by_basename[bname]
    return None


# ──────────────────────────────────────────────────────────────
# 2.  IoU / mAP
# ──────────────────────────────────────────────────────────────

def compute_iou_xyxy(box1, box2) -> float:
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


def map_from_best_iou(best_iou: float) -> float:
    thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    flags = [1.0 if best_iou >= t else 0.0 for t in thresholds]
    return float(np.mean(flags))


# ──────────────────────────────────────────────────────────────
# 3.  BLEU / ROUGE 文本相似度（与 eval_stage2_cot.py evaluate_caption_task 一致）
# ──────────────────────────────────────────────────────────────

def _needs_char_tokenize(s: str) -> bool:
    if " " in s:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def evaluate_caption_task(pred_text: str, gt_text: str) -> dict:
    """与 eval_stage2_cot.py 中 evaluate_caption_task 完全一致"""
    result = {
        "pred_text": pred_text,
        "gt_text": gt_text,
        "bleu": 0.0,
        "rouge_l": 0.0,
        "rouge_1": 0.0,
        "rouge_2": 0.0,
    }
    if not NLTK_AVAILABLE and not ROUGE_AVAILABLE:
        return result

    pred_clean = re.sub(r"<\|im_end\|>", "", str(pred_text)).strip()
    pred_clean = re.sub(r"<\|im_start\|>", "", pred_clean).strip()
    gt_clean = re.sub(r"<\|im_end\|>", "", str(gt_text)).strip()
    gt_clean = re.sub(r"<\|im_start\|>", "", gt_clean).strip()

    # 移除 grounding 标签（type2 预测中可能残留）
    m = re.search(r"<ground>(.*?)</ground>", pred_clean, flags=re.DOTALL)
    if m is not None:
        pred_clean = m.group(1).strip()
    pred_clean = re.sub(r"<objects>.*?</objects>", "", pred_clean, flags=re.DOTALL).strip()
    pred_clean = re.sub(r"</?ground>", "", pred_clean).strip()
    pred_clean = re.sub(r"</?objects>", "", pred_clean).strip()
    pred_clean = re.sub(r"<region\d+>", "", pred_clean).strip()
    gt_clean = re.sub(r"<region\d+>", "", gt_clean).strip()

    if _needs_char_tokenize(pred_clean) or _needs_char_tokenize(gt_clean):
        pred_tokens = list(pred_clean)
        gt_tokens = list(gt_clean)
        pred_for_rouge = " ".join(pred_tokens)
        gt_for_rouge = " ".join(gt_tokens)
    else:
        pred_tokens = pred_clean.split()
        gt_tokens = gt_clean.split()
        pred_for_rouge = pred_clean
        gt_for_rouge = gt_clean

    if NLTK_AVAILABLE:
        try:
            smoothing = SmoothingFunction().method1
            result["bleu"] = sentence_bleu(
                [gt_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        except Exception:
            result["bleu"] = 0.0

    if pred_for_rouge.strip() and gt_for_rouge.strip() and ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            scores = scorer.score(gt_for_rouge, pred_for_rouge)
            result["rouge_1"] = scores["rouge1"].fmeasure
            result["rouge_2"] = scores["rouge2"].fmeasure
            result["rouge_l"] = scores["rougeL"].fmeasure
        except Exception:
            pass

    return result


# ──────────────────────────────────────────────────────────────
# 4.  主逻辑
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="从 results.json 重算 type0/1/2 指标")
    parser.add_argument("--results_json", default="eval_result/stage2_layerGated_taskAdaptive/results.json", help="已有评估结果 results.json")
    parser.add_argument("--stage2_json", default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json", help="stage2 评估集 JSON（用于 id->image 映射）")
    parser.add_argument("--origin_json", default="/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json", help="原始数据集 JSON（用于 GT bbox）")
    parser.add_argument("--output_json", default="metrics_corrected.json", help="输出 metrics 文件路径")
    args = parser.parse_args()

    # ── 加载数据 ──
    print(f"加载 results:   {args.results_json}")
    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"加载 stage2:    {args.stage2_json}")
    with open(args.stage2_json, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)

    print(f"加载 origin:    {args.origin_json}")
    origin_by_id, origin_by_basename = build_origin_gt_map(args.origin_json)
    print(f"  origin GT: by_id={len(origin_by_id)}, by_basename={len(origin_by_basename)}")

    # ── 构建 stage2 id -> image 映射 ──
    stage2_by_id = {str(item["id"]): item["image"] for item in stage2_data if "id" in item and "image" in item}

    # ── 重新聚合指标 ──
    task_metrics = {
        "type0": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type1": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type2": {"bleus": [], "rouge_ls": [], "rouge_1s": [], "rouge_2s": []},
    }
    task_counts = {}
    updated_results = []
    skipped = {"type0": 0, "type1": 0}

    for item in results:
        task_type = item.get("task_type", "unknown")
        if task_type not in ("type0", "type1", "type2"):
            updated_results.append(item)
            continue

        task_counts[task_type] = task_counts.get(task_type, 0) + 1

        if task_type in ("type0", "type1"):
            sample_id = str(item.get("id", ""))
            # 优先从 stage2_data 找 image path（results.json 中 image 路径可能为绝对路径）
            image_path = item.get("image") or stage2_by_id.get(sample_id, "")
            gt_bbox = lookup_gt_bbox(sample_id, image_path, origin_by_id, origin_by_basename)

            if gt_bbox is None:
                print(f"  Warning: {sample_id} ({task_type}) 在 origin_data 中找不到 GT bbox，跳过")
                skipped[task_type] += 1
                continue

            pred_bboxes = item.get("pred_bboxes", [])

            # 用真实 GT bbox 重新计算 best_iou
            best_iou = 0.0
            best_bbox = None
            for pb in pred_bboxes:
                iou = compute_iou_xyxy(pb, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox = pb

            ap = map_from_best_iou(best_iou)
            if not pred_bboxes:
                tp, fp, fn = 0, 0, 1
            elif best_iou >= 0.5:
                tp, fp, fn = 1, 0, 0
            else:
                tp, fp, fn = 0, 1, 1

            task_metrics[task_type]["tp"] += tp
            task_metrics[task_type]["fp"] += fp
            task_metrics[task_type]["fn"] += fn
            task_metrics[task_type]["all_ious"].append(float(best_iou))
            task_metrics[task_type]["maps"].append(float(ap))

            # 更新 results 中该样本的字段
            new_item = dict(item)
            new_item["gt_bbox"] = gt_bbox          # 替换为真实 GT
            new_item["best_pred_bbox"] = best_bbox
            new_item["best_iou"] = float(best_iou)
            new_item["map"] = float(ap)
            new_item["tp"] = tp
            new_item["fp"] = fp
            new_item["fn"] = fn
            updated_results.append(new_item)

        elif task_type == "type2":
            # 重新计算 BLEU/ROUGE
            pred_text = item.get("pred_text", item.get("prediction", ""))
            gt_text = item.get("gt_text", item.get("ground_truth", ""))
            metrics = evaluate_caption_task(pred_text, gt_text)

            task_metrics["type2"]["bleus"].append(metrics["bleu"])
            task_metrics["type2"]["rouge_ls"].append(metrics["rouge_l"])
            task_metrics["type2"]["rouge_1s"].append(metrics["rouge_1"])
            task_metrics["type2"]["rouge_2s"].append(metrics["rouge_2"])

            new_item = dict(item)
            new_item.update(metrics)
            updated_results.append(new_item)

    # ── 汇总 metrics ──
    metrics_out = {"task_distribution": {k: task_counts[k] for k in sorted(task_counts)}}

    for task_type in ("type0", "type1"):
        m = task_metrics[task_type]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(m["all_ious"])) if m["all_ious"] else 0.0
        map_val = float(np.mean(m["maps"])) if m["maps"] else 0.0

        metrics_out[f"{task_type}_precision"] = precision
        metrics_out[f"{task_type}_recall"] = recall
        metrics_out[f"{task_type}_f1"] = f1
        metrics_out[f"{task_type}_mean_iou"] = mean_iou
        metrics_out[f"{task_type}_map"] = map_val
        metrics_out[f"{task_type}_tp"] = tp
        metrics_out[f"{task_type}_fp"] = fp
        metrics_out[f"{task_type}_fn"] = fn

        n_total = task_counts.get(task_type, 0)
        n_skipped = skipped.get(task_type, 0)
        print(f"\n{task_type}: total={n_total}, skipped={n_skipped}, evaluated={n_total-n_skipped}")
        print(f"  Precision@0.5: {precision:.4f}")
        print(f"  Recall@0.5:     {recall:.4f}")
        print(f"  F1@0.5:        {f1:.4f}")
        print(f"  Mean IoU:      {mean_iou:.4f}")
        print(f"  mAP@0.5:.95:   {map_val:.4f}")

    if task_metrics["type2"]["bleus"]:
        metrics_out["type2_mean_bleu"] = float(np.mean(task_metrics["type2"]["bleus"]))
        metrics_out["type2_mean_rouge_l"] = float(np.mean(task_metrics["type2"]["rouge_ls"]))
        metrics_out["type2_mean_rouge_1"] = float(np.mean(task_metrics["type2"]["rouge_1s"]))
        metrics_out["type2_mean_rouge_2"] = float(np.mean(task_metrics["type2"]["rouge_2s"]))
        print(f"\ntype2: total={len(task_metrics['type2']['bleus'])}")
        print(f"  Mean BLEU:    {metrics_out['type2_mean_bleu']:.4f}")
        print(f"  Mean ROUGE-L: {metrics_out['type2_mean_rouge_l']:.4f}")
        print(f"  Mean ROUGE-1: {metrics_out['type2_mean_rouge_1']:.4f}")
        print(f"  Mean ROUGE-2: {metrics_out['type2_mean_rouge_2']:.4f}")

    # ── 保存 ──
    output_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics 保存至: {args.output_json}")

    corrected_results_json = os.path.join(output_dir, "results_corrected.json")
    with open(corrected_results_json, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, ensure_ascii=False, indent=2)
    print(f"Results 保存至: {corrected_results_json}")


if __name__ == "__main__":
    main()
