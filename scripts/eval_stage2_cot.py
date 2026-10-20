#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 阶段2 多任务评估脚本（任务 0,1,2,4,5）

支持五种任务类型（与 cervical_dataset 约束一致）：
- 类型0 (task_id=0)：基础定位 - IoU、Precision、Recall、F1、mAP
- 类型1 (task_id=1)：带描述的特征定位 - IoU、Precision、Recall、F1、mAP
- 类型2 (task_id=2)：区域描述 - BLEU、ROUGE、SBERT（英文对齐管线，默认开启）
- 类型4 (task_id=4)：来源分类 - 准确率 + 每个类别的 Precision/Recall/F1/Support
- 类型5 (task_id=5)：区域细胞类型+诊断依据（CoT） - 从【诊断结果】中提取类别，按分类任务计算准确率 + 每类别指标

评估时自动为 user 文本追加与 dataset/cervical_dataset.py 相同的任务约束。
"""

import os
import sys
import json
import argparse
import inspect
import re
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.flash_attn_compat  # noqa: F401  — hide broken flash_attn before transformers

import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False

from dataset.cervical_dataset import CervicalDataset, TASK_CONSTRAINT_TEXTS
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, draw_bboxes_and_save
from utils.data_utils import parse_grounding_output
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import (
    HierarchicalFeatureFusion,
    visualize_hierarchical_attention,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VLM_FO1_BASE = os.path.join(_PROJECT_ROOT, "resources", "VLM-FO1_Qwen2.5-VL-3B-v01")
DEFAULT_TASK2_EN_GT_REF = os.path.join(
    _PROJECT_ROOT, "eval_results", "en_gt_ref_stage2_cot_val.json"
)
DEFAULT_TASK2_SBERT_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


def _compute_and_attach_task2_sbert(
    results: list,
    metrics: dict,
    *,
    language: str,
    enabled: bool,
    en_gt_ref: Optional[str],
    sbert_model: str,
    sbert_device: str,
    output_dir: str,
) -> dict:
    """Post-process type2 samples with English-pipeline SBERT; update metrics + results.json."""
    if not enabled:
        return metrics
    has_type2 = any(
        (r.get("task_type") == "type2" or r.get("task_id_effective") == 2) for r in results
    )
    if not has_type2:
        return metrics

    try:
        from scripts.compute_task2_sbert_en import (
            augment_results_with_sbert,
            metrics_from_sbert_stats,
        )
    except Exception as e:
        print(f"[Eval][Task2 SBERT] import failed, skip: {e}")
        return metrics

    ref = en_gt_ref or DEFAULT_TASK2_EN_GT_REF
    if language == "zh" and (not ref or not os.path.exists(ref)):
        print(
            f"[Eval][Task2 SBERT] WARNING: EN GT ref missing ({ref}); "
            "will fall back to rule-translated Chinese GT."
        )

    try:
        # Free VLM GPU memory before loading SBERT when scoring on the same device.
        if isinstance(sbert_device, str) and sbert_device.startswith("cuda"):
            try:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        stats = augment_results_with_sbert(
            results,
            en_gt_ref=Path(ref) if ref else None,
            sbert_model=sbert_model or DEFAULT_TASK2_SBERT_MODEL,
            device=sbert_device,
            language=language,
            require_aligned_gt=False,
            log_prefix="Eval/Task2 SBERT",
        )
        if stats.get("skipped"):
            return metrics
        metrics.update(metrics_from_sbert_stats(stats))

        results_file = os.path.join(output_dir, "results.json")
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(
            f"[Eval][Task2 SBERT] mean={metrics.get('type2_mean_sbert', 0):.4f} "
            f"(n={metrics.get('type2_sbert_n', 0)}) → {metrics_file}"
        )
    except Exception as e:
        print(f"[Eval][Task2 SBERT] failed, keep BLEU/ROUGE only: {e}")
        import traceback

        traceback.print_exc()
    return metrics

# ---------------- 语言相关：Type3/4/5 候选类别集合 ----------------
# 说明：最终指标表的类别名要求“英文原样”，所以这里按 language 分开维护
TYPE3_CELL_TYPE_CLASSES_BY_LANG = {
    "zh": ["ASCUS", "ASC_H", "EpitheliumNormal", "HSIL", "LSIL"],
    "en": ["ASCUS", "ASC_H", "Normal", "HSIL", "LSIL"],
}

TYPE4_SOURCE_CLASSES_BY_LANG = {
    "zh": [
        "上皮成团",
        "宫颈鳞状上皮中层",
        "宫颈鳞状上皮底层",
        "宫颈鳞状上皮表层",
    ],
    "en": [
        "epithelial cluster",
        "cervical squamous mid layer",
        "cervical squamous basal layer",
        "cervical squamous superficial layer",
    ],
}

TYPE5_CELL_TYPE_CLASSES_BY_LANG = {
    # task5 的分类类别与 task3 保持一致（从 diagnosis 中抽取）
    "zh": TYPE3_CELL_TYPE_CLASSES_BY_LANG["zh"],
    "en": TYPE3_CELL_TYPE_CLASSES_BY_LANG["en"],
}

# 运行时根据 --language 选择（默认 zh）
TYPE3_CELL_TYPE_CLASSES = TYPE3_CELL_TYPE_CLASSES_BY_LANG["zh"]
TYPE4_SOURCE_CLASSES = TYPE4_SOURCE_CLASSES_BY_LANG["zh"]
TYPE5_CELL_TYPE_CLASSES = TYPE5_CELL_TYPE_CLASSES_BY_LANG["zh"]

try:
    from peft import LoraConfig, PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    PeftModel = None
    PEFT_AVAILABLE = False


def _peft_model_from_pretrained_compat(model, adapter_path: str):
    """
    加载 LoRA 权重。若 adapter_config.json 含当前已安装 peft 不支持的字段
    （如较新训练端写入的 alora_invocation_tokens），则按 LoraConfig 签名过滤后再加载。
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT 不可用")
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return PeftModel.from_pretrained(model, adapter_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if str(raw.get("peft_type", "LORA")).upper() != "LORA":
        return PeftModel.from_pretrained(model, adapter_path)
    sig = inspect.signature(LoraConfig.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    dropped = sorted(k for k in raw if k not in allowed)
    if dropped:
        print(f"[PEFT compat] 忽略 adapter_config 中当前 peft 不支持的键: {dropped}")
    try:
        lora_config = LoraConfig(**filtered)
    except TypeError as e:
        raise RuntimeError(
            f"无法根据 {cfg_path} 构造 LoraConfig: {e}。"
            "若 adapter 由较新版本训练产生，可尝试: pip install -U 'peft>=0.14'"
        ) from e
    return PeftModel.from_pretrained(model, adapter_path, config=lora_config)


def get_task_type_from_item(data_item: dict) -> str:
    """任务类型：支持 0,1,2,3,4,5。"""
    task_id = data_item.get("task_id", None)
    if task_id is not None:
        try:
            tid = int(task_id)
            if tid in (0, 1, 2, 3, 4, 5):
                return f"type{tid}"
        except (ValueError, TypeError):
            pass
    return "unknown"


def _expected_task_id_for_task_type(task_type: str):
    """type0..type5 -> 0..5；其它返回 None。"""
    if not isinstance(task_type, str) or not task_type.startswith("type"):
        return None
    try:
        v = int(task_type.replace("type", "", 1))
        return v if 0 <= v <= 5 else None
    except ValueError:
        return None


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
    """
    从 origin_data 构建的 GT map 中查找该样本的真实标注 bbox。

    查找顺序：按 id 精确匹配 > 按图片 basename 匹配。
    不再 fallback 到 stage2 数据集的 regions（regions[0].bbox 与模型输入一致，
    用它做 GT 会导致满分假象）。
    """
    sid = str(eval_item.get("id", ""))
    if sid in by_id:
        return by_id[sid]
    bname = os.path.basename(str(eval_item.get("image", "")))
    return by_basename.get(bname)


def _extract_class_from_text(text: str, valid_classes: list):
    if not text:
        return None
    t = str(text).strip()
    if t in valid_classes:
        return t
    for cls in valid_classes:
        if cls in t:
            return cls
    return None


def _extract_class_from_text_last(text: str, valid_classes: list):
    """更适用于 CoT：当文本包含“排除X...诊断结论Y”时，取最后一次出现的类别。"""
    if not text:
        return None
    t = str(text)
    best_cls = None
    best_pos = -1
    for cls in valid_classes:
        pos = t.rfind(cls)
        if pos != -1 and pos > best_pos:
            best_pos = pos
            best_cls = cls
    return best_cls


def _extract_diagnosis_section(text: str) -> str:
    """从模型/GT 输出中提取【诊断结果】/【诊断结论】/英文 Final diagnosis/D iagnosis 后的内容。"""
    if not text:
        return ""
    text = str(text).strip()
    # 匹配中文诊断段落
    m = re.search(r"(【诊断结果】|【诊断结论】)\s*(.*?)(?=【|$)", text, re.DOTALL)
    if m:
        return m.group(2).strip()

    # 匹配英文诊断段落（你的英文样例用 <Final diagnosis>）
    m2 = re.search(
        r"(<Final diagnosis>|<Final Diagnosis>|<Diagnosis>)\s*(.*?)(?=<|$)",
        text,
        flags=re.DOTALL,
    )
    if m2:
        return m2.group(2).strip()

    return text


def evaluate_caption_task(pred_text: str, gt_text: str) -> dict:
    """
    Type2 区域描述：与 scripts/eval_stage1_align.py 中 evaluate_caption_task 一致
    （清理模板、中文无空格且含汉字时按字符分词、BLEU method1、ROUGE use_stemmer=False）。
    """
    result = {
        "pred_text": pred_text,
        "gt_text": gt_text,
        "bleu": 0.0,
        "rouge_l": 0.0,
        "rouge_1": 0.0,
        "rouge_2": 0.0,
    }
    if not BLEU_ROUGE_AVAILABLE:
        return result
    pred_text_clean = re.sub(r"<\|im_end\|>", "", str(pred_text)).strip()
    pred_text_clean = re.sub(r"<\|im_start\|>", "", pred_text_clean).strip()
    gt_text_clean = re.sub(r"<\|im_end\|>", "", str(gt_text)).strip()
    gt_text_clean = re.sub(r"<\|im_start\|>", "", gt_text_clean).strip()
    m = re.search(r"<ground>(.*?)</ground>", pred_text_clean, flags=re.DOTALL)
    if m is not None:
        pred_text_clean = m.group(1).strip()
    pred_text_clean = re.sub(r"<objects>.*?</objects>", "", pred_text_clean, flags=re.DOTALL).strip()
    pred_text_clean = re.sub(r"</?ground>", "", pred_text_clean).strip()
    pred_text_clean = re.sub(r"</?objects>", "", pred_text_clean).strip()
    pred_text_clean = re.sub(r"<region\d+>", "", pred_text_clean).strip()
    gt_text_clean = re.sub(r"<region\d+>", "", gt_text_clean).strip()

    def _needs_char_tokenize(s: str) -> bool:
        if " " in s:
            return False
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    if _needs_char_tokenize(pred_text_clean) or _needs_char_tokenize(gt_text_clean):
        pred_tokens = list(pred_text_clean)
        gt_tokens = list(gt_text_clean)
        pred_text_for_rouge = " ".join(pred_tokens)
        gt_text_for_rouge = " ".join(gt_tokens)
    else:
        pred_tokens = pred_text_clean.split()
        gt_tokens = gt_text_clean.split()
        pred_text_for_rouge = pred_text_clean
        gt_text_for_rouge = gt_text_clean

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

    if not pred_text_for_rouge.strip() or not gt_text_for_rouge.strip():
        result["rouge_1"] = 0.0
        result["rouge_2"] = 0.0
        result["rouge_l"] = 0.0
    else:
        try:
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=False,
            )
            scores = scorer.score(gt_text_for_rouge, pred_text_for_rouge)
            result["rouge_1"] = scores["rouge1"].fmeasure
            result["rouge_2"] = scores["rouge2"].fmeasure
            result["rouge_l"] = scores["rougeL"].fmeasure
        except Exception:
            result["rouge_1"] = 0.0
            result["rouge_2"] = 0.0
            result["rouge_l"] = 0.0
    return result


def compute_per_class_metrics(pred_gt_pairs: list, valid_classes: list) -> dict:
    """
    pred_gt_pairs: [(pred_class, gt_class), ...]，None 表示未识别。
    返回：accuracy, per_class {cls: {precision, recall, f1, support}}, macro_precision, macro_recall, macro_f1
    """
    total = len(pred_gt_pairs)
    correct = sum(1 for p, g in pred_gt_pairs if p is not None and g is not None and p == g)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for cls in valid_classes:
        tp = sum(1 for p, g in pred_gt_pairs if p == cls and g == cls)
        fp = sum(1 for p, g in pred_gt_pairs if p == cls and g != cls)
        fn = sum(1 for p, g in pred_gt_pairs if p != cls and g == cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn}

    precisions = [per_class[c]["precision"] for c in valid_classes if per_class[c]["support"] > 0]
    recalls = [per_class[c]["recall"] for c in valid_classes if per_class[c]["support"] > 0]
    f1s = [per_class[c]["f1"] for c in valid_classes if per_class[c]["support"] > 0]
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def add_task_constraints_to_user_text(user_text: str, task_id: int, language: str) -> str:
    """与 dataset/cervical_dataset.py 一致：按 language 追加任务输出格式约束。"""
    lang = (language or "zh").strip().lower()
    if lang not in TASK_CONSTRAINT_TEXTS:
        lang = "zh"
    c = TASK_CONSTRAINT_TEXTS[lang]

    has_type2 = any(k in user_text for k in c["type2_keywords"])
    has_type34 = any(k in user_text for k in c["type34_keywords"])
    has_type5 = any(k in user_text for k in c["type5_keywords"])

    if task_id == 2 and not has_type2:
        user_text = user_text + "\n" + c["task2_suffix"]
    elif task_id in (3, 4) and not has_type34:
        user_text = user_text + "\n" + c["task34_suffix"]
    elif task_id == 5 and not has_type5:
        user_text = user_text + "\n" + c["task5_suffix"]
    return user_text


def _path_basename_lower(path: str) -> str:
    if not path:
        return ""
    return os.path.basename(os.path.normpath(path)).lower()


def _is_official_vlm_fo1_model_dir(path: str) -> bool:
    """
    是否为「官方/发布版」VLM-FO1 权重目录（例如 resources/VLM-FO1_Qwen2.5-VL-3B-v01）。
    注意：不能用整段绝对路径判断——仓库若在名为 VLM-FO1 的目录下，
    任意 outputs/... 路径都会误包含子串 'vlm-fo1'。
    """
    return "vlm-fo1" in _path_basename_lower(path)


def _is_lora_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    ac = os.path.join(path, "adapter_config.json")
    if not os.path.exists(ac):
        return False
    return os.path.exists(os.path.join(path, "adapter_model.safetensors")) or os.path.exists(
        os.path.join(path, "adapter_model.bin")
    )


def _load_full_weights_into_model(model, checkpoint_path: str) -> None:
    """将 checkpoint 目录下的完整权重加载到已 merge 的 model（strict=False）。"""
    has_full = os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or os.path.exists(
        os.path.join(checkpoint_path, "model.safetensors.index.json")
    )
    if not has_full:
        print("Warning: checkpoint 中未找到完整权重文件，跳过加载。")
        return
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
            for weight_file in set(index.get("weight_map", {}).values()):
                weight_path = os.path.join(checkpoint_path, weight_file)
                if os.path.exists(weight_path):
                    checkpoint_state.update(load_file(weight_path))
    if not checkpoint_state:
        print("Warning: 未能读取 checkpoint 权重。")
        return
    model_state = {
        k: v
        for k, v in checkpoint_state.items()
        if not k.startswith("trainer_state") and not k.startswith("optimizer")
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"完整权重加载完成: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")


def load_checkpoint_model(
    base_model_path,
    checkpoint_path,
    device="cuda",
    prefetch_hierarchical_fusion: bool = False,
    hierarchical_num_heads: int = 8,
    hierarchical_fusion_mode: str = "legacy",
    fpn_num_levels=None,
):
    """
    - base 为含 vlm-fo1 的官方目录：直接加载，再叠 checkpoint（LoRA 或完整）。
    - base 为 StageA 等输出目录（如 outputs/cervical_stage1_align_lora）：先加载默认基座，
      再加载 base 的 LoRA；若 checkpoint 与 base 不同，则 merge 后再加载 StageB LoRA 或完整权重。

    prefetch_hierarchical_fusion:
        在第一次 ``PeftModel.from_pretrained`` 之前挂载 ``hierarchical_fusion``，
        以便从含 ``modules_to_save`` 的 StageA adapter 中恢复门控权重。
    """
    if not checkpoint_path:
        checkpoint_path = base_model_path
    base_abs = os.path.abspath(os.path.realpath(base_model_path))
    ckpt_abs = os.path.abspath(os.path.realpath(checkpoint_path))

    def _maybe_apply_vt_fpn(m):
        try:
            from scripts.vt_fpn_utils import (
                apply_vt_fpn_ablation,
                load_fpn_ablation_meta,
                resolve_fpn_num_levels_for_eval,
                resolve_use_hierarchical_fusion_for_eval,
            )
            meta = load_fpn_ablation_meta(checkpoint_path)
            if fpn_num_levels is None and meta is None:
                return
            n = resolve_fpn_num_levels_for_eval(checkpoint_path, fpn_num_levels)
            use_hf = resolve_use_hierarchical_fusion_for_eval(checkpoint_path, None)
            if prefetch_hierarchical_fusion:
                use_hf = True
            apply_vt_fpn_ablation(
                m,
                fpn_num_levels=n,
                use_hierarchical_fusion=use_hf,
            )
        except Exception as e:
            print(f"Warning: VT FPN ablation apply skipped: {e}")

    def _prefetch_hf(m):
        if prefetch_hierarchical_fusion:
            maybe_enable_hierarchical_fusion(
                m,
                enabled=True,
                num_heads=hierarchical_num_heads,
                fusion_mode=hierarchical_fusion_mode,
            )

    def _before_adapter_load(m):
        _maybe_apply_vt_fpn(m)
        _prefetch_hf(m)

    # ---------- 官方 VLM-FO1 路径（仅看路径最后一级目录名）----------
    if base_model_path and _is_official_vlm_fo1_model_dir(base_model_path):
        print(f"\n加载基础模型: {base_model_path}")
        tokenizer, model, image_processors = load_pretrained_model(
            base_model_path, load_8bit=False, load_4bit=False, device=device
        )
        if ckpt_abs == base_abs:
            return tokenizer, model, image_processors
        if _is_lora_dir(checkpoint_path):
            if PEFT_AVAILABLE:
                _before_adapter_load(model)
                print(f"加载 LoRA: {checkpoint_path}")
                model = _peft_model_from_pretrained_compat(model, checkpoint_path)
            else:
                print("Warning: PEFT 不可用，无法加载 LoRA")
            return tokenizer, model, image_processors
        if os.path.exists(os.path.join(checkpoint_path, "config.json")):
            print(f"从 checkpoint 加载完整模型: {checkpoint_path}")
            tokenizer, model, image_processors = load_pretrained_model(
                checkpoint_path, load_8bit=False, load_4bit=False, device=device
            )
        else:
            print("Warning: 未找到 LoRA 或完整 checkpoint，仅使用基础模型")
        return tokenizer, model, image_processors

    # ---------- 非官方路径（如 StageA LoRA 目录）----------
    print(f"\n基础路径非官方 VLM-FO1 发布目录，从默认基座加载: {DEFAULT_VLM_FO1_BASE}")
    tokenizer, model, image_processors = load_pretrained_model(
        DEFAULT_VLM_FO1_BASE, load_8bit=False, load_4bit=False, device=device
    )
    _before_adapter_load(model)
    if not PEFT_AVAILABLE:
        print("Warning: PEFT 不可用，无法加载 LoRA，仅基座模型")
        return tokenizer, model, image_processors

    if _is_lora_dir(base_model_path):
        print(f"加载 StageA/base LoRA: {base_model_path}")
        model = _peft_model_from_pretrained_compat(model, base_model_path)
        if ckpt_abs != base_abs:
            if _is_lora_dir(checkpoint_path):
                print(f"合并 StageA LoRA 后加载 StageB LoRA: {checkpoint_path}")
                model = model.merge_and_unload()
                model = _peft_model_from_pretrained_compat(model, checkpoint_path)
            elif os.path.exists(os.path.join(checkpoint_path, "config.json")):
                print(f"合并 StageA LoRA 后加载完整 checkpoint: {checkpoint_path}")
                model = model.merge_and_unload()
                _load_full_weights_into_model(model, checkpoint_path)
            else:
                print("Warning: checkpoint 与 base 不同但未识别为 LoRA/完整权重，保留 StageA LoRA 模型")
        return tokenizer, model, image_processors

    # base 无 LoRA：仅从 checkpoint 加载
    if _is_lora_dir(checkpoint_path):
        print(f"加载 LoRA: {checkpoint_path}")
        _before_adapter_load(model)
        model = _peft_model_from_pretrained_compat(model, checkpoint_path)
    elif os.path.exists(os.path.join(checkpoint_path, "config.json")) and _is_official_vlm_fo1_model_dir(checkpoint_path):
        print(f"从 checkpoint 加载完整模型: {checkpoint_path}")
        tokenizer, model, image_processors = load_pretrained_model(
            checkpoint_path, load_8bit=False, load_4bit=False, device=device
        )
    else:
        print("Warning: base 无 LoRA 且 checkpoint 无法加载，仅基座模型")
    return tokenizer, model, image_processors


def _unwrap_hierarchical_fusion_module(hf):
    """解包 PEFT ``ModulesToSaveWrapper``：``forward`` 与 ``last_gates`` 等缓存在内层真实模块上。"""
    if hf is None:
        return None
    mts = getattr(hf, "modules_to_save", None)
    if mts is not None and hasattr(hf, "active_adapter"):
        try:
            key = hf.active_adapter
            if key in mts:
                return mts[key]
        except Exception:
            pass
    orig = getattr(hf, "original_module", None)
    if orig is not None:
        return orig
    return hf


def maybe_enable_hierarchical_fusion(model, enabled: bool = False, num_heads: int = 8, fusion_mode: str = "legacy"):
    if not enabled:
        return
    core = model.get_model() if hasattr(model, "get_model") else model
    extractor = getattr(core, "object_vp_extractor", None)
    if extractor is None:
        print("Warning: object_vp_extractor not found, skip hierarchical fusion enable.")
        return
    if not getattr(extractor, "use_simpleFPN_for_vt", False):
        print("Warning: hierarchical fusion currently expects mm_use_simpleFPN_for_vt=true. Skip.")
        return
    num_levels = int(getattr(extractor, "fpn_num_levels", 4))
    extractor.use_hierarchical_fusion = True
    extractor.hierarchical_num_heads = int(num_heads)
    fusion_mode = str(fusion_mode).strip().lower()
    if fusion_mode not in ("legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"):
        fusion_mode = "legacy"
    extractor.hierarchical_fusion_mode = fusion_mode

    existing = getattr(extractor, "hierarchical_fusion", None)
    inner = _unwrap_hierarchical_fusion_module(existing) if existing is not None else None
    need_rebuild = (
        inner is None
        or getattr(inner, "num_levels", 4) != num_levels
        or getattr(inner, "fusion_mode", fusion_mode) != fusion_mode
    )
    if need_rebuild:
        extractor.hierarchical_fusion = HierarchicalFeatureFusion(
            feature_dim=512,
            out_dim=2048,
            num_heads=int(num_heads),
            fusion_mode=fusion_mode,
            num_levels=num_levels,
        )
    elif inner is not None and hasattr(inner, "fusion_mode"):
        inner.fusion_mode = fusion_mode
    ref_param = next(model.parameters())
    extractor.hierarchical_fusion.to(device=ref_param.device, dtype=ref_param.dtype)
    print(
        f"[Eval] Hierarchical fusion enabled "
        f"(heads={num_heads}, mode={fusion_mode}, num_levels={num_levels})."
    )


def _build_metrics_snapshot(task_metrics: dict, task_counts: dict) -> dict:
    metrics = {"task_distribution": dict(task_counts)}

    for task_type in ["type0", "type1"]:
        m = task_metrics[task_type]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"{task_type}_precision"] = precision
        metrics[f"{task_type}_recall"] = recall
        metrics[f"{task_type}_f1"] = f1
        metrics[f"{task_type}_mean_iou"] = float(np.mean(m["all_ious"])) if m["all_ious"] else 0.0
        metrics[f"{task_type}_map"] = float(np.mean(m["maps"])) if m["maps"] else 0.0
        metrics[f"{task_type}_tp"] = tp
        metrics[f"{task_type}_fp"] = fp
        metrics[f"{task_type}_fn"] = fn

    if task_metrics["type2"]["bleus"]:
        metrics["type2_mean_bleu"] = float(np.mean(task_metrics["type2"]["bleus"]))
        metrics["type2_mean_rouge_l"] = float(np.mean(task_metrics["type2"]["rouge_ls"]))
        metrics["type2_mean_rouge_1"] = float(np.mean(task_metrics["type2"]["rouge_1s"]))
        metrics["type2_mean_rouge_2"] = float(np.mean(task_metrics["type2"]["rouge_2s"]))

    if task_metrics["type4"]:
        type4_metrics = compute_per_class_metrics(task_metrics["type4"], TYPE4_SOURCE_CLASSES)
        metrics["type4_accuracy"] = type4_metrics["accuracy"]
        metrics["type4_correct"] = type4_metrics["correct"]
        metrics["type4_total"] = type4_metrics["total"]
        metrics["type4_macro_precision"] = type4_metrics["macro_precision"]
        metrics["type4_macro_recall"] = type4_metrics["macro_recall"]
        metrics["type4_macro_f1"] = type4_metrics["macro_f1"]
        metrics["type4_per_class"] = type4_metrics["per_class"]

    if task_metrics["type3"]:
        type3_metrics = compute_per_class_metrics(task_metrics["type3"], TYPE3_CELL_TYPE_CLASSES)
        metrics["type3_accuracy"] = type3_metrics["accuracy"]
        metrics["type3_correct"] = type3_metrics["correct"]
        metrics["type3_total"] = type3_metrics["total"]
        metrics["type3_macro_precision"] = type3_metrics["macro_precision"]
        metrics["type3_macro_recall"] = type3_metrics["macro_recall"]
        metrics["type3_macro_f1"] = type3_metrics["macro_f1"]
        metrics["type3_per_class"] = type3_metrics["per_class"]

    if task_metrics["type5"]:
        type5_metrics = compute_per_class_metrics(task_metrics["type5"], TYPE5_CELL_TYPE_CLASSES)
        metrics["type5_accuracy"] = type5_metrics["accuracy"]
        metrics["type5_correct"] = type5_metrics["correct"]
        metrics["type5_total"] = type5_metrics["total"]
        metrics["type5_macro_precision"] = type5_metrics["macro_precision"]
        metrics["type5_macro_recall"] = type5_metrics["macro_recall"]
        metrics["type5_macro_f1"] = type5_metrics["macro_f1"]
        metrics["type5_per_class"] = type5_metrics["per_class"]
    return metrics


def _summarize_sample_gate_list(sample_gate_list: list) -> dict:
    """
    sample_gate_list: List[[g0..gK-1]]，默认直接使用 ROI-level 门控向量（尽量少压缩差异）。
    """
    if not sample_gate_list:
        return {
            "count": 0,
            "mean": [],
            "std": [],
        }
    arr = np.array(sample_gate_list, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).round(6).tolist(),
        "std": arr.std(axis=0).round(6).tolist(),
    }


def _format_gate_mean_str(mean: list) -> str:
    if not mean:
        return ""
    parts = [f"L{i}={float(mean[i]):.4f}" for i in range(len(mean))]
    return " ".join(parts)


def _build_gate_stats_snapshot(hier_gate_samples: dict) -> dict:
    summary = {
        "all": _summarize_sample_gate_list(hier_gate_samples.get("all", [])),
        "by_task_type": {},
        "by_cell_type": {},
        "note": "gate statistics are aggregated on ROI-level gate vectors directly (less compression than sample-first mean).",
    }
    for k, v in sorted(hier_gate_samples.get("by_task_type", {}).items()):
        summary["by_task_type"][k] = _summarize_sample_gate_list(v)
    for k, v in sorted(hier_gate_samples.get("by_cell_type", {}).items()):
        summary["by_cell_type"][k] = _summarize_sample_gate_list(v)
    return summary


def _save_progress(output_dir: str, results: list, task_metrics: dict, task_counts: dict, hier_gate_samples: dict = None) -> dict:
    metrics = _build_metrics_snapshot(task_metrics, task_counts)
    if hier_gate_samples is not None:
        metrics["hierarchical_gate_stats"] = _build_gate_stats_snapshot(hier_gate_samples)
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if hier_gate_samples is not None:
        gate_stats_file = os.path.join(output_dir, "gate_stats.json")
        with open(gate_stats_file, "w", encoding="utf-8") as f:
            json.dump(metrics["hierarchical_gate_stats"], f, ensure_ascii=False, indent=2)
    return metrics


def evaluate_model(
    model,
    tokenizer,
    image_processors,
    dataset,
    device="cuda",
    output_dir="eval_result/stage2_01245",
    model_name=None,
    path_feature_root=None,
    visualize=False,
    language: str = "zh",
    save_hierarchical_attn_dir: str = None,
    save_hierarchical_attn_every: int = 0,
    save_hierarchical_attn_per_task_max: int = 10,
    save_hierarchical_attn_seed: int = 42,
    task_ids_to_eval=None,
    origin_by_id=None,
    origin_by_basename=None,
    save_every_n_samples: int = 1000,
    run_title: str = None,
    max_samples: int = 0,
    compute_task2_sbert: bool = True,
    task2_en_gt_ref: Optional[str] = None,
    task2_sbert_model: Optional[str] = None,
    task2_sbert_device: Optional[str] = None,
):
    if run_title is None:
        run_title = "阶段2 多任务评估汇总（任务 0,1,2,3,4,5）"
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    origin_by_id = origin_by_id or {}
    origin_by_basename = origin_by_basename or {}
    vis_dir = os.path.join(output_dir, "visualizations") if visualize else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    if save_hierarchical_attn_dir:
        os.makedirs(save_hierarchical_attn_dir, exist_ok=True)

    # 随机分层采样：每个 task 最多导出 N 张注意力图
    attn_selected_indices_by_task = {}
    valid_task_types = ("type0", "type1", "type2", "type3", "type4", "type5")
    if task_ids_to_eval is None:
        task_ids_to_eval = {0, 1, 2, 3, 4, 5}
    else:
        task_ids_to_eval = {int(x) for x in task_ids_to_eval if int(x) in (0, 1, 2, 3, 4, 5)}
    allowed_task_types = {f"type{tid}" for tid in sorted(task_ids_to_eval)}
    print(f"[Eval] 仅评估任务: {sorted(task_ids_to_eval)}")
    n_total = len(dataset)
    n_iter = n_total if int(max_samples) <= 0 else min(n_total, int(max_samples))
    if n_iter < n_total:
        print(f"[Eval] 样本上限: 仅跑前 {n_iter}/{n_total} 条（dataset 顺序，与 --task_ids 筛选叠加）")
    if save_hierarchical_attn_dir and save_hierarchical_attn_per_task_max > 0:
        data_source = getattr(dataset, "data", None)
        if isinstance(data_source, list):
            task_to_indices = defaultdict(list)
            for i, item in enumerate(data_source):
                t = get_task_type_from_item(item if isinstance(item, dict) else {})
                if t in valid_task_types and t in allowed_task_types:
                    task_to_indices[t].append(i)
            rng = random.Random(int(save_hierarchical_attn_seed))
            for t, idxs in task_to_indices.items():
                k = min(int(save_hierarchical_attn_per_task_max), len(idxs))
                attn_selected_indices_by_task[t] = set(rng.sample(idxs, k)) if k > 0 else set()
            print(
                "[Eval] Hierarchical attention export plan (random per task): "
                + ", ".join(f"{t}={len(attn_selected_indices_by_task.get(t, set()))}" for t in valid_task_types)
            )
        else:
            print("Warning: dataset.data not found, fallback to --save_hierarchical_attn_every behavior.")

    task_metrics = {
        "type0": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type1": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type2": {"bleus": [], "rouge_ls": [], "rouge_1s": [], "rouge_2s": []},
        "type3": [],  # list of (pred_class, gt_class)
        "type4": [],  # list of (pred_class, gt_class)
        "type5": [],  # list of (pred_class, gt_class)
    }
    task_counts = defaultdict(int)
    task_id_raw_stats = defaultdict(int)
    task_id_effective_stats = defaultdict(int)
    task_id_effective_by_task_type = defaultdict(lambda: defaultdict(int))
    task_id_diag_seen = 0
    task_id_diag_preview_max = 10
    task_id_diag_roll_every = 500
    results = []
    hier_gate_samples = {
        "all": [],
        "by_task_type": defaultdict(list),
        "by_cell_type": defaultdict(list),
    }

    for idx in tqdm(range(n_iter), desc="Evaluating", ncols=100):
        try:
            sample = dataset[idx]
            if hasattr(dataset, "data"):
                data_item = dataset.data[idx]
            else:
                data_item = dataset[idx] if isinstance(dataset[idx], dict) else None
            if data_item is None:
                continue

            image_path = data_item["image"]
            image = Image.open(image_path).convert("RGB")
            user_text = None
            assistant_text = None
            for conv in data_item.get("conversations", []):
                if conv.get("from") == "user":
                    user_text = conv.get("value", "").replace("<image>", "").strip()
                    if user_text.startswith("\n"):
                        user_text = user_text[1:]
                elif conv.get("from") == "assistant":
                    assistant_text = conv.get("value", "")

            if user_text is None or assistant_text is None:
                continue

            task_type = get_task_type_from_item(data_item)
            if task_type == "unknown" or task_type not in ("type0", "type1", "type2", "type3", "type4", "type5"):
                continue
            if task_type not in allowed_task_types:
                continue
            task_counts[task_type] += 1

            task_id = None
            raw_task_id = data_item.get("task_id", None)
            try:
                if raw_task_id is not None:
                    task_id = int(raw_task_id)
            except Exception:
                pass
            effective_task_id = task_id if task_id is not None else -1
            raw_key = "__missing__" if raw_task_id is None else str(raw_task_id)
            task_id_raw_stats[raw_key] += 1
            task_id_effective_stats[str(effective_task_id)] += 1
            task_id_effective_by_task_type[task_type][str(effective_task_id)] += 1
            task_id_diag_seen += 1
            if task_id_diag_seen <= task_id_diag_preview_max:
                tqdm.write(
                    f"[Eval][TaskIDDiag][sample={idx}] "
                    f"task_type={task_type}, raw_task_id={raw_task_id}, effective_task_id={effective_task_id}"
                )
            elif task_id_diag_seen % task_id_diag_roll_every == 0:
                eff_dist = dict(sorted(task_id_effective_stats.items(), key=lambda kv: int(kv[0])))
                tqdm.write(f"[Eval][TaskIDDiag] seen={task_id_diag_seen}, effective_dist={eff_dist}")
            user_text = add_task_constraints_to_user_text(user_text, effective_task_id, language=language)

            core = model.get_model() if hasattr(model, "get_model") else model
            extractor = getattr(core, "object_vp_extractor", None)
            hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
            hf_inner = _unwrap_hierarchical_fusion_module(hf)
            if hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
                # Clear cache before each sample; only collect gates generated by current forward.
                hf_inner.last_gates = None
                hf_inner.last_gates_raw = None

            gt_regions = data_item.get("regions", [])
            bbox_list_for_model = [r["bbox"] for r in gt_regions]
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
                path_feature_root=path_feature_root,
                task_ids=[effective_task_id],
            )

            output_ids = model.generate(**generation_kwargs)
            output_text = tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
                skip_special_tokens=False,
            ).strip()
            # Hierarchical gate statistics (ROI-level aggregation to preserve variation).
            try:
                if hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
                    sample_gates = getattr(hf_inner, "last_gates", None)
                    if sample_gates is not None and sample_gates.numel() > 0:
                        # Keep ROI-level vectors to reduce compression of gate differences.
                        gate_rows = sample_gates.detach().float().cpu().tolist()
                        hier_gate_samples["all"].extend(gate_rows)
                        hier_gate_samples["by_task_type"][task_type].extend(gate_rows)

                        gt_cell_type = None
                        if task_type == "type3":
                            gt_cell_type = _extract_class_from_text(assistant_text, TYPE3_CELL_TYPE_CLASSES)
                        elif task_type == "type5":
                            gt_section = _extract_diagnosis_section(assistant_text)
                            gt_cell_type = _extract_class_from_text_last(gt_section or assistant_text, TYPE5_CELL_TYPE_CLASSES)
                        if gt_cell_type:
                            hier_gate_samples["by_cell_type"][gt_cell_type].extend(gate_rows)
            except Exception as e:
                print(f"Warning: failed to collect hierarchical gate stats for sample {idx}: {e}")

            should_save_attn = False
            if save_hierarchical_attn_dir:
                if save_hierarchical_attn_per_task_max > 0 and attn_selected_indices_by_task:
                    should_save_attn = idx in attn_selected_indices_by_task.get(task_type, set())
                elif save_hierarchical_attn_every > 0:
                    should_save_attn = (idx % save_hierarchical_attn_every == 0)
            if should_save_attn:
                try:
                    if hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
                        attn_path = os.path.join(save_hierarchical_attn_dir, f"sample_{idx}_{task_type}.png")
                        visualize_hierarchical_attention(hf_inner, save_path=attn_path)
                except Exception as e:
                    print(f"Warning: failed to save hierarchical attention for sample {idx}: {e}")

            result_item = {
                "id": data_item.get("id", f"sample_{idx}"),
                "image": image_path,
                "task_type": task_type,
                "task_id_raw": raw_task_id,
                "task_id_effective": effective_task_id,
                "user_text": user_text,
                "prediction": output_text,
                "ground_truth": assistant_text,
            }

            if task_type in ["type0", "type1"]:
                pred_dict = parse_grounding_output(output_text)
                pred_indices = sorted({i for _, idxs in pred_dict.items() for i in idxs if isinstance(i, int)})
                pred_bboxes = [bbox_list_for_model[i] for i in pred_indices if 0 <= i < len(bbox_list_for_model)]
                gt_bbox = _pick_gt_bbox_for_eval_item(data_item, origin_by_id, origin_by_basename)
                if gt_bbox is None:
                    print(
                        f"Warning: sample {data_item.get('id', idx)} "
                        f"(task_type={task_type}) 在 origin_data 中未找到 GT bbox，跳过该样本"
                    )
                    continue

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

                task_metrics[task_type]["tp"] += tp
                task_metrics[task_type]["fp"] += fp
                task_metrics[task_type]["fn"] += fn
                task_metrics[task_type]["all_ious"].append(float(best_iou))
                task_metrics[task_type]["maps"].append(float(ap))
                result_item.update({
                    "pred_indices": pred_indices,
                    "pred_bboxes": pred_bboxes,
                    "gt_bbox": gt_bbox,
                    "best_pred_bbox": best_bbox,
                    "best_iou": float(best_iou),
                    "map": float(ap),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            elif task_type == "type2":
                metrics = evaluate_caption_task(output_text, assistant_text)
                task_metrics[task_type]["bleus"].append(metrics["bleu"])
                task_metrics[task_type]["rouge_ls"].append(metrics["rouge_l"])
                task_metrics[task_type]["rouge_1s"].append(metrics["rouge_1"])
                task_metrics[task_type]["rouge_2s"].append(metrics["rouge_2"])
                result_item.update(metrics)

            elif task_type == "type3":
                pred_class = _extract_class_from_text(output_text, TYPE3_CELL_TYPE_CLASSES)
                gt_class = _extract_class_from_text(assistant_text, TYPE3_CELL_TYPE_CLASSES)
                task_metrics[task_type].append((pred_class, gt_class))
                result_item["pred_class"] = pred_class
                result_item["gt_class"] = gt_class
                result_item["correct"] = pred_class == gt_class and pred_class is not None and gt_class is not None

            elif task_type == "type4":
                pred_class = _extract_class_from_text(output_text, TYPE4_SOURCE_CLASSES)
                gt_class = _extract_class_from_text(assistant_text, TYPE4_SOURCE_CLASSES)
                task_metrics[task_type].append((pred_class, gt_class))
                result_item["pred_class"] = pred_class
                result_item["gt_class"] = gt_class
                result_item["correct"] = pred_class == gt_class and pred_class is not None and gt_class is not None

            elif task_type == "type5":
                pred_section = _extract_diagnosis_section(output_text)
                gt_section = _extract_diagnosis_section(assistant_text)
                # task5 文本中包含大量“排除X...”，优先取“诊断结论段内最后一次出现”的类别
                pred_class = _extract_class_from_text_last(pred_section or output_text, TYPE5_CELL_TYPE_CLASSES)
                gt_class = _extract_class_from_text_last(gt_section or assistant_text, TYPE5_CELL_TYPE_CLASSES)
                task_metrics[task_type].append((pred_class, gt_class))
                result_item["pred_class"] = pred_class
                result_item["gt_class"] = gt_class
                result_item["correct"] = pred_class == gt_class and pred_class is not None and gt_class is not None

            results.append(result_item)
            if save_every_n_samples <= 1 or (len(results) % int(save_every_n_samples) == 0):
                _save_progress(output_dir, results, task_metrics, task_counts, hier_gate_samples=hier_gate_samples)

            if visualize and task_type in ["type0", "type1"] and idx % 10 == 0:
                pred_dict = parse_grounding_output(output_text)
                region_id_to_bbox = {}
                for r in gt_regions:
                    rid = r.get("region_id", "")
                    mat = re.match(r"region(\d+)", rid)
                    if mat:
                        region_id_to_bbox[int(mat.group(1))] = r["bbox"]
                pred_indices = set()
                for _, indices in pred_dict.items():
                    pred_indices.update(indices)
                pred_bboxes = [region_id_to_bbox[i] for i in sorted(pred_indices) if i in region_id_to_bbox]
                fo1_bboxes = {"positive cervical cell": pred_bboxes} if pred_bboxes else {}
                vis_path = os.path.join(vis_dir, f"sample_{idx}_{task_type}.jpg")
                draw_bboxes_and_save(image=image, fo1_bboxes=fo1_bboxes, output_path=vis_path, color="red")

        except Exception as e:
            print(f"Error sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总指标并最终落盘
    metrics = _save_progress(output_dir, results, task_metrics, task_counts, hier_gate_samples=hier_gate_samples)

    # Type2 SBERT（英文对齐管线）：在 VLM 推理结束后补算并写回 metrics/results
    sbert_dev = task2_sbert_device or device
    metrics = _compute_and_attach_task2_sbert(
        results,
        metrics,
        language=language,
        enabled=bool(compute_task2_sbert),
        en_gt_ref=task2_en_gt_ref,
        sbert_model=task2_sbert_model or DEFAULT_TASK2_SBERT_MODEL,
        sbert_device=sbert_dev,
        output_dir=output_dir,
    )

    # 打印
    print("\n" + "=" * 60)
    print(run_title)
    print("=" * 60)
    print("\n任务分布:")
    for t, c in task_counts.items():
        print(f"  {t}: {c} 个样本")
    print("\nTask ID 统计（eval 传给模型前）:")
    print(f"  raw task_id 分布: {dict(sorted(task_id_raw_stats.items(), key=lambda kv: kv[0]))}")
    print(f"  effective task_id 分布: {dict(sorted(task_id_effective_stats.items(), key=lambda kv: int(kv[0])))}")
    print("  effective task_id 按 task_type 分布:")
    for t in sorted(task_id_effective_by_task_type.keys()):
        dist = dict(sorted(task_id_effective_by_task_type[t].items(), key=lambda kv: int(kv[0])))
        print(f"    {t}: {dist}")

    # 轻量诊断：effective_task_id=-1 或与 task_type 编号不一致时，task_adaptive 门控可能与分桶标签错位
    n_m1 = int(task_id_effective_stats.get("-1", 0))
    if n_m1 > 0:
        print(
            f"\n[Eval][TaskID 诊断] WARNING: 共 {n_m1} 条样本 effective_task_id=-1（task_id 缺失或解析失败）。"
            "模型内区域门控 task_adaptive 将按 type1 条件计算，与按 JSON 推得的 task_type 分桶可能不一致。"
        )
    print("\n[Eval][TaskID 诊断] 各 task_type 下 effective_task_id 是否与类型编号一致（仅已评任务）:")
    for t in sorted(allowed_task_types):
        exp = _expected_task_id_for_task_type(t)
        dist = dict(task_id_effective_by_task_type.get(t, {}))
        if not dist:
            print(f"    {t}: （无样本）")
            continue
        dist_sorted = dict(sorted(dist.items(), key=lambda kv: int(kv[0])))
        m1 = int(dist.get("-1", 0))
        mism = []
        if exp is not None:
            for k, v in dist.items():
                if k == "-1":
                    continue
                try:
                    if int(k) != exp:
                        mism.append(f"id={k}: {v}")
                except ValueError:
                    mism.append(f"id={k!r}: {v}")
        if m1 or mism:
            parts = []
            if m1:
                parts.append(f"effective=-1 有 {m1} 条")
            if mism:
                parts.append("非预期 id: " + "; ".join(mism))
            print(f"    {t}: 期望仅 id={exp} | " + " | ".join(parts) + f" | 完整分布 {dist_sorted}")
        else:
            n_ok = sum(v for k, v in dist.items() if k != "-1" and exp is not None and int(k) == exp)
            print(f"    {t}: 与 task_id={exp} 一致（{n_ok} 条）分布 {dist_sorted}")

    for t in ["type0", "type1"]:
        if task_counts.get(t, 0) == 0:
            continue
        print(f"\n{t} (定位) 指标:")
        print(f"  Precision: {metrics.get(f'{t}_precision', 0):.4f}")
        print(f"  Recall:    {metrics.get(f'{t}_recall', 0):.4f}")
        print(f"  F1:        {metrics.get(f'{t}_f1', 0):.4f}")
        print(f"  Mean IoU:  {metrics.get(f'{t}_mean_iou', 0):.4f}")
        print(f"  mAP:       {metrics.get(f'{t}_map', 0):.4f}")

    if task_metrics["type2"]["bleus"]:
        print("\nType2 (区域描述) 指标:")
        print(f"  Mean BLEU:    {metrics.get('type2_mean_bleu', 0):.4f}")
        print(f"  Mean ROUGE-L: {metrics.get('type2_mean_rouge_l', 0):.4f}")
        print(f"  Mean ROUGE-1: {metrics.get('type2_mean_rouge_1', 0):.4f}")
        print(f"  Mean ROUGE-2: {metrics.get('type2_mean_rouge_2', 0):.4f}")
        if "type2_mean_sbert" in metrics:
            print(
                f"  Mean SBERT:   {metrics.get('type2_mean_sbert', 0):.4f} "
                f"(rule_gt={metrics.get('type2_mean_sbert_rule_gt', 0):.4f}, "
                f"n={metrics.get('type2_sbert_n', 0)}, "
                f"model={metrics.get('type2_sbert_model', '')})"
            )

    if task_metrics["type4"]:
        print("\nType4 (来源分类) 指标:")
        print(f"  Accuracy:       {metrics.get('type4_accuracy', 0):.4f} ({metrics.get('type4_correct', 0)}/{metrics.get('type4_total', 0)})")
        print(f"  Macro P/R/F1:   P={metrics.get('type4_macro_precision', 0):.4f}  R={metrics.get('type4_macro_recall', 0):.4f}  F1={metrics.get('type4_macro_f1', 0):.4f}")
        print("  每类别 (Precision, Recall, F1, Support):")
        for cls in TYPE4_SOURCE_CLASSES:
            pc = metrics.get("type4_per_class", {}).get(cls, {})
            p, r, f, s = pc.get("precision", 0), pc.get("recall", 0), pc.get("f1", 0), pc.get("support", 0)
            print(f"    {cls}: P={p:.4f} R={r:.4f} F1={f:.4f} support={s}")

    if task_metrics["type3"]:
        print("\nType3 (细胞类型分类) 指标:")
        print(f"  Accuracy:       {metrics.get('type3_accuracy', 0):.4f} ({metrics.get('type3_correct', 0)}/{metrics.get('type3_total', 0)})")
        print(f"  Macro P/R/F1:   P={metrics.get('type3_macro_precision', 0):.4f}  R={metrics.get('type3_macro_recall', 0):.4f}  F1={metrics.get('type3_macro_f1', 0):.4f}")
        print("  每类别 (Precision, Recall, F1, Support):")
        for cls in TYPE3_CELL_TYPE_CLASSES:
            pc = metrics.get("type3_per_class", {}).get(cls, {})
            p, r, f, s = pc.get("precision", 0), pc.get("recall", 0), pc.get("f1", 0), pc.get("support", 0)
            print(f"    {cls}: P={p:.4f} R={r:.4f} F1={f:.4f} support={s}")

    if task_metrics["type5"]:
        print("\nType5 (区域细胞类型+诊断依据，从【诊断结果】提取类别) 指标:")
        print(f"  Accuracy:       {metrics.get('type5_accuracy', 0):.4f} ({metrics.get('type5_correct', 0)}/{metrics.get('type5_total', 0)})")
        print(f"  Macro P/R/F1:   P={metrics.get('type5_macro_precision', 0):.4f}  R={metrics.get('type5_macro_recall', 0):.4f}  F1={metrics.get('type5_macro_f1', 0):.4f}")
        print("  每类别 (Precision, Recall, F1, Support):")
        for cls in TYPE5_CELL_TYPE_CLASSES:
            pc = metrics.get("type5_per_class", {}).get(cls, {})
            p, r, f, s = pc.get("precision", 0), pc.get("recall", 0), pc.get("f1", 0), pc.get("support", 0)
            print(f"    {cls}: P={p:.4f} R={r:.4f} F1={f:.4f} support={s}")

    gate_stats = metrics.get("hierarchical_gate_stats", {})
    if gate_stats:
        overall = gate_stats.get("all", {})
        mean = overall.get("mean", []) or []
        print("\nHierarchical Gate (ROI-aggregated) 统计:")
        gm = _format_gate_mean_str(mean)
        _line = f"  All samples: count={overall.get('count', 0)}"
        if gm:
            _line += f" {gm}"
        print(_line)
        if gate_stats.get("by_cell_type"):
            print("  By cell type:")
            for ct, s in gate_stats["by_cell_type"].items():
                m = s.get("mean", []) or []
                suf = _format_gate_mean_str(m)
                _line2 = f"    {ct}: count={s.get('count', 0)}"
                if suf:
                    _line2 += f" {suf}"
                print(_line2)

    print("\n" + "=" * 60)
    return metrics


def main():
    def _parse_task_ids_csv(s: str):
        ids = []
        for p in str(s).split(","):
            p = p.strip()
            if not p:
                continue
            try:
                v = int(p)
            except Exception:
                continue
            if v in (0, 1, 2, 3, 4, 5):
                ids.append(v)
        if not ids:
            return [0, 1, 2, 3, 4, 5]
        return sorted(set(ids))

    parser = argparse.ArgumentParser(description="阶段2 多任务评估（任务 0,1,2,4,5）")
    parser.add_argument("--base_model", type=str, default="outputs/cervical_PAfull_A_fusion", help="基础模型路径")
    parser.add_argument("--checkpoint", type=str, default="outputs/cervical_PAfull_B_fusion", help="Stage2 checkpoint")
    parser.add_argument("--test_json", type=str, default="data/vlm_fo1_data/dataset/final/stage2_val.json", help="测试集 JSON")
    parser.add_argument("--output_dir", type=str, default="eval_result/stage2_01245", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_feature_root", type=str, default=None, help="病理特征目录，None 表示不使用")
    parser.add_argument("--visualize", action="store_true", help="是否保存定位任务可视化")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"], help="约束文案语言，与训练时一致")
    parser.add_argument("--use_hierarchical_fusion", action="store_true", help="启用层级注意力融合（HFRE增强）")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8, help="层级注意力头数")
    parser.add_argument("--hierarchical_fusion_mode", type=str, default="legacy", choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"], help="层级融合模式")
    parser.add_argument("--save_hierarchical_attn_dir", type=str, default=None, help="保存层级注意力热力图目录")
    parser.add_argument("--save_hierarchical_attn_every", type=int, default=0, help="每 N 个样本保存一次层级注意力图，0 关闭")
    parser.add_argument("--save_hierarchical_attn_per_task_max", type=int, default=10, help="每个 task_id 随机导出注意力图上限")
    parser.add_argument("--save_hierarchical_attn_seed", type=int, default=42, help="注意力图随机采样种子")
    parser.add_argument("--task_ids", type=str, default="0,1,2,3,4,5", help="要评估的 task_id 列表，如 2,3,4,5")
    parser.add_argument("--origin_json", type=str, default=None, help="原始数据集 JSON（用于 task0/1 的 GT bbox）")
    parser.add_argument("--save_every_n_samples", type=int, default=1000, help="中间结果每 N 个样本保存一次")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多评估的样本条数（按 dataset 下标顺序）；0 表示不限制，跑全量",
    )
    parser.add_argument(
        "--compute_task2_sbert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估结束后对 type2 计算英文管线 SBERT（默认开启；--no-compute_task2_sbert 关闭）",
    )
    parser.add_argument(
        "--task2_en_gt_ref",
        type=str,
        default=DEFAULT_TASK2_EN_GT_REF,
        help="对齐英文 GT（按 sample id），默认 eval_results/en_gt_ref_stage2_cot_val.json",
    )
    parser.add_argument(
        "--task2_sbert_model",
        type=str,
        default=DEFAULT_TASK2_SBERT_MODEL,
        help="SBERT 模型名或本地路径",
    )
    parser.add_argument(
        "--task2_sbert_device",
        type=str,
        default=None,
        help="SBERT 设备（默认与 --device 相同；VLM 占满显存时可设为 cpu）",
    )
    args = parser.parse_args()
    selected_task_ids = _parse_task_ids_csv(args.task_ids)

    # 按语言切换候选类别集合（保证最终指标表使用“英文原样”）
    global TYPE3_CELL_TYPE_CLASSES, TYPE4_SOURCE_CLASSES, TYPE5_CELL_TYPE_CLASSES
    if args.language == "en":
        TYPE3_CELL_TYPE_CLASSES = TYPE3_CELL_TYPE_CLASSES_BY_LANG["en"]
        TYPE4_SOURCE_CLASSES = TYPE4_SOURCE_CLASSES_BY_LANG["en"]
        TYPE5_CELL_TYPE_CLASSES = TYPE5_CELL_TYPE_CLASSES_BY_LANG["en"]
    else:
        TYPE3_CELL_TYPE_CLASSES = TYPE3_CELL_TYPE_CLASSES_BY_LANG["zh"]
        TYPE4_SOURCE_CLASSES = TYPE4_SOURCE_CLASSES_BY_LANG["zh"]
        TYPE5_CELL_TYPE_CLASSES = TYPE5_CELL_TYPE_CLASSES_BY_LANG["zh"]

    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        prefetch_hierarchical_fusion=args.use_hierarchical_fusion,
        hierarchical_num_heads=args.hierarchical_num_heads,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
    )
    maybe_enable_hierarchical_fusion(
        model,
        enabled=args.use_hierarchical_fusion,
        num_heads=args.hierarchical_num_heads,
        fusion_mode=args.hierarchical_fusion_mode,
    )
    model_name_for_template = (
        args.base_model if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model) else DEFAULT_VLM_FO1_BASE
    )

    print(f"\n加载测试集: {args.test_json}")
    if not os.path.exists(args.test_json):
        print(f"Error: 测试集不存在: {args.test_json}")
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
    print(f"测试集大小: {len(test_dataset)}")
    origin_by_id, origin_by_basename = {}, {}
    if args.origin_json:
        if os.path.exists(args.origin_json):
            origin_by_id, origin_by_basename = _build_origin_gt_map(args.origin_json)
            print(f"已加载 origin GT bbox: by_id={len(origin_by_id)}, by_basename={len(origin_by_basename)}")
        else:
            print(f"Warning: origin_json 不存在，task0/1 将回退为使用 test_json 的 regions 作为 GT: {args.origin_json}")

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
        run_title="阶段2 多任务评估汇总（任务 0,1,2,3,4,5）",
        max_samples=args.max_samples,
        compute_task2_sbert=args.compute_task2_sbert,
        task2_en_gt_ref=args.task2_en_gt_ref,
        task2_sbert_model=args.task2_sbert_model,
        task2_sbert_device=args.task2_sbert_device,
    )
    print(f"\n评估完成，结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
