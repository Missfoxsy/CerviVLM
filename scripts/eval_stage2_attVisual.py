#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 阶段2 多任务评估脚本（任务 0,1,2,4,5）

支持五种任务类型（与 cervical_dataset 约束一致）：
- 类型0 (task_id=0)：基础定位 - IoU、Precision、Recall、F1、mAP
- 类型1 (task_id=1)：带描述的特征定位 - IoU、Precision、Recall、F1、mAP
- 类型2 (task_id=2)：区域描述 - BLEU、ROUGE
- 类型4 (task_id=4)：来源分类 - 准确率 + 每个类别的 Precision/Recall/F1/Support
- 类型5 (task_id=5)：区域细胞类型+诊断依据（CoT） - 从【诊断结果】中提取类别，按分类任务计算准确率 + 每类别指标

评估时自动为 user 文本追加与 dataset/cervical_dataset.py 相同的任务约束。
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import random
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
from collections import defaultdict
import re

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _ensure_task_attn_dirs(base_dir: str, task_type: str):
    """按任务类型创建注意力导出目录。"""
    task_root = os.path.join(base_dir, task_type)
    static_dir = os.path.join(task_root, "static")
    dynamic_dir = os.path.join(task_root, "dynamic")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(dynamic_dir, exist_ok=True)
    return task_root, static_dir, dynamic_dir


def _to_numpy_attention(attn_obj):
    """将不同形态 attention 转为 [heads, q, k] numpy。"""
    if attn_obj is None:
        return None
    if isinstance(attn_obj, torch.Tensor):
        arr = attn_obj.detach().float().cpu().numpy()
    elif isinstance(attn_obj, np.ndarray):
        arr = attn_obj.astype(np.float32)
    else:
        return None

    # 兼容 [bs, heads, q, k] / [heads, q, k] / [q, k]
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return None
    return arr


def _extract_step_attn_tensors(generation_output):
    """
    从 generate(return_dict_in_generate=True, output_attentions=True) 中提取 step attentions。
    返回 list[Tensor/ndarray]，每个元素对应一个 step 的“最后一层 attention”。
    """
    if generation_output is None:
        return []

    attn_steps = getattr(generation_output, "attentions", None)
    if attn_steps is None:
        attn_steps = getattr(generation_output, "decoder_attentions", None)
    if attn_steps is None:
        return []

    parsed = []
    for step_obj in attn_steps:
        # 常见结构: step_obj 是 tuple(layers), 每层 [bs, heads, q, k]
        if isinstance(step_obj, (tuple, list)) and len(step_obj) > 0:
            last_layer = step_obj[-1]
            parsed.append(last_layer)
        else:
            parsed.append(step_obj)
    return parsed


def _guess_image_token_indices(inputs, tokenizer):
    """尽量从输入 token 中识别 image token 位置。失败则返回空列表。"""
    if inputs is None:
        return []
    if isinstance(inputs, torch.Tensor):
        ids = inputs[0].detach().cpu().tolist()
    elif isinstance(inputs, np.ndarray):
        ids = inputs[0].tolist()
    else:
        return []

    special_token_candidates = [
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
        "</img>",
    ]
    special_ids = set()
    for tok in special_token_candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid >= 0:
                special_ids.add(int(tid))
        except Exception:
            continue
    if not special_ids:
        return []

    idxs = [i for i, tid in enumerate(ids) if int(tid) in special_ids]
    return idxs


def _vector_to_heatmap(vec: np.ndarray, out_w: int, out_h: int):
    """将一维注意力向量近似重排为方形热图，再 resize 到图像尺寸。"""
    if vec is None or vec.size == 0:
        return None
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.maximum(vec, 0.0)
    s = float(vec.sum())
    if s > 0:
        vec = vec / s

    side = int(np.sqrt(vec.size))
    if side < 2:
        return None
    usable = side * side
    vec = vec[:usable]
    grid = vec.reshape(side, side)
    # 归一化到 [0,1]
    gmin, gmax = float(grid.min()), float(grid.max())
    if gmax > gmin:
        grid = (grid - gmin) / (gmax - gmin)
    else:
        grid = np.zeros_like(grid, dtype=np.float32)

    hm = Image.fromarray(np.uint8(np.clip(grid * 255.0, 0, 255)), mode="L")
    hm = hm.resize((out_w, out_h), resample=Image.BILINEAR)
    return hm


def _overlay_heatmap_on_image(image: Image.Image, heatmap_l: Image.Image, alpha: float = 0.45):
    """将灰度热图叠加到原图，输出 RGB。"""
    base = image.convert("RGB")
    if heatmap_l is None:
        return base
    hm = np.asarray(heatmap_l, dtype=np.uint8)
    # 简单红黄伪彩
    r = hm
    g = np.minimum(255, hm * 2).astype(np.uint8)
    b = np.zeros_like(hm, dtype=np.uint8)
    color = np.stack([r, g, b], axis=-1)
    color_img = Image.fromarray(color, mode="RGB")
    return Image.blend(base, color_img, alpha=max(0.0, min(1.0, float(alpha))))


def _render_step_label(image: Image.Image, text: str):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((8, 8, 220, 32), fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return out


def export_dynamic_attention_from_generate(
    generation_output,
    generation_kwargs,
    tokenizer,
    image: Image.Image,
    out_prefix: str,
    max_steps: int = 48,
):
    """
    导出 step-by-step 动态注意力：
    - out_prefix + "_step_XXXX.png"（逐步帧）
    - out_prefix + "_dynamic.gif"（动图）
    - out_prefix + "_static_token_attn.png"（step 平均静态图）
    """
    step_attn_objs = _extract_step_attn_tensors(generation_output)
    if not step_attn_objs:
        return False, "no attentions in generation output"

    inputs = generation_kwargs.get("inputs", None)
    img_token_indices = _guess_image_token_indices(inputs, tokenizer)
    if not img_token_indices:
        return False, "cannot infer image token indices"

    if max_steps is None or int(max_steps) <= 0:
        max_steps = len(step_attn_objs)
    max_steps = min(int(max_steps), len(step_attn_objs))

    frames = []
    step_vecs = []
    img_w, img_h = image.size

    for i in range(max_steps):
        arr = _to_numpy_attention(step_attn_objs[i])
        if arr is None:
            continue
        # arr: [heads, q, k] -> 取最后 query，head 平均
        attn_last_q = arr[:, -1, :] if arr.shape[1] > 0 else None
        if attn_last_q is None or attn_last_q.shape[-1] <= 0:
            continue
        attn_token = attn_last_q.mean(axis=0)
        valid_indices = [idx for idx in img_token_indices if idx < attn_token.shape[0]]
        if not valid_indices:
            continue
        vec = attn_token[valid_indices]
        step_vecs.append(vec)
        hm = _vector_to_heatmap(vec, out_w=img_w, out_h=img_h)
        if hm is None:
            continue
        overlay = _overlay_heatmap_on_image(image, hm, alpha=0.45)
        overlay = _render_step_label(overlay, f"step {i + 1}")
        step_path = f"{out_prefix}_step_{i + 1:04d}.png"
        overlay.save(step_path)
        frames.append(overlay)

    if not frames:
        return False, "no valid dynamic frames generated"

    gif_path = f"{out_prefix}_dynamic.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
    )

    # step 平均，得到静态 token-attn 图
    try:
        max_len = min(v.shape[0] for v in step_vecs) if step_vecs else 0
        if max_len > 0:
            stack = np.stack([v[:max_len] for v in step_vecs], axis=0)
            avg_vec = stack.mean(axis=0)
            hm_static = _vector_to_heatmap(avg_vec, out_w=img_w, out_h=img_h)
            if hm_static is not None:
                static_overlay = _overlay_heatmap_on_image(image, hm_static, alpha=0.45)
                static_path = f"{out_prefix}_static_token_attn.png"
                static_overlay.save(static_path)
    except Exception:
        pass
    return True, None

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
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


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


def calculate_map(pred_indices_set: set, gt_indices_set: set, iou_thresholds: list = None) -> float:
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    aps = []
    for _ in iou_thresholds:
        tp = len(pred_indices_set & gt_indices_set)
        fp = len(pred_indices_set - gt_indices_set)
        fn = len(gt_indices_set - pred_indices_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        aps.append(precision)
    return float(np.mean(aps)) if aps else 0.0


def evaluate_localization_task(pred_dict: dict, gt_dict: dict, gt_regions: list) -> dict:
    pred_indices_set = set()
    for label, indices in pred_dict.items():
        pred_indices_set.update(indices)
    gt_indices_set = set()
    for label, indices in gt_dict.items():
        gt_indices_set.update(indices)
    tp = len(pred_indices_set & gt_indices_set)
    fp = len(pred_indices_set - gt_indices_set)
    fn = len(gt_indices_set - pred_indices_set)
    all_ious = [1.0 if idx in gt_indices_set else 0.0 for idx in pred_indices_set]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    map_value = calculate_map(pred_indices_set, gt_indices_set)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": mean_iou, "map": map_value,
        "pred_indices": sorted(pred_indices_set), "gt_indices": sorted(gt_indices_set),
    }


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


def load_checkpoint_model(base_model_path, checkpoint_path, device="cuda"):
    """
    - base 为含 vlm-fo1 的官方目录：直接加载，再叠 checkpoint（LoRA 或完整）。
    - base 为 StageA 等输出目录（如 outputs/cervical_stage1_align_lora）：先加载默认基座，
      再加载 base 的 LoRA；若 checkpoint 与 base 不同，则 merge 后再加载 StageB LoRA 或完整权重。
    """
    if not checkpoint_path:
        checkpoint_path = base_model_path
    base_abs = os.path.abspath(os.path.realpath(base_model_path))
    ckpt_abs = os.path.abspath(os.path.realpath(checkpoint_path))

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
                print(f"加载 LoRA: {checkpoint_path}")
                model = PeftModel.from_pretrained(model, checkpoint_path)
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
    if not PEFT_AVAILABLE:
        print("Warning: PEFT 不可用，无法加载 LoRA，仅基座模型")
        return tokenizer, model, image_processors

    if _is_lora_dir(base_model_path):
        print(f"加载 StageA/base LoRA: {base_model_path}")
        model = PeftModel.from_pretrained(model, base_model_path)
        if ckpt_abs != base_abs:
            if _is_lora_dir(checkpoint_path):
                print(f"合并 StageA LoRA 后加载 StageB LoRA: {checkpoint_path}")
                model = model.merge_and_unload()
                model = PeftModel.from_pretrained(model, checkpoint_path)
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
        model = PeftModel.from_pretrained(model, checkpoint_path)
    elif os.path.exists(os.path.join(checkpoint_path, "config.json")) and _is_official_vlm_fo1_model_dir(checkpoint_path):
        print(f"从 checkpoint 加载完整模型: {checkpoint_path}")
        tokenizer, model, image_processors = load_pretrained_model(
            checkpoint_path, load_8bit=False, load_4bit=False, device=device
        )
    else:
        print("Warning: base 无 LoRA 且 checkpoint 无法加载，仅基座模型")
    return tokenizer, model, image_processors


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
    extractor.use_hierarchical_fusion = True
    extractor.hierarchical_num_heads = int(num_heads)
    fusion_mode = str(fusion_mode).strip().lower()
    if fusion_mode not in ("legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"):
        fusion_mode = "legacy"
    extractor.hierarchical_fusion_mode = fusion_mode
    if not hasattr(extractor, "hierarchical_fusion"):
        extractor.hierarchical_fusion = HierarchicalFeatureFusion(
            feature_dim=512,
            out_dim=2048,
            num_heads=int(num_heads),
            fusion_mode=fusion_mode,
        )
    elif hasattr(extractor.hierarchical_fusion, "fusion_mode"):
        extractor.hierarchical_fusion.fusion_mode = fusion_mode
    ref_param = next(model.parameters())
    extractor.hierarchical_fusion.to(device=ref_param.device, dtype=ref_param.dtype)
    print(f"[Eval] Hierarchical fusion enabled (heads={num_heads}, mode={fusion_mode}).")


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
    dynamic_attn_max_steps: int = 48,
    task_ids_to_eval=None,
    max_samples_per_task: int = 10,
    only_eval_attn_samples: bool = True,
):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
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
    if int(max_samples_per_task) > 0:
        print(f"[Eval] 每个任务最多评估样本数: {int(max_samples_per_task)}")
    else:
        print("[Eval] 每个任务评估样本数: 不限制")
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
    if save_hierarchical_attn_dir and only_eval_attn_samples:
        print("[Eval] 仅评估会导出注意力的样本，其余样本将跳过。")

    # 预先抽样评估索引：每个任务随机取 max_samples_per_task 条，仅遍历这些样本
    eval_indices = None
    data_source = getattr(dataset, "data", None)
    if isinstance(data_source, list) and int(max_samples_per_task) > 0:
        task_to_indices_for_eval = defaultdict(list)
        for i, item in enumerate(data_source):
            t = get_task_type_from_item(item if isinstance(item, dict) else {})
            if t in allowed_task_types:
                task_to_indices_for_eval[t].append(i)
        rng_eval = random.Random(int(save_hierarchical_attn_seed))
        selected_eval_indices = set()
        for t in sorted(allowed_task_types):
            idxs = task_to_indices_for_eval.get(t, [])
            k = min(int(max_samples_per_task), len(idxs))
            if k > 0:
                selected_eval_indices.update(rng_eval.sample(idxs, k))
        eval_indices = sorted(selected_eval_indices)
        print(f"[Eval] 预抽样后实际评估样本数: {len(eval_indices)}")
    else:
        eval_indices = list(range(len(dataset)))
        if int(max_samples_per_task) > 0:
            print("Warning: 无法预抽样（dataset.data 不可用），将遍历全量并在循环中按任务上限截断。")

    task_metrics = {
        "type0": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type1": {"tp": 0, "fp": 0, "fn": 0, "all_ious": [], "maps": []},
        "type2": {"bleus": [], "rouge_ls": [], "rouge_1s": [], "rouge_2s": []},
        "type3": [],  # list of (pred_class, gt_class)
        "type4": [],  # list of (pred_class, gt_class)
        "type5": [],  # list of (pred_class, gt_class)
    }
    task_counts = defaultdict(int)
    results = []
    dynamic_attn_unsupported_warned = False
    dynamic_attn_enabled = True
    attention_export_success_count = 0

    for idx in tqdm(eval_indices, desc="Evaluating", ncols=100):
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

            should_save_attn = False
            if save_hierarchical_attn_dir:
                if save_hierarchical_attn_per_task_max > 0 and attn_selected_indices_by_task:
                    should_save_attn = idx in attn_selected_indices_by_task.get(task_type, set())
                elif save_hierarchical_attn_every > 0:
                    should_save_attn = (idx % save_hierarchical_attn_every == 0)

            # 仅保留会导出注意力的样本，节省时间
            if save_hierarchical_attn_dir and only_eval_attn_samples and not should_save_attn:
                continue

            # 每个任务类型仅评估前 N 条（N<=0 表示不限制）
            if int(max_samples_per_task) > 0 and task_counts[task_type] >= int(max_samples_per_task):
                continue
            task_counts[task_type] += 1

            task_id = None
            try:
                if data_item.get("task_id") is not None:
                    task_id = int(data_item["task_id"])
            except Exception:
                pass
            user_text = add_task_constraints_to_user_text(user_text, task_id or -1, language=language)

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
                task_ids=[task_id if task_id is not None else -1],
            )
            # eval 场景下禁用 streamer，避免异常时 tokenizer decode 触发 NoneType 报错
            generation_kwargs.pop("streamer", None)

            generation_output = None
            if save_hierarchical_attn_dir and should_save_attn:
                try:
                    generation_output = model.generate(
                        **generation_kwargs,
                        return_dict_in_generate=True,
                        output_attentions=True,
                    )
                except TypeError as e:
                    if not dynamic_attn_unsupported_warned:
                        print(f"Warning: model.generate 不支持动态注意力导出参数，降级为普通推理。原因: {e}")
                        dynamic_attn_unsupported_warned = True
                    dynamic_attn_enabled = False
                    generation_output = model.generate(**generation_kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Warning: sample {idx} 导出动态注意力触发 OOM，降级为普通推理。")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        dynamic_attn_enabled = False
                        try:
                            retry_kwargs = dict(generation_kwargs)
                            retry_kwargs["max_new_tokens"] = min(int(retry_kwargs.get("max_new_tokens", 512)), 128)
                            generation_output = model.generate(**retry_kwargs)
                        except RuntimeError as e2:
                            if "out of memory" in str(e2).lower():
                                print(f"Warning: sample {idx} 普通推理仍 OOM，跳过该样本。")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            raise
                    else:
                        raise
                except Exception:
                    # 其它异常不在这里吞掉，交给外层样本级异常处理
                    raise
            else:
                try:
                    generation_output = model.generate(**generation_kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Warning: sample {idx} 推理 OOM，缩短生成长度重试。")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        retry_kwargs = dict(generation_kwargs)
                        retry_kwargs["max_new_tokens"] = min(int(retry_kwargs.get("max_new_tokens", 512)), 128)
                        try:
                            generation_output = model.generate(**retry_kwargs)
                        except RuntimeError as e2:
                            if "out of memory" in str(e2).lower():
                                print(f"Warning: sample {idx} 重试仍 OOM，跳过该样本。")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            raise
                    else:
                        raise

            output_ids = generation_output.sequences if hasattr(generation_output, "sequences") else generation_output
            output_text = tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
                skip_special_tokens=False,
            ).strip()
            attention_saved_for_sample = False
            if should_save_attn:
                try:
                    _, static_dir, dynamic_dir = _ensure_task_attn_dirs(save_hierarchical_attn_dir, task_type)
                    static_prefix = f"sample_{idx}"
                    core = model.get_model() if hasattr(model, "get_model") else model
                    extractor = getattr(core, "object_vp_extractor", None)
                    hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
                    if hf is not None and getattr(extractor, "use_hierarchical_fusion", False):
                        attn_path = os.path.join(static_dir, f"{static_prefix}_hierarchical.png")
                        visualize_hierarchical_attention(hf, save_path=attn_path)
                        attention_saved_for_sample = True
                    if dynamic_attn_enabled and hasattr(generation_output, "sequences"):
                        dynamic_prefix = os.path.join(dynamic_dir, static_prefix)
                        ok, err = export_dynamic_attention_from_generate(
                            generation_output=generation_output,
                            generation_kwargs=generation_kwargs,
                            tokenizer=tokenizer,
                            image=image,
                            out_prefix=dynamic_prefix,
                            max_steps=dynamic_attn_max_steps,
                        )
                        if ok:
                            attention_saved_for_sample = True
                        elif err:
                            print(f"Warning: dynamic attention not exported for sample {idx}: {err}")
                except Exception as e:
                    print(f"Warning: failed to save hierarchical attention for sample {idx}: {e}")
            if should_save_attn and not attention_saved_for_sample:
                # 本脚本目标是注意力可视化：若当前样本未产出任何注意力图，则不计入评估
                print(f"Warning: sample {idx} 未成功导出注意力，跳过该样本评估。")
                continue
            if should_save_attn and attention_saved_for_sample:
                attention_export_success_count += 1

            result_item = {
                "id": data_item.get("id", f"sample_{idx}"),
                "image": image_path,
                "task_type": task_type,
                "user_text": user_text,
                "prediction": output_text,
                "ground_truth": assistant_text,
            }

            if task_type in ["type0", "type1"]:
                pred_dict = parse_grounding_output(output_text)
                gt_dict = parse_grounding_output(assistant_text)
                metrics = evaluate_localization_task(pred_dict, gt_dict, gt_regions)
                for k in ["tp", "fp", "fn"]:
                    task_metrics[task_type][k] += metrics[k]
                task_metrics[task_type]["all_ious"].extend(
                    [1.0 if i in metrics["gt_indices"] else 0.0 for i in metrics["pred_indices"]]
                )
                task_metrics[task_type]["maps"].append(metrics["map"])
                result_item.update(metrics)

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

        # 所有目标任务均达到上限后提前结束，避免遍历完整数据集
        if int(max_samples_per_task) > 0:
            if all(task_counts.get(t, 0) >= int(max_samples_per_task) for t in allowed_task_types):
                print("[Eval] 已达到每任务样本上限，提前结束评估。")
                break

    # 汇总指标
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

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        # per_class 可序列化
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 打印
    print("\n" + "=" * 60)
    print("阶段2 评估结果汇总（任务 0,1,2,4,5）")
    print("=" * 60)
    print("\n任务分布:")
    for t, c in task_counts.items():
        print(f"  {t}: {c} 个样本")

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

    print("\n" + "=" * 60)
    if save_hierarchical_attn_dir and attention_export_success_count == 0:
        raise RuntimeError("未成功导出任何注意力图，已按注意力优先策略终止。请检查显存/模型配置后重试。")
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
    parser.add_argument("--test_json", type=str, default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/final/stage2_val.json", help="测试集 JSON")
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
    parser.add_argument("--dynamic_attn_max_steps", type=int, default=48, help="动态注意力最多导出步数")
    parser.add_argument("--task_ids", type=str, default="0,1,2,3,4,5", help="要评估的 task_id 列表，如 2,3,4,5")
    parser.add_argument("--max_samples_per_task", type=int, default=10, help="每个 task_id 最多评估样本数，<=0 表示不限制")
    parser.add_argument("--only_eval_attn_samples", dest="only_eval_attn_samples", action="store_true", default=True, help="仅评估会导出注意力的样本（默认开启）")
    parser.add_argument("--eval_all_samples", dest="only_eval_attn_samples", action="store_false", help="评估所有样本（关闭仅注意力样本模式）")
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

    tokenizer, model, image_processors = load_checkpoint_model(args.base_model, args.checkpoint, device=args.device)
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
        dynamic_attn_max_steps=args.dynamic_attn_max_steps,
        task_ids_to_eval=selected_task_ids,
        max_samples_per_task=args.max_samples_per_task,
        only_eval_attn_samples=args.only_eval_attn_samples,
    )
    print(f"\n评估完成，结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
