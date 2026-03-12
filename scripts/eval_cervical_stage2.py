#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞定位第二阶段多任务评估脚本

支持五种任务类型的评估（通过task_id字段识别）：
1. 类型0 (task_id=0)：基础定位(OD) - 使用IoU、Precision、Recall、F1、mAP
2. 类型1 (task_id=1)：带描述的特征定位（REC）- 使用IoU、Precision、Recall、F1、mAP
3. 类型2 (task_id=2)：区域描述(Region Caption) - 使用BLEU、ROUGE
4. 类型3 (task_id=3)：细胞类型分类(Classification) - 使用准确率
5. 类型4 (task_id=4)：来源分类(Classification) - 使用准确率
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import re

# BLEU和ROUGE评估库
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False
    print("Warning: nltk or rouge-score not available. Installing...")
    print("Please run: pip install nltk rouge-score")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    extract_predictions_to_bboxes,
    draw_bboxes_and_save
)
from utils.data_utils import (
    calculate_iou,
    match_bboxes,
    parse_grounding_output
)

# ---- Type3 (cell type classification) label sets ----
TYPE3_CELL_TYPE_CLASSES = [
    "ASCUS",
    "ASC_H",
    "EpitheliumNormal",
    "GlandTumor",
    "HPV",
    "HSIL",
    "LSIL",
]

# ---- Type4 (source classification) label sets ----
TYPE4_SOURCE_CLASSES = [
    "上皮成团",
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮底层",
    "宫颈鳞状上皮表层",
    "腺上皮成团",
]

# LoRA支持
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA weights will not be loaded.")


def get_task_type_from_item(data_item: dict) -> str:
    """
    从数据项的task_id字段获取任务类型（只使用task_id或task_type字段）
    
    Args:
        data_item: 数据项字典
        
    Returns:
        任务类型: 'type0', 'type1', 'type2', 'type3', 'type4', 'unknown'
    """
    # 优先使用task_id字段（新数据集使用task_id 0-4）
    task_id = data_item.get("task_id", None)
    if task_id is not None:
        if isinstance(task_id, (int, np.integer)):
            if 0 <= int(task_id) <= 4:
                return f"type{int(task_id)}"
        elif isinstance(task_id, str):
            try:
                tid = int(task_id.strip())
                if 0 <= tid <= 4:
                    return f"type{tid}"
            except ValueError:
                pass
    
    # 回退到旧的task_type字段（兼容旧数据集，使用1-4）
    raw = data_item.get("task_type", None)
    if raw is not None:
        if isinstance(raw, (int, np.integer)):
            if 1 <= int(raw) <= 4:
                return f"type{int(raw)}"
        elif isinstance(raw, str):
            s = raw.strip().lower()
            if s.isdigit():
                n = int(s)
                if 1 <= n <= 4:
                    return f"type{n}"
            if s in {"type1", "type2", "type3", "type4"}:
                return s
            if s in {"1", "2", "3", "4"}:
                return f"type{s}"
            if s.startswith("type") and s[4:].isdigit():
                n = int(s[4:])
                if 1 <= n <= 4:
                    return f"type{n}"
    
    # 如果都没有，返回unknown
    return "unknown"


def calculate_map(pred_indices_set: set, gt_indices_set: set, iou_thresholds: list = None) -> float:
    """
    计算mAP (mean Average Precision)
    
    Args:
        pred_indices_set: 预测的region索引集合
        gt_indices_set: 真实的region索引集合
        iou_thresholds: IoU阈值列表，默认使用COCO标准 [0.5:0.05:0.95]
        
    Returns:
        mAP值
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    # 对于基于索引匹配的任务，IoU要么是1.0（匹配）要么是0.0（不匹配）
    # 计算每个IoU阈值下的AP
    aps = []
    for iou_threshold in iou_thresholds:
        # 计算TP, FP, FN
        tp = len(pred_indices_set & gt_indices_set)
        fp = len(pred_indices_set - gt_indices_set)
        fn = len(gt_indices_set - pred_indices_set)
        
        # 计算precision和recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 对于这种任务，AP就是precision（因为只有一个recall点）
        # 但为了符合mAP标准，我们使用precision作为AP
        aps.append(precision)
    
    # mAP是所有IoU阈值下AP的平均值
    map_value = np.mean(aps) if len(aps) > 0 else 0.0
    return map_value


def evaluate_localization_task(pred_dict: dict, gt_dict: dict, gt_regions: list) -> dict:
    """
    评估定位任务（类型1和类型2）
    
    Args:
        pred_dict: 预测结果字典 {label: [region_indices]}
        gt_dict: 真实标签字典 {label: [region_indices]}
        gt_regions: 真实区域列表
        
    Returns:
        评估指标字典
    """
    # 提取region索引
    pred_indices_set = set()
    for label, indices in pred_dict.items():
        pred_indices_set.update(indices)
    
    gt_indices_set = set()
    for label, indices in gt_dict.items():
        gt_indices_set.update(indices)
    
    # 计算TP, FP, FN
    tp = len(pred_indices_set & gt_indices_set)
    fp = len(pred_indices_set - gt_indices_set)
    fn = len(gt_indices_set - pred_indices_set)
    
    # 计算IoU（对于定位任务，正确预测的region IoU为1.0）
    all_ious = []
    for pred_idx in pred_indices_set:
        if pred_idx in gt_indices_set:
            all_ious.append(1.0)
        else:
            all_ious.append(0.0)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    
    # 计算mAP
    map_value = calculate_map(pred_indices_set, gt_indices_set)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'map': map_value,
        'pred_indices': sorted(pred_indices_set),
        'gt_indices': sorted(gt_indices_set)
    }


def evaluate_caption_task(pred_text: str, gt_text: str) -> dict:
    """
    评估区域描述任务（类型3）
    
    Args:
        pred_text: 预测文本
        gt_text: 真实文本
        
    Returns:
        评估指标字典（使用BLEU和ROUGE）
    """
    result = {
        'pred_text': pred_text,
        'gt_text': gt_text,
        'bleu': 0.0,
        'rouge_l': 0.0,
        'rouge_1': 0.0,
        'rouge_2': 0.0
    }
    
    if not BLEU_ROUGE_AVAILABLE:
        # 库不可用时返回默认值，不打印警告（避免重复输出）
        return result
    
    # 清理特殊标记（用于BLEU和ROUGE计算）
    # 1) 清理 chatml 特殊标记
    pred_text_clean = re.sub(r'<\|im_end\|>', '', str(pred_text)).strip()
    pred_text_clean = re.sub(r'<\|im_start\|>', '', pred_text_clean).strip()
    gt_text_clean = re.sub(r'<\|im_end\|>', '', str(gt_text)).strip()
    gt_text_clean = re.sub(r'<\|im_start\|>', '', gt_text_clean).strip()
    
    # 2) 清理多模态/定位模板标记，避免把 <ground>/<objects>/<region*> 当作文本参与 BLEU/ROUGE
    #    如果存在 <ground>...</ground>，优先取内部文本作为“描述内容”
    m = re.search(r"<ground>(.*?)</ground>", pred_text_clean, flags=re.DOTALL)
    if m is not None:
        pred_text_clean = m.group(1).strip()
    pred_text_clean = re.sub(r"<objects>.*?</objects>", "", pred_text_clean, flags=re.DOTALL).strip()
    pred_text_clean = re.sub(r"</?ground>", "", pred_text_clean).strip()
    pred_text_clean = re.sub(r"</?objects>", "", pred_text_clean).strip()
    pred_text_clean = re.sub(r"<region\\d+>", "", pred_text_clean).strip()
    gt_text_clean = re.sub(r"<region\\d+>", "", gt_text_clean).strip()

    # 3) 分词策略：对中文（无空格）启用字符级分词，避免 BLEU/ROUGE 全为 0 的“假象”
    def _needs_char_tokenize(s: str) -> bool:
        if " " in s:
            return False
        # 粗略判断是否包含中日韩字符
        return re.search(r"[\\u4e00-\\u9fff]", s) is not None

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
    
    # 计算BLEU分数
    try:
        smoothing = SmoothingFunction().method1
        # 使用1-gram到4-gram的BLEU分数
        bleu_1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        # 使用BLEU-4作为主要指标
        result['bleu'] = bleu_4
        result['bleu_1'] = bleu_1
        result['bleu_2'] = bleu_2
        result['bleu_3'] = bleu_3
    except Exception as e:
        # 如果计算失败，所有BLEU指标设为0
        result['bleu'] = 0.0
        result['bleu_1'] = 0.0
        result['bleu_2'] = 0.0
        result['bleu_3'] = 0.0
    
    # 计算ROUGE分数（也使用清理后的文本）
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gt_text_for_rouge, pred_text_for_rouge)
        result['rouge_1'] = scores['rouge1'].fmeasure
        result['rouge_2'] = scores['rouge2'].fmeasure
        result['rouge_l'] = scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Warning: ROUGE calculation failed: {e}")
        result['rouge_l'] = 0.0
        result['rouge_1'] = 0.0
        result['rouge_2'] = 0.0
    
    return result


def evaluate_classification_task(pred_text: str, gt_text: str, valid_classes: list) -> dict:
    """
    评估分类任务（类型3或类型4）
    
    Args:
        pred_text: 预测文本
        gt_text: 真实文本
        valid_classes: 有效类别列表
        
    Returns:
        评估指标字典
    """
    pred_class = _extract_class_from_text(pred_text, valid_classes)
    gt_class = _extract_class_from_text(gt_text, valid_classes)
    
    correct = (pred_class == gt_class) and (pred_class is not None) and (gt_class is not None)
    
    return {
        "correct": correct,
        "pred_class": pred_class,
        "gt_class": gt_class,
        "pred_text": pred_text,
        "gt_text": gt_text,
    }


def _extract_class_from_text(text: str, valid_classes: list) -> str | None:
    """Return the first matched class in text; if exact match exists, prefer it."""
    if text is None:
        return None
    t = str(text).strip()
    if t in valid_classes:
        return t
    for cls in valid_classes:
        if cls in t:
            return cls
    return None




def load_checkpoint_model(base_model_path, checkpoint_path, device="cuda"):
    """
    加载checkpoint模型（支持LoRA）
    
    Args:
        base_model_path: 基础模型路径
        checkpoint_path: checkpoint路径
        device: 设备
        
    Returns:
        tokenizer, model, image_processors
    """
    print(f"\n加载基础模型: {base_model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        base_model_path,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    
    # 检查是否有LoRA权重
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        if PEFT_AVAILABLE:
            print(f"加载LoRA权重: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print("LoRA权重加载成功")
        else:
            print("Warning: LoRA权重存在但PEFT不可用，使用基础模型")
    else:
        # 尝试从checkpoint目录加载完整模型
        if os.path.exists(os.path.join(checkpoint_path, "config.json")):
            print(f"从checkpoint加载完整模型: {checkpoint_path}")
            tokenizer, model, image_processors = load_pretrained_model(
                checkpoint_path,
                load_8bit=False,
                load_4bit=False,
                device=device
            )
        else:
            print(f"Warning: 未找到模型权重，使用基础模型")
    
    return tokenizer, model, image_processors


def evaluate_model(
    model,
    tokenizer,
    image_processors,
    dataset,
    device="cuda",
    output_dir="eval_results_stage2",
    visualize=True,
    model_name=None,
    debug_type4_n: int = 0,
    path_feature_root: str = None,
):
    """
    多任务评估模型性能
    
    Args:
        model: 模型对象
        tokenizer: Tokenizer对象
        image_processors: 图像处理器元组
        dataset: 评估数据集
        device: 设备
        output_dir: 输出目录
        visualize: 是否保存可视化结果
        model_name: 模型名称
        
    Returns:
        评估结果字典
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # 按任务类型统计指标
    task_metrics = {
        'type0': {'tp': 0, 'fp': 0, 'fn': 0, 'all_ious': [], 'maps': []},
        'type1': {'tp': 0, 'fp': 0, 'fn': 0, 'all_ious': [], 'maps': []},
        'type2': {'bleus': [], 'rouge_ls': [], 'rouge_1s': [], 'rouge_2s': []},
        'type3': {'correct': 0, 'total': 0},
        'type4': {'correct': 0, 'total': 0}
    }
    
    results = []
    task_counts = defaultdict(int)
    # 用于调试：记录每种任务类型已打印的样本数量
    debug_counters = {
        'type0': 0,
        'type1': 0,
        'type2': 0,
        'type3': 0,
        'type4': 0
    }
    
    print(f"评估 {len(dataset)} 个样本...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            try:
                # 获取数据样本
                sample = dataset[idx]
                
                # 获取原始数据项
                if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                    original_idx = dataset.indices[idx]
                    data_item = dataset.dataset.data[original_idx]
                elif hasattr(dataset, 'data'):
                    data_item = dataset.data[idx]
                else:
                    data_item = dataset[idx] if isinstance(dataset[idx], dict) else None
                
                if data_item is None:
                    continue
                
                # 准备输入
                image_path = data_item['image']
                image = Image.open(image_path).convert("RGB")
                
                # 构建messages
                conversations = data_item['conversations']
                user_text = None
                assistant_text = None
                for conv in conversations:
                    if conv['from'] == 'user':
                        user_text = conv['value'].replace('<image>', '').strip()
                        if user_text.startswith('\n'):
                            user_text = user_text[1:]
                    elif conv['from'] == 'assistant':
                        assistant_text = conv['value']
                
                if user_text is None or assistant_text is None:
                    continue
                
                # 分类任务类型：使用数据集中的 task_id 或 task_type 字段
                task_type = get_task_type_from_item(data_item)
                task_counts[task_type] += 1

                # ------------------------------------------------------------------
                # 为不同任务增加“输出格式约束”，与训练/推理阶段保持一致，降低多任务时的任务混淆
                # - Type2（区域描述）：强约束只输出描述文本，禁止 <ground>/<objects>/<region*> 模板
                # - Type3/4（分类）：强约束只输出一个候选类别字符串
                # ------------------------------------------------------------------
                task_id_raw = data_item.get("task_id", None)
                task_id = None
                try:
                    if task_id_raw is not None:
                        task_id = int(task_id_raw)
                except Exception:
                    task_id = None

                # 检查是否已经包含约束（避免重复添加）
                has_type2_constraint = any(keyword in user_text for keyword in [
                    "描述该区域的病理属性", "不要输出任何 <ground>", "<objects>", "<region*>"
                ])
                has_type34_constraint = any(keyword in user_text for keyword in [
                    "只输出一个类别名称", "不要输出解释"
                ])

                if task_id == 2 and not has_type2_constraint:
                    user_text = (
                        user_text
                        + "\n"
                        + "请直接用一句话描述该区域的病理属性（例如阳性/阴性、细胞类型、来源、核特征等）。"
                          "不要输出任何 <ground>、<objects>、<region*> 等标签或坐标。"
                    )
                elif task_id in (3, 4) and not has_type34_constraint:
                    user_text = (
                        user_text
                        + "\n"
                        + "请从问题给出的候选类别中选择，并且只输出一个类别名称，不要输出解释、标点或其他内容。"
                    )
                
                # 获取真实边界框（用于定位任务）
                gt_regions = data_item.get('regions', [])
                bbox_list_for_model = [region['bbox'] for region in gt_regions]
                
                # 构建messages用于推理
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": user_text}
                    ],
                    "bbox_list": bbox_list_for_model
                }]
                
                # 准备模型输入
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
                    # 在推理阶段同样传入病理特征根目录，使得与训练阶段一致地加载 HoVer-Net 特征
                    path_feature_root=path_feature_root,
                )
                
                # 生成预测
                output_ids = model.generate(**generation_kwargs)
                output_text = tokenizer.decode(
                    output_ids[0, generation_kwargs['inputs'].shape[1]:],
                    skip_special_tokens=False
                ).strip()
                
                # 调试打印：每种任务类型打印前10条结果
                if task_type in debug_counters and debug_counters[task_type] < 10:
                    debug_counters[task_type] += 1
                    print("\n" + "=" * 80)
                    print(f"[{task_type.upper()} Debug {debug_counters[task_type]}/10] Sample ID: {data_item.get('id', f'sample_{idx}')}")
                    print("-" * 80)
                    print(f"User Text: {user_text[:200]}")
                    print(f"Model Output: {output_text[:500]}")
                    print(f"Ground Truth: {assistant_text[:500]}")
                    if task_type in ['type0', 'type1']:
                        pred_dict = parse_grounding_output(output_text)
                        gt_dict = parse_grounding_output(assistant_text)
                        print(f"Predicted regions: {pred_dict}")
                        print(f"GT regions: {gt_dict}")
                    elif task_type in ['type3', 'type4']:
                        valid_classes = TYPE3_CELL_TYPE_CLASSES if task_type == 'type3' else TYPE4_SOURCE_CLASSES
                        pred_class = _extract_class_from_text(output_text, valid_classes)
                        gt_class = _extract_class_from_text(assistant_text, valid_classes)
                        print(f"Predicted class: {pred_class}")
                        print(f"GT class: {gt_class}")
                        print(f"Correct: {pred_class == gt_class and pred_class is not None}")
                    print("=" * 80)
                
                # 根据任务类型进行评估
                result_item = {
                    "id": data_item.get("id", f"sample_{idx}"),
                    "image": image_path,
                    "task_type": task_type,
                    "user_text": user_text,
                    "prediction": output_text,
                    "ground_truth": assistant_text
                }
                
                if task_type in ['type0', 'type1']:
                    # 定位任务评估（type0: 基础定位, type1: 带描述的特征定位）
                    pred_dict = parse_grounding_output(output_text)
                    gt_dict = parse_grounding_output(assistant_text)
                    
                    metrics = evaluate_localization_task(pred_dict, gt_dict, gt_regions)
                    task_metrics[task_type]['tp'] += metrics['tp']
                    task_metrics[task_type]['fp'] += metrics['fp']
                    task_metrics[task_type]['fn'] += metrics['fn']
                    task_metrics[task_type]['all_ious'].extend(
                        [1.0 if idx in metrics['gt_indices'] else 0.0 
                         for idx in metrics['pred_indices']]
                    )
                    task_metrics[task_type]['maps'].append(metrics['map'])
                    
                    result_item.update(metrics)
                    
                elif task_type == 'type2':
                    # 区域描述任务评估
                    metrics = evaluate_caption_task(output_text, assistant_text)
                    task_metrics[task_type]['bleus'].append(metrics['bleu'])
                    task_metrics[task_type]['rouge_ls'].append(metrics['rouge_l'])
                    task_metrics[task_type]['rouge_1s'].append(metrics['rouge_1'])
                    task_metrics[task_type]['rouge_2s'].append(metrics['rouge_2'])
                    
                    result_item.update(metrics)
                    
                elif task_type == 'type3':
                    # 细胞类型分类任务评估（单轮）
                    metrics = evaluate_classification_task(output_text, assistant_text, TYPE3_CELL_TYPE_CLASSES)
                    task_metrics[task_type]['total'] += 1
                    if metrics['correct']:
                        task_metrics[task_type]['correct'] += 1
                    
                    result_item.update(metrics)
                    
                elif task_type == 'type4':
                    # 来源分类任务评估（单轮）
                    metrics = evaluate_classification_task(output_text, assistant_text, TYPE4_SOURCE_CLASSES)
                    task_metrics[task_type]['total'] += 1
                    if metrics['correct']:
                        task_metrics[task_type]['correct'] += 1
                    
                    result_item.update(metrics)
                
                results.append(result_item)
                
                # 可视化（仅对定位任务，每10个样本保存一次）
                if visualize and task_type in ['type0', 'type1'] and idx % 10 == 0:
                    vis_path = os.path.join(vis_dir, f"sample_{idx}_{task_type}.jpg")
                    
                    # 构建bbox字典用于可视化
                    pred_dict = parse_grounding_output(output_text)
                    gt_dict = parse_grounding_output(assistant_text)
                    
                    # 建立region_id到bbox的映射
                    region_id_to_bbox = {}
                    for region in gt_regions:
                        region_id = region.get('region_id', '')
                        if region_id:
                            match = re.match(r'region(\d+)', region_id)
                            if match:
                                region_idx = int(match.group(1))
                                region_id_to_bbox[region_idx] = region['bbox']
                    
                    # 获取预测和GT的bbox
                    pred_indices_set = set()
                    for label, indices in pred_dict.items():
                        pred_indices_set.update(indices)
                    
                    gt_indices_set = set()
                    for label, indices in gt_dict.items():
                        gt_indices_set.update(indices)
                    
                    pred_bboxes = [region_id_to_bbox[idx] for idx in sorted(pred_indices_set) if idx in region_id_to_bbox]
                    gt_bboxes_sorted = [region_id_to_bbox[idx] for idx in sorted(gt_indices_set) if idx in region_id_to_bbox]
                    
                    fo1_bboxes = {}
                    if len(pred_bboxes) > 0:
                        fo1_bboxes["positive cervical cell"] = pred_bboxes
                    
                    draw_bboxes_and_save(
                        image=image,
                        fo1_bboxes=fo1_bboxes,
                        output_path=vis_path,
                        color='red'
                    )
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算总体指标
    metrics = {}
    
    # 任务类型统计
    metrics['task_distribution'] = dict(task_counts)
    
    # 类型0和类型1：定位任务指标
    for task_type in ['type0', 'type1']:
        tp = task_metrics[task_type]['tp']
        fp = task_metrics[task_type]['fp']
        fn = task_metrics[task_type]['fn']
        all_ious = task_metrics[task_type]['all_ious']
        maps = task_metrics[task_type]['maps']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
        mean_map = np.mean(maps) if len(maps) > 0 else 0.0
        
        metrics[f'{task_type}_precision'] = precision
        metrics[f'{task_type}_recall'] = recall
        metrics[f'{task_type}_f1'] = f1
        metrics[f'{task_type}_mean_iou'] = mean_iou
        metrics[f'{task_type}_map'] = mean_map
        metrics[f'{task_type}_tp'] = tp
        metrics[f'{task_type}_fp'] = fp
        metrics[f'{task_type}_fn'] = fn
    
    # 类型2：区域描述任务指标
    if len(task_metrics['type2']['bleus']) > 0:
        metrics['type2_mean_bleu'] = np.mean(task_metrics['type2']['bleus'])
        metrics['type2_mean_rouge_l'] = np.mean(task_metrics['type2']['rouge_ls'])
        metrics['type2_mean_rouge_1'] = np.mean(task_metrics['type2']['rouge_1s'])
        metrics['type2_mean_rouge_2'] = np.mean(task_metrics['type2']['rouge_2s'])
    
    # 类型3：细胞类型分类任务指标
    if task_metrics['type3']['total'] > 0:
        total = task_metrics['type3']['total']
        metrics['type3_accuracy'] = task_metrics['type3']['correct'] / total
        metrics['type3_correct'] = task_metrics['type3']['correct']
        metrics['type3_total'] = total
    
    # 类型4：来源分类任务指标
    if task_metrics['type4']['total'] > 0:
        total = task_metrics['type4']['total']
        metrics['type4_accuracy'] = task_metrics['type4']['correct'] / total
        metrics['type4_correct'] = task_metrics['type4']['correct']
        metrics['type4_total'] = total
    
    # 保存结果
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"\n任务分布:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} 个样本")
    
    print(f"\n类型0 (基础定位) 指标:")
    print(f"  Precision: {metrics.get('type0_precision', 0):.4f}")
    print(f"  Recall: {metrics.get('type0_recall', 0):.4f}")
    print(f"  F1: {metrics.get('type0_f1', 0):.4f}")
    print(f"  Mean IoU: {metrics.get('type0_mean_iou', 0):.4f}")
    print(f"  mAP: {metrics.get('type0_map', 0):.4f}")
    
    print(f"\n类型1 (带描述的特征定位) 指标:")
    print(f"  Precision: {metrics.get('type1_precision', 0):.4f}")
    print(f"  Recall: {metrics.get('type1_recall', 0):.4f}")
    print(f"  F1: {metrics.get('type1_f1', 0):.4f}")
    print(f"  Mean IoU: {metrics.get('type1_mean_iou', 0):.4f}")
    print(f"  mAP: {metrics.get('type1_map', 0):.4f}")
    
    print(f"\n类型2 (区域描述) 指标:")
    print(f"  Mean BLEU: {metrics.get('type2_mean_bleu', 0):.4f}")
    print(f"  Mean ROUGE-L: {metrics.get('type2_mean_rouge_l', 0):.4f}")
    print(f"  Mean ROUGE-1: {metrics.get('type2_mean_rouge_1', 0):.4f}")
    print(f"  Mean ROUGE-2: {metrics.get('type2_mean_rouge_2', 0):.4f}")
    
    print(f"\n类型3 (细胞类型分类) 指标:")
    print(f"  Accuracy: {metrics.get('type3_accuracy', 0):.4f}")
    print(f"  Correct: {metrics.get('type3_correct', 0)}")
    print(f"  Total: {metrics.get('type3_total', 0)}")
    
    print(f"\n类型4 (来源分类) 指标:")
    print(f"  Accuracy: {metrics.get('type4_accuracy', 0):.4f}")
    print(f"  Correct: {metrics.get('type4_correct', 0)}")
    print(f"  Total: {metrics.get('type4_total', 0)}")
    
    print("\n" + "=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM-FO1 Stage 2 multi-task model')
    parser.add_argument(
        '--base_model',
        type=str,
        default='resources/VLM-FO1_Qwen2.5-VL-3B-v01',
        help='Path to base model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/cervical_PAfull_B',
        help='Path to checkpoint (LoRA weights or full model)'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        # default='dataset/data20260115/stage2_val.json',  # 需要构建包含四种任务的测试集
        default='/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/final/stage2_val.json', 
        help='Path to evaluation data JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='eval_result/PAfull_B',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--path_feature_root',
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/path_features",
        help='Optional: directory of precomputed pathology features (*.pt). If set, pathology tokens will be injected during generation.'
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--debug_type4_n',
        type=int,
        default=10,
        help='Print first N type4 samples with detailed predictions/GT for debugging (deprecated, kept for compatibility)'
    )
    
    args = parser.parse_args()
    
    # 加载模型
    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device
    )
    
    # 加载数据集
    print(f"\n加载测试集: {args.test_json}")
    if not os.path.exists(args.test_json):
        print(f"Warning: 测试集文件不存在: {args.test_json}")
        print("请先构建包含四种任务类型的测试集")
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
        # 评估阶段同样传入病理特征目录，使数据加载逻辑与训练保持一致
        path_feature_root=args.path_feature_root,
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 评估模型
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        image_processors=image_processors,
        dataset=test_dataset,
        device=args.device,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        model_name=args.base_model,
        debug_type4_n=args.debug_type4_n,
        path_feature_root=args.path_feature_root,
    )
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

