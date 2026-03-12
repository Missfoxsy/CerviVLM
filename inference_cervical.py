#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞定位推理脚本

使用训练好的checkpoint进行推理，从测试集中选择一条数据进行推理。
"""

import torch
import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict
from PIL import Image
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
    load_image,
    make_message_context,
)
from vlm_fo1.constants import IGNORE_INDEX

# LoRA支持
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA weights will not be loaded.")


def normalize_label_to_en(label: str) -> str:
    """Normalize Chinese/variant labels to English to avoid font issues."""
    if not isinstance(label, str):
        return label
    key = label.strip().lower()
    mapping = {
        "阳性": "positive",
        "阴性": "negative",
        "阳性宫颈细胞": "positive cervical cell",
        "阴性宫颈细胞": "negative cervical cell",
        "positive": "positive",
        "negative": "negative",
        "positive cervical cell": "positive cervical cell",
        "negative cervical cell": "negative cervical cell",
    }
    return mapping.get(key, label)


def load_checkpoint_model(base_model_path, checkpoint_path, device="cuda"):
    """
    加载checkpoint模型（支持LoRA权重）
    
    Args:
        base_model_path: 基础模型路径
        checkpoint_path: checkpoint路径（包含LoRA权重）
        device: 设备
        
    Returns:
        tokenizer, model, image_processors
    """
    # 加载基础模型
    print(f"Loading base model from: {base_model_path}")
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
            print(f"Loading LoRA weights from: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            # 合并LoRA权重到基础模型（可选，用于推理加速）
            # model = model.merge_and_unload()
            print("LoRA weights loaded successfully")
        else:
            print("Warning: LoRA weights found but PEFT is not available. Using base model only.")
    else:
        print("No LoRA weights found, using base model only")
    
    model.eval()
    return tokenizer, model, image_processors


def prepare_messages_from_data(data_item):
    """
    从数据项准备消息格式
    
    Args:
        data_item: 数据项字典，包含image, conversations, regions
        
    Returns:
        messages: 符合VLM-FO1格式的消息列表
        bbox_list: 边界框列表
    """
    # 提取用户消息
    user_msg = None
    for conv in data_item['conversations']:
        if conv['from'] == 'user':
            user_msg = conv['value']
            break
    
    if user_msg is None:
        raise ValueError("No user message found in conversations")
    
    # 提取文本（移除<image>标签）
    user_text = user_msg.replace('<image>', '').strip()
    if user_text.startswith('\n'):
        user_text = user_text[1:]
    
    # 构建边界框列表
    bbox_list = []
    for region in data_item['regions']:
        bbox = region['bbox']
        if len(bbox) == 4:
            bbox_list.append(bbox)
    
    # 构建用户消息
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": data_item['image']}
            },
            {
                "type": "text",
                "text": user_text
            }
        ],
        "bbox_list": bbox_list
    }
    
    messages = [user_message]
    return messages, bbox_list


def main():
    parser = argparse.ArgumentParser(description='Inference with trained VLM-FO1 model for cervical cell localization')
    parser.add_argument(
        '--base_model',
        type=str,
        default='resources/VLM-FO1_Qwen2.5-VL-3B-v01',
        help='Path to base model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/cervical_finetune0115',
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        default='dataset/data20260115/stage1_val_region_text.json',
        help='Path to test JSON file'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=972,
        help='Index of sample to use for inference (default: 0)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='inference_results',
        help='Output directory for visualization'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--merge_lora',
        action='store_true',
        help='Merge LoRA weights into base model (faster inference)'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载测试数据
    print(f"Loading test data from: {args.test_json}")
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.sample_idx >= len(test_data):
        print(f"Error: sample_idx {args.sample_idx} is out of range. Total samples: {len(test_data)}")
        return
    
    # 选择数据样本
    data_item = test_data[args.sample_idx]
    print(f"\nUsing sample {args.sample_idx}: {data_item['id']}")
    print(f"Image path: {data_item['image']}")
    print(f"Number of regions: {len(data_item['regions'])}")
    
    # 检查图像文件是否存在
    if not os.path.exists(data_item['image']):
        print(f"Error: Image file not found: {data_item['image']}")
        return
    
    # 加载checkpoint模型
    print(f"\nLoading checkpoint model from: {args.checkpoint}")
    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device
    )
    
    # 如果启用merge_lora，合并LoRA权重
    if args.merge_lora and PEFT_AVAILABLE and hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("LoRA weights merged")
    
    # 准备消息和边界框
    print("\nPreparing input messages...")
    messages, bbox_list = prepare_messages_from_data(data_item)
    
    print(f"User text: {messages[0]['content'][1]['text']}")
    print(f"Number of bboxes: {len(bbox_list)}")
    if len(bbox_list) > 0:
        print(f"First bbox: {bbox_list[0]}")
    
    # 准备模型输入
    print("\nPreparing model inputs...")
    generation_kwargs = prepare_inputs(
        args.base_model,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=4096,
        top_p=0.05,
        temperature=0.0,
        do_sample=False
    )
    
    # 运行推理
    print("\nRunning inference...")
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(
            output_ids[0, generation_kwargs['inputs'].shape[1]:],
            skip_special_tokens=False
        ).strip()
    
    print(f"\nModel output: {outputs}")
    
    # 解析输出中的边界框
    print("\nExtracting predictions...")
    predicted_bboxes = extract_predictions_to_bboxes(outputs, bbox_list)
    predicted_bboxes = {normalize_label_to_en(k): v for k, v in predicted_bboxes.items()}
    
    print(f"Predicted bounding boxes (dict with {len(predicted_bboxes)} labels):")
    for label, bboxes in predicted_bboxes.items():
        print(f"  Label '{label}': {len(bboxes)} bboxes")
        for i, bbox in enumerate(bboxes):
            print(f"    Bbox {i}: {bbox}")
    
    # 加载原始图像
    image_path = data_item['image']
    img_pil = Image.open(image_path).convert("RGB")
    
    # 可视化结果
    output_image_path = os.path.join(args.output_dir, f"inference_{data_item['id']}.jpg")
    print(f"\nSaving visualization to: {output_image_path}")
    # predicted_bboxes已经是字典格式，可以直接使用
    draw_bboxes_and_save(
        image=img_pil.copy(),  # 使用copy避免修改原图
        fo1_bboxes=predicted_bboxes,
        output_path=output_image_path
    )
    
    # 保存ground truth bboxes用于对比
    gt_image_path = os.path.join(args.output_dir, f"ground_truth_{data_item['id']}.jpg")
    print(f"Saving ground truth visualization to: {gt_image_path}")
    # 从assistant消息中提取标签
    assistant_msg = None
    for conv in data_item['conversations']:
        if conv['from'] == 'assistant':
            assistant_msg = conv['value']
            break
    
    # Ground truth: 直接按 assistant 输出解析，支持“阳性/阴性”多类别（多个<ground>...）
    # assistant_msg 一般是形如：<ground>positive cervical cell</ground><objects><region0>...</objects>
    # 或者包含多个 ground/object 段落（对应不同类别）
    gt_bboxes_dict = {}
    if assistant_msg:
        gt_bboxes_dict = extract_predictions_to_bboxes(assistant_msg, bbox_list)
        gt_bboxes_dict = {normalize_label_to_en(k): v for k, v in gt_bboxes_dict.items()}
    # 兜底：如果解析不到（比如数据异常），至少把所有proposal画出来
    if not gt_bboxes_dict:
        gt_bboxes_dict = {"positive cervical cell": bbox_list}
    draw_bboxes_and_save(
        image=img_pil.copy(),  # 使用copy避免修改原图
        fo1_bboxes=gt_bboxes_dict,
        output_path=gt_image_path
    )
    
    # 保存结果到JSON
    num_pred_bboxes = sum(len(v) for v in predicted_bboxes.values()) if isinstance(predicted_bboxes, dict) else 0
    num_gt_bboxes = sum(len(v) for v in gt_bboxes_dict.values()) if isinstance(gt_bboxes_dict, dict) else len(bbox_list)
    result = {
        'sample_id': data_item['id'],
        'image_path': image_path,
        'user_text': messages[0]['content'][1]['text'],
        'model_output': outputs,
        'ground_truth_bboxes': gt_bboxes_dict,
        'predicted_bboxes': predicted_bboxes,
        'num_gt_bboxes': num_gt_bboxes,
        'num_predicted_bboxes': num_pred_bboxes
    }
    
    result_json_path = os.path.join(args.output_dir, f"result_{data_item['id']}.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {result_json_path}")
    print("\nInference completed!")


if __name__ == '__main__':
    main()

