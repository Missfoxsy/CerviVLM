#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞定位第二阶段多任务推理脚本

支持五种任务类型的推理（通过task_id字段识别）：
1. 类型0 (task_id=0)：基础定位(OD) - 框出图中的阳性宫颈细胞位置
2. 类型1 (task_id=1)：带描述的特征定位（REC）- 找出图中特定特征的细胞
3. 类型2 (task_id=2)：区域描述(Region Caption) - 描述图中区域的特征
4. 类型3 (task_id=3)：细胞类型分类(Classification) - 判断细胞类型
5. 类型4 (task_id=4)：来源分类(Classification) - 判断细胞来源
"""

import torch
import os
import json
import argparse
import re
import random
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
    load_image,
    make_message_context,
)
from vlm_fo1.constants import IGNORE_INDEX
from utils.data_utils import parse_grounding_output

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


def get_task_type_from_id(task_id: int) -> str:
    """
    从task_id获取任务类型
    
    Args:
        task_id: 任务ID (0-4)
        
    Returns:
        任务类型: 'type0', 'type1', 'type2', 'type3', 'type4'
    """
    if task_id is None:
        return 'unknown'
    if 0 <= task_id <= 4:
        return f'type{task_id}'
    return 'unknown'


def normalize_and_save_feature_maps(
    features,
    output_dir: str,
    prefix: str,
    sample_id: str,
    base_image: Image.Image = None,
):
    """
    将多尺度特征图保存为灰度 PNG，方便可视化。
    
    Args:
        features: List[Tensor] 或 Tensor 或 dict（包含键 image_features）
                  形状通常为 [(B, C, H, W), ...]
        output_dir: 保存目录（会在其中创建 features 子目录）
        prefix: 文件名前缀（如 'davit' 或 'fpn'）
        sample_id: 当前样本 id，用于区分文件
    """
    if features is None:
        return

    # DavitVisionTower.forward 返回的是 List[dict]，这里取第一个样本
    if isinstance(features, (list, tuple)) and len(features) > 0 and isinstance(features[0], dict):
        features = features[0]

    # 支持 dict（如 {"image_features": [...], "last_feat": ...}）
    if isinstance(features, dict):
        if "image_features" in features:
            features = features["image_features"]
        else:
            return

    # 单层特征也统一转为 list
    if not isinstance(features, (list, tuple)):
        features = [features]

    feat_dir = os.path.join(output_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    for level, fmap in enumerate(features):
        if fmap is None:
            continue
        try:
            # 取 batch 中第一个样本 (C, H, W)
            if fmap.dim() == 4:
                fm = fmap[0]
            elif fmap.dim() == 3:
                fm = fmap
            else:
                continue

            # bf16 在保存/转 numpy/PIL 时不被支持，这里统一转为 float32
            fm = fm.detach().to(dtype=torch.float32)

            # 使用通道最大响应得到单通道 (1, H, W)，相比均值更能突出显著区域
            if fm.dim() == 3:
                # 先取绝对值再做通道最大，避免正负抵消
                fm = fm.abs().max(0, keepdim=True)[0]

            # 归一化到 [0, 1]
            fm_min = fm.min()
            fm_max = fm.max()
            fm = (fm - fm_min) / (fm_max - fm_min + 1e-6)

            # 如果没有提供原图，只保存归一化灰度特征图
            if base_image is None:
                save_path = os.path.join(
                    feat_dir,
                    f"{prefix}_l{level}_{sample_id}.png"
                )
                save_image(fm, save_path)
                print(f"[Feature] Saved {prefix} level {level} feature map to: {save_path}")
            else:
                # 将特征图 resize 到与原图相同尺寸并叠加为半透明热力图（hot 颜色映射）
                base_w, base_h = base_image.size
                fm_np = fm.squeeze(0).cpu().numpy()  # (H, W)
                fm_img = Image.fromarray(
                    np.uint8(np.clip(fm_np, 0.0, 1.0) * 255), mode="L"
                )
                fm_img = fm_img.resize((base_w, base_h), Image.BILINEAR)

                # 使用简单 hot 风格颜色映射：黑 -> 红 -> 黄 -> 白
                fm_arr = np.array(fm_img, dtype=np.float32) / 255.0  # [0,1]
                r = np.clip(3 * fm_arr, 0, 1)
                g = np.clip(3 * fm_arr - 1, 0, 1)
                b = np.clip(3 * fm_arr - 2, 0, 1)
                heat_rgb_arr = np.stack(
                    [r * 255.0, g * 255.0, b * 255.0], axis=-1
                ).astype(np.uint8)
                heat_rgb = Image.fromarray(heat_rgb_arr, mode="RGB").convert("RGBA")
                # 设置透明度（0-255）
                alpha = 128
                heat_rgb.putalpha(alpha)

                base_rgba = base_image.convert("RGBA")
                overlay = Image.alpha_composite(base_rgba, heat_rgb)

                save_path = os.path.join(
                    feat_dir,
                    f"{prefix}_l{level}_{sample_id}_overlay.png"
                )
                overlay.save(save_path)
                print(f"[Feature] Saved {prefix} level {level} overlay heatmap to: {save_path}")
        except Exception as e:
            print(f"Warning: failed to save {prefix} feature level {level}: {e}")


def normalize_and_save_feature_maps_roi(
    features,
    output_dir: str,
    prefix: str,
    sample_id: str,
    base_image: Image.Image,
    bbox_list,
    target_level: int = 0,
):
    """
    仅在给定 ROI（bbox）内可视化某一层特征图的响应，并叠加在原图上。
    
    Args:
        features: 同 normalize_and_save_feature_maps，支持 List[Tensor] 或 Davit 的 List[dict]/dict 结构
        output_dir: 输出目录
        prefix: 文件名前缀
        sample_id: 样本 id
        base_image: 原始 PIL 图像（RGB）
        bbox_list: 当前样本的 bbox 列表（原图坐标系）
        target_level: 需要可视化的特征层索引（如 0/1/2/3）
    """
    if features is None or base_image is None or not bbox_list:
        return

    # DavitVisionTower.forward 返回的是 List[dict]，这里取第一个样本
    if isinstance(features, (list, tuple)) and len(features) > 0 and isinstance(features[0], dict):
        features = features[0]

    if isinstance(features, dict):
        if "image_features" in features:
            features = features["image_features"]
        else:
            return

    if not isinstance(features, (list, tuple)):
        features = [features]

    if target_level < 0 or target_level >= len(features):
        print(f"Warning: target_level {target_level} out of range for prefix {prefix}, total levels {len(features)}")
        return

    fmap = features[target_level]
    if fmap is None:
        return

    try:
        # 取 batch 中第一个样本 (C, H, W)
        if fmap.dim() == 4:
            fm = fmap[0]
        elif fmap.dim() == 3:
            fm = fmap
        else:
            return

        fm = fm.detach().to(dtype=torch.float32)

        # 通道最大响应 + 归一化
        if fm.dim() == 3:
            fm = fm.abs().max(0, keepdim=True)[0]
        fm_min = fm.min()
        fm_max = fm.max()
        fm = (fm - fm_min) / (fm_max - fm_min + 1e-6)

        # resize 到原图大小
        base_w, base_h = base_image.size
        fm_np = fm.squeeze(0).cpu().numpy()  # (H, W)
        fm_img = Image.fromarray(
            np.uint8(np.clip(fm_np, 0.0, 1.0) * 255), mode="L"
        )
        fm_img = fm_img.resize((base_w, base_h), Image.BILINEAR)

        # 构造 ROI 掩膜（binary mask）
        mask = np.zeros((base_h, base_w), dtype=np.float32)
        for bbox in bbox_list:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(base_w, int(x1)))
            y1 = max(0, min(base_h, int(y1)))
            x2 = max(0, min(base_w, int(x2)))
            y2 = max(0, min(base_h, int(y2)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0

        fm_arr = np.array(fm_img, dtype=np.float32) / 255.0
        fm_arr = fm_arr * mask  # 只保留 bbox 区域内响应

        # hot 颜色映射
        r = np.clip(3 * fm_arr, 0, 1)
        g = np.clip(3 * fm_arr - 1, 0, 1)
        b = np.clip(3 * fm_arr - 2, 0, 1)
        heat_rgb_arr = np.stack(
            [r * 255.0, g * 255.0, b * 255.0], axis=-1
        ).astype(np.uint8)
        heat_rgb = Image.fromarray(heat_rgb_arr, mode="RGB").convert("RGBA")
        heat_rgb.putalpha(128)

        base_rgba = base_image.convert("RGBA")
        overlay = Image.alpha_composite(base_rgba, heat_rgb)

        feat_dir = os.path.join(output_dir, "features")
        os.makedirs(feat_dir, exist_ok=True)
        save_path = os.path.join(
            feat_dir,
            f"{prefix}_l{target_level}_{sample_id}_roi_overlay.png"
        )
        overlay.save(save_path)
        print(f"[Feature] Saved ROI-only {prefix} level {target_level} overlay heatmap to: {save_path}")
    except Exception as e:
        print(f"Warning: failed to save ROI feature heatmap for {prefix} level {target_level}: {e}")


def load_checkpoint_model(base_model_path, checkpoint_path, device="cuda", merge_lora=False):
    """
    加载checkpoint模型（支持LoRA权重）
    
    Args:
        base_model_path: 基础模型路径
        checkpoint_path: checkpoint路径（包含LoRA权重）
        device: 设备
        merge_lora: 是否合并LoRA权重
        
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
            if merge_lora:
                print("Merging LoRA weights into base model...")
                model = model.merge_and_unload()
                print("LoRA weights merged")
            else:
                print("LoRA weights loaded successfully")
        else:
            print("Warning: LoRA weights found but PEFT is not available. Using base model only.")
    else:
        # 尝试从checkpoint目录加载完整模型
        if os.path.exists(os.path.join(checkpoint_path, "config.json")):
            print(f"Loading full model from checkpoint: {checkpoint_path}")
            tokenizer, model, image_processors = load_pretrained_model(
                checkpoint_path,
                load_8bit=False,
                load_4bit=False,
                device=device
            )
        else:
            print("No LoRA weights or full model found, using base model only")
    
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
    
    # ------------------------------------------------------------------
    # 为不同任务增加“输出格式约束”，与训练/评估阶段保持一致，降低多任务时的任务混淆
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

    # 构建边界框列表（对于类型3和类型4，也需要bbox_list，因为需要region引用）
    bbox_list = []
    for region in data_item.get('regions', []):
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


def find_samples_by_task_type(test_data, task_type=None):
    """
    根据任务类型查找样本
    
    Args:
        test_data: 测试数据列表
        task_type: 任务类型 ('type0', 'type1', 'type2', 'type3', 'type4')，None表示所有类型
        
    Returns:
        匹配的样本索引列表
    """
    matched_indices = []
    for idx, item in enumerate(test_data):
        task_id = item.get('task_id')
        item_task_type = get_task_type_from_id(task_id)
        if task_type is None or item_task_type == task_type:
            matched_indices.append(idx)
    
    return matched_indices


def main():
    parser = argparse.ArgumentParser(description='Multi-task inference with trained VLM-FO1 Stage 2 model')
    parser.add_argument(
        '--base_model',
        type=str,
        default='resources/VLM-FO1_Qwen2.5-VL-3B-v01',
        help='Path to base model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/cervical_multitask',
        help='Path to checkpoint directory (default: outputs/cervical_finetune_stage2)'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        default='dataset/data20260115/stage2_val.json',
        help='Path to test JSON file'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=None,
        help='Index of sample to use for inference (if None, will select by task_type)'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        choices=['type0', 'type1', 'type2', 'type3', 'type4', 'all'],
        default='type0',
        help='Task type to test (type0: OD, type1: REC, type2: Caption, type3: Cell Type Classification, type4: Source Classification, all: test all types)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of samples to test (only used when task_type is not None)'
    )
    parser.add_argument(
        '--save_features',
        action='store_true',
        help='Whether to save DaViT and FPN multi-scale feature maps for each sample'
    )
    parser.add_argument(
        '--overlay_features',
        action='store_true',
        help='If set, overlay feature heatmaps on the original image; otherwise save standalone feature maps'
    )
    parser.add_argument(
        '--davit_level',
        type=int,
        default=2,
        help='Which DaViT feature level to visualize on ROI (0-based index)'
    )
    parser.add_argument(
        '--fpn_level',
        type=int,
        default=1,
        help='Which FPN feature level to visualize on ROI (0-based index)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='inference_results/cervical_0115',
        help='Output directory for visualization'
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
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/sample1000/path_features",
        help='Optional: directory of precomputed pathology features (*.pt). If set, pathology tokens will be injected during inference.'
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
    if not os.path.exists(args.test_json):
        print(f"Error: Test file not found: {args.test_json}")
        return
    
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Total test samples: {len(test_data)}")
    
    # 选择样本
    if args.sample_idx is not None:
        # 直接使用指定的索引
        if args.sample_idx >= len(test_data):
            print(f"Error: sample_idx {args.sample_idx} is out of range. Total samples: {len(test_data)}")
            return
        sample_indices = [args.sample_idx]
    else:
        # 根据任务类型选择样本
        if args.task_type == 'all':
            # 测试所有类型，每种类型选择num_samples个样本
            sample_indices = []
            for task_type in ['type0', 'type1', 'type2', 'type3', 'type4']:
                matched_indices = find_samples_by_task_type(test_data, task_type)
                if len(matched_indices) > 0:
                    # 随机选择指定数量的样本
                    random.shuffle(matched_indices)
                    num_to_select = min(args.num_samples, len(matched_indices))
                    sample_indices.extend(matched_indices[:num_to_select])
        else:
            # 测试指定类型的样本
            matched_indices = find_samples_by_task_type(test_data, args.task_type)
            if len(matched_indices) == 0:
                print(f"Error: No samples found for task type: {args.task_type}")
                return
            # 随机选择指定数量的样本
            random.shuffle(matched_indices)
            sample_indices = matched_indices[:args.num_samples]
    
    print(f"\nSelected {len(sample_indices)} sample(s) for inference")
    
    # 加载checkpoint模型
    print(f"\nLoading checkpoint model from: {args.checkpoint}")
    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        merge_lora=args.merge_lora
    )

    # ------------------------------------------------------------------
    # 可选：注册 DaViT 和 FPN 的 forward hook，用于抓取多尺度特征
    # ------------------------------------------------------------------
    davit_features = {}
    fpn_features = {}
    davit_handle = None
    fpn_handle = None

    if args.save_features:
        # DaViT 辅助视觉塔特征（多尺度）
        try:
            if hasattr(model, "get_vision_tower_aux"):
                vision_tower_aux = model.get_vision_tower_aux()
            else:
                vision_tower_aux = None

            # 直接在辅助视觉塔 (DavitVisionTower) 上挂 hook，
            # encode_regions 中会调用 self.get_model().get_vision_tower_aux()(images)
            if vision_tower_aux is not None:
                def _davit_hook(module, inputs, output):
                    """
                    output: List[dict]，每个元素类似
                        {
                            "image_features": [f1, f2, f3, f4],
                            "last_feat": f4
                        }
                    """
                    davit_features.clear()
                    davit_features["output"] = output

                davit_handle = vision_tower_aux.register_forward_hook(_davit_hook)
                print("[Hook] Registered DaViT (aux vision tower) feature hook.")
            else:
                print("[Hook] Auxiliary vision tower not found; skip DaViT hook.")
        except Exception as e:
            print(f"Warning: failed to register DaViT hook: {e}")
            davit_handle = None

        # FPN（SimpleFP）在 Hybrid Finegrained Region Encoder 中
        try:
            core_model = model.get_model() if hasattr(model, "get_model") else None
            object_vp_extractor = getattr(core_model, "object_vp_extractor", None)
            simple_fpn = getattr(object_vp_extractor, "simple_fpn", None) if object_vp_extractor is not None else None

            if simple_fpn is not None:
                def _fpn_hook(module, inputs, output):
                    # output: List[Tensor]，多尺度 FPN 特征
                    fpn_features.clear()
                    fpn_features["output"] = output

                fpn_handle = simple_fpn.register_forward_hook(_fpn_hook)
                print("[Hook] Registered SimpleFPN feature hook.")
            else:
                print("[Hook] SimpleFPN module not found on object_vp_extractor; skip FPN hook.")
        except Exception as e:
            print(f"Warning: failed to register FPN hook: {e}")
            fpn_handle = None
    
    # 对每个样本进行推理
    all_results = []
    
    for sample_idx in sample_indices:
        # 每个样本前清空上一次保存的特征
        davit_features.clear()
        fpn_features.clear()

        data_item = test_data[sample_idx]
        
        # 从task_id获取任务类型
        task_id = data_item.get('task_id')
        task_type = get_task_type_from_id(task_id)
        
        # 提取用户和助手消息用于显示
        user_text = ''
        assistant_text = ''
        for conv in data_item['conversations']:
            if conv['from'] == 'user':
                user_text = conv['value']
            elif conv['from'] == 'assistant':
                assistant_text = conv['value']
        
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx}: {data_item['id']}")
        print(f"Task Type: {task_type}")
        print(f"Image path: {data_item['image']}")
        print(f"User text: {user_text[:100]}...")
        print(f"{'='*60}")
        
        # 检查图像文件是否存在
        if not os.path.exists(data_item['image']):
            print(f"Error: Image file not found: {data_item['image']}")
            continue
        
        # 准备消息和边界框
        messages, bbox_list = prepare_messages_from_data(data_item)
        
        print(f"Number of regions: {len(bbox_list)}")
        
        # 准备模型输入
        generation_kwargs = prepare_inputs(
            args.base_model,
            model,
            image_processors,
            tokenizer,
            messages,
            max_tokens=512,
            top_p=0.05,
            temperature=0.0,
            do_sample=False,
            # 推理阶段同样传入病理特征根目录，使行为与训练/评估阶段一致
            path_feature_root=args.path_feature_root,
        )
        
        # 运行推理
        print("Running inference...")
        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)
            outputs = tokenizer.decode(
                output_ids[0, generation_kwargs['inputs'].shape[1]:],
                skip_special_tokens=False
            ).strip()
        
        print(f"\nModel output: {outputs}")
        print(f"Ground truth: {assistant_text}")
        
        # 加载原始图像（叠加模式时才需要）
        image_path = data_item['image']
        img_pil = Image.open(image_path).convert("RGB")
        
        # ------------------------------------------------------------------
        # 可视化 DaViT / FPN 多尺度特征图
        # - 若 overlay_features=True：叠加在原图上，并额外输出 ROI 版本
        # - 若 overlay_features=False：仅输出纯特征图（不叠加原图）
        # ------------------------------------------------------------------
        if args.save_features:
            try:
                davit_out = davit_features.get("output", None)
                if davit_out is not None:
                    # 全图可视化
                    normalize_and_save_feature_maps(
                        davit_out,
                        args.output_dir,
                        prefix=f"davit_{task_type}",
                        sample_id=str(data_item['id']),
                        base_image=img_pil if args.overlay_features else None,
                    )
                    # ROI 只在叠加模式下有意义
                    if args.overlay_features:
                        normalize_and_save_feature_maps_roi(
                            davit_out,
                            args.output_dir,
                            prefix=f"davit_{task_type}",
                            sample_id=str(data_item['id']),
                            base_image=img_pil,
                            bbox_list=bbox_list,
                            target_level=args.davit_level,
                        )
                else:
                    print("[Feature] No DaViT features captured for this sample.")
            except Exception as e:
                print(f"Warning: failed to visualize DaViT features: {e}")

            try:
                fpn_out = fpn_features.get("output", None)
                if fpn_out is not None:
                    normalize_and_save_feature_maps(
                        fpn_out,
                        args.output_dir,
                        prefix=f"fpn_{task_type}",
                        sample_id=str(data_item['id']),
                        base_image=img_pil if args.overlay_features else None,
                    )
                    if args.overlay_features:
                        normalize_and_save_feature_maps_roi(
                            fpn_out,
                            args.output_dir,
                            prefix=f"fpn_{task_type}",
                            sample_id=str(data_item['id']),
                            base_image=img_pil,
                            bbox_list=bbox_list,
                            target_level=args.fpn_level,
                        )
                else:
                    print("[Feature] No FPN features captured for this sample.")
            except Exception as e:
                print(f"Warning: failed to visualize FPN features: {e}")
        
        # 根据任务类型进行不同的处理
        result = {
            'sample_id': data_item['id'],
            'sample_idx': sample_idx,
            'task_type': task_type,
            'image_path': image_path,
            'user_text': user_text,
            'model_output': outputs,
            'ground_truth': assistant_text
        }
        
        if task_type in ['type0', 'type1']:
            # 定位任务：提取bbox并可视化
            print("\nExtracting bounding boxes...")
            # 修复outputs中的<object>标签
            outputs_fixed = outputs.replace('<object>', '<objects>').replace('</object>', '</objects>')
            predicted_bboxes = extract_predictions_to_bboxes(outputs_fixed, bbox_list)
            # 将标签转换为英文（避免中文乱码）
            predicted_bboxes = {normalize_label_to_en(k): v for k, v in predicted_bboxes.items()}
            
            print(f"Predicted bounding boxes (dict with {len(predicted_bboxes)} labels):")
            for label, bboxes in predicted_bboxes.items():
                print(f"  Label '{label}': {len(bboxes)} bboxes")
            
            # 可视化预测结果
            output_image_path = os.path.join(args.output_dir, f"pred_{task_type}_{data_item['id']}.jpg")
            if len(predicted_bboxes) > 0:
                draw_bboxes_and_save(
                    image=img_pil.copy(),
                    fo1_bboxes=predicted_bboxes,
                    output_path=output_image_path,
                    color='red'
                )
                print(f"Prediction visualization saved to: {output_image_path}")
            
            # 可视化ground truth
            gt_image_path = os.path.join(args.output_dir, f"gt_{task_type}_{data_item['id']}.jpg")
            gt_label = "positive cervical cell"
            if assistant_text:
                ground_match = re.search(r'<ground>(.*?)</ground>', assistant_text)
                if ground_match:
                    gt_label = ground_match.group(1).strip()
            
            # 从GT中提取bbox
            gt_dict = parse_grounding_output(assistant_text)
            gt_label = normalize_label_to_en(gt_label)
            
            region_id_to_bbox = {}
            for region in data_item.get('regions', []):
                region_id = region.get('region_id', '')
                if region_id:
                    match = re.match(r'region(\d+)', region_id)
                    if match:
                        region_idx = int(match.group(1))
                        region_id_to_bbox[region_idx] = region['bbox']
            
            # 构建GT bboxes字典（支持多标签，并将标签转换为英文）
            gt_bboxes_dict = {}
            if gt_dict:
                # 从gt_dict中按标签分组构建bboxes
                for label, indices in gt_dict.items():
                    label_en = normalize_label_to_en(label)
                    if label_en not in gt_bboxes_dict:
                        gt_bboxes_dict[label_en] = []
                    for idx in indices:
                        if idx in region_id_to_bbox:
                            gt_bboxes_dict[label_en].append(region_id_to_bbox[idx])
            else:
                # 如果gt_dict为空，使用gt_label作为默认标签，包含所有regions
                all_gt_bboxes = [region['bbox'] for region in data_item.get('regions', []) if len(region.get('bbox', [])) == 4]
                if all_gt_bboxes:
                    gt_bboxes_dict = {gt_label: all_gt_bboxes}
            
            if len(gt_bboxes_dict) > 0:
                draw_bboxes_and_save(
                    image=img_pil.copy(),
                    fo1_bboxes=gt_bboxes_dict,
                    output_path=gt_image_path,
                    color='green'
                )
                print(f"Ground truth visualization saved to: {gt_image_path}")
            
            result['predicted_bboxes'] = predicted_bboxes
            result['ground_truth_bboxes'] = gt_bboxes_dict
            
        else:
            # 类型2、3、4：文本生成任务（区域描述或分类），不需要bbox可视化
            print("\nText generation task - no bbox visualization needed")
            result['is_text_task'] = True
        
        all_results.append(result)
        
        # 保存单个样本的结果
        result_json_path = os.path.join(args.output_dir, f"result_{task_type}_{data_item['id']}.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Result saved to: {result_json_path}")
    
    # 移除 hook，避免潜在的内存泄露
    if davit_handle is not None:
        davit_handle.remove()
    if fpn_handle is not None:
        fpn_handle.remove()

    # 保存所有结果汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(all_results),
            'task_type': args.task_type,
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Total samples tested: {len(all_results)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

