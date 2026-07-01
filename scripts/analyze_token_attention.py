#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 VLM 中 visual/path token 的注意力分布（基于真实序列位置）。

实现要点：
1) 不使用全局奇偶位猜测 token 类型；
2) 先通过模型的 encode_images/encode_regions 重建每个样本的 region token 位置；
3) 在 output_attentions=True 的输出上聚合 visual/path key 的平均注意力；
4) 输出配对统计与可视化图。
"""

import os
import sys
import json
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX


@dataclass
class SampleTokenTypeMask:
    # -1: non-region, 0: visual token, 1: path token
    token_types: torch.Tensor
    valid_len: int


class CervicalDataCollator:
    """与训练脚本一致的轻量版 collator。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        for ids, lbls in zip(input_ids, labels):
            pad_length = max_length - len(ids)
            padded_ids = torch.cat([ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
            padded_lbls = torch.cat([lbls, torch.full((pad_length,), IGNORE_INDEX)])
            attn = torch.ones(len(ids), dtype=torch.long)
            attn = torch.cat([attn, torch.zeros(pad_length, dtype=torch.long)])
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_masks.append(attn)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["labels"] = torch.stack(padded_labels)
        batch["attention_mask"] = torch.stack(attention_masks)

        batch["images"] = torch.stack([f["images"] for f in features])
        batch["image_grid_thws"] = [f["image_grid_thws"] for f in features]

        if "images_aux" in features[0] and features[0]["images_aux"] is not None:
            batch["images_aux"] = torch.stack([f["images_aux"] for f in features])

        if "bbox_list" in features[0] and features[0]["bbox_list"] is not None:
            bbox_lists = []
            for f in features:
                bbox_lists.append(f["bbox_list"] if f["bbox_list"] is not None else torch.tensor([], dtype=torch.float32))
            batch["bbox_list"] = bbox_lists

        if any(("path_features" in f and f["path_features"] is not None) for f in features):
            batch["path_features"] = [f.get("path_features", None) for f in features]

        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attention distribution between visual/path region tokens.")
    parser.add_argument("--config", type=str, required=True, help="Path to train/eval yaml config.")
    parser.add_argument("--dataset_json", type=str, default=None, help="Override dataset json path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA path to load.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Analysis batch size (recommend 1)")
    parser.add_argument("--max_batches", type=int, default=20, help="Max analyzed batches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/token_attention_analysis")
    parser.add_argument("--plot_name", type=str, default="visual_vs_path_attention_boxplot.png")
    parser.add_argument("--report_name", type=str, default="attention_report.json")
    return parser.parse_args()


def get_model_for_encode(model):
    # PeftModel 通常会把未知属性委托给 base model；这里做兜底。
    if hasattr(model, "encode_images") and hasattr(model, "encode_regions"):
        return model
    if hasattr(model, "base_model"):
        cand = model.base_model
        if hasattr(cand, "encode_images") and hasattr(cand, "encode_regions"):
            return cand
        if hasattr(cand, "model") and hasattr(cand.model, "encode_images") and hasattr(cand.model, "encode_regions"):
            return cand.model
    raise RuntimeError("Cannot locate model object that provides encode_images/encode_regions.")


def load_model_and_data(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_path = cfg["model"]["base_model"]
    tokenizer, model, image_processors = load_pretrained_model(
        base_model_path,
        load_8bit=False,
        load_4bit=False,
        device=args.device,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.lora_path:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for --lora_path")
        model = PeftModel.from_pretrained(model, args.lora_path)

    data_cfg = cfg.get("data", {})
    dataset_json = args.dataset_json or data_cfg.get("test_json") or data_cfg.get("train_json")
    if dataset_json is None:
        raise ValueError("No dataset json found. Please provide --dataset_json or config.data.test_json/train_json")

    dataset = CervicalDataset(
        json_path=dataset_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=base_model_path,
        max_regions=data_cfg.get("max_regions", 100),
        image_size=data_cfg.get("image_size", None),
        is_train=False,
        verbose=False,
        path_feature_root=data_cfg.get("path_feature_root", None),
        language=data_cfg.get("language", "zh"),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CervicalDataCollator(tokenizer),
    )
    return cfg, tokenizer, model, loader


def build_region_token_type_masks(model, batch, device: str) -> List[SampleTokenTypeMask]:
    """
    在不改模型代码的前提下，重建每个样本在“多模态展开后序列”中的 region token 类型位置。
    """
    mm_model = get_model_for_encode(model)
    vision_tower_aux = mm_model.get_vision_tower_aux()
    vision_tower = mm_model.get_vision_tower()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    images = batch["images"].to(device)
    image_grid_thws = batch["image_grid_thws"]
    bbox_list = batch.get("bbox_list", None)
    path_features = batch.get("path_features", None)

    if vision_tower_aux is None or bbox_list is None:
        return [SampleTokenTypeMask(token_types=torch.full((1,), -1, dtype=torch.long), valid_len=1) for _ in range(input_ids.shape[0])]

    image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3 or img.ndim == 2]
    images_minibatch = [images[idx] for idx in image_idx] if len(image_idx) > 0 else []
    images_minibatch_aux = [batch["images_aux"][idx].unsqueeze(0).to(device) for idx in image_idx] if "images_aux" in batch else []

    with torch.no_grad():
        image_features_minibatch, image_grid_thws_minibatch, vt_multi_level_features_minibatch = mm_model.encode_images(
            images_minibatch, image_grid_thws
        )
        patch_size = mm_model.get_model().get_vision_tower().config.patch_size if hasattr(mm_model.get_model().get_vision_tower(), "config") else 14
        vt_images_size_minibatch = [im_grid_thw[0][-2:] * patch_size for im_grid_thw in image_grid_thws]
        region_features = mm_model.encode_regions(
            images_minibatch_aux,
            bbox_list,
            vt_multi_level_features_minibatch if vision_tower is not None else None,
            vt_images_size_minibatch if vision_tower is not None else None,
            path_features=path_features,
        )

    masks: List[SampleTokenTypeMask] = []
    tokenizer_model_max_length = getattr(mm_model.config, "tokenizer_model_max_length", None)

    cur_image_idx = 0
    for b in range(input_ids.shape[0]):
        cur_ids = input_ids[b][attention_mask[b].bool()].tolist()
        mask_list: List[int] = []
        cur_region_idx = 0

        for tok in cur_ids:
            if tok == IMAGE_TOKEN_INDEX:
                if cur_image_idx < len(image_features_minibatch):
                    img_len = int(image_features_minibatch[cur_image_idx].shape[0])
                else:
                    img_len = 0
                mask_list.extend([-1] * img_len)
                cur_image_idx += 1
            elif tok == DEFAULT_REGION_INDEX:
                if b < len(region_features) and cur_region_idx < region_features[b].shape[0]:
                    t = int(region_features[b][cur_region_idx].shape[0])  # T=1 or 2
                    if t == 2:
                        mask_list.extend([0, 1])  # visual, path
                    elif t == 1:
                        mask_list.extend([0])
                    else:
                        mask_list.extend([-1] * t)
                cur_region_idx += 1
            else:
                mask_list.append(-1)

        if tokenizer_model_max_length is not None:
            mask_list = mask_list[:tokenizer_model_max_length]

        if len(mask_list) == 0:
            mask_list = [-1]

        masks.append(SampleTokenTypeMask(token_types=torch.tensor(mask_list, dtype=torch.long), valid_len=len(mask_list)))

    return masks


def run_analysis(model, loader, device: str, max_batches: int):
    model.eval()
    visual_scores = []
    path_scores = []
    paired_scores = []  # (visual, path, layer_idx)
    analyzed_batches = 0
    skipped_samples = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        with torch.no_grad():
            masks = build_region_token_type_masks(model, batch, device=device)

            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "images": batch["images"].to(device),
                "image_grid_thws": batch["image_grid_thws"],
                "output_attentions": True,
                "return_dict": True,
            }
            if "images_aux" in batch:
                model_inputs["images_aux"] = batch["images_aux"].to(device)
            if "bbox_list" in batch:
                model_inputs["bbox_list"] = batch["bbox_list"]
            if "path_features" in batch:
                model_inputs["path_features"] = batch["path_features"]

            outputs = model(**model_inputs)
            attentions = outputs.attentions  # tuple[L] each: [B, H, Q, K]

        for b in range(len(masks)):
            tok_types = masks[b].token_types
            if tok_types.numel() <= 1:
                skipped_samples += 1
                continue

            visual_idx = (tok_types == 0).nonzero(as_tuple=False).flatten()
            path_idx = (tok_types == 1).nonzero(as_tuple=False).flatten()
            if visual_idx.numel() == 0 or path_idx.numel() == 0:
                skipped_samples += 1
                continue

            # 与 attention 的实际 K 维度对齐
            k_len = attentions[0][b].shape[-1]
            visual_idx = visual_idx[visual_idx < k_len]
            path_idx = path_idx[path_idx < k_len]
            if visual_idx.numel() == 0 or path_idx.numel() == 0:
                skipped_samples += 1
                continue

            for lid, attn_l in enumerate(attentions):
                # [H, Q, K] -> scalar
                attn_b = attn_l[b]
                v_score = float(attn_b[:, :, visual_idx].mean().item())
                p_score = float(attn_b[:, :, path_idx].mean().item())
                visual_scores.append(v_score)
                path_scores.append(p_score)
                paired_scores.append((v_score, p_score, lid))

        analyzed_batches += 1

    return {
        "visual_scores": visual_scores,
        "path_scores": path_scores,
        "paired_scores": paired_scores,
        "analyzed_batches": analyzed_batches,
        "skipped_samples": skipped_samples,
    }


def summarize_and_save(result: Dict, output_dir: str, plot_name: str, report_name: str):
    os.makedirs(output_dir, exist_ok=True)
    visual = np.array(result["visual_scores"], dtype=np.float32)
    path = np.array(result["path_scores"], dtype=np.float32)

    summary = {
        "num_points": int(len(visual)),
        "analyzed_batches": int(result["analyzed_batches"]),
        "skipped_samples": int(result["skipped_samples"]),
        "visual_mean": float(np.mean(visual)) if len(visual) > 0 else None,
        "path_mean": float(np.mean(path)) if len(path) > 0 else None,
        "visual_std": float(np.std(visual)) if len(visual) > 0 else None,
        "path_std": float(np.std(path)) if len(path) > 0 else None,
        "paired_ttest_pvalue": None,
        "paired_ttest_tstat": None,
    }

    if SCIPY_AVAILABLE and len(visual) > 1 and len(path) > 1 and len(visual) == len(path):
        t_stat, p_value = stats.ttest_rel(visual, path)
        summary["paired_ttest_pvalue"] = float(p_value)
        summary["paired_ttest_tstat"] = float(t_stat)

    if MATPLOTLIB_AVAILABLE and len(visual) > 0 and len(path) > 0:
        fig_path = os.path.join(output_dir, plot_name)
        plt.figure(figsize=(6, 4))
        plt.boxplot([visual, path], labels=["Visual Tokens", "Path Tokens"])
        plt.ylabel("Average Attention Weight (as keys)")
        plt.title("Attention Distribution: Visual vs Path Tokens")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        summary["plot_path"] = fig_path

    report_path = os.path.join(output_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, _, model, loader = load_model_and_data(args)
    result = run_analysis(model, loader, device=args.device, max_batches=args.max_batches)
    summarize_and_save(result, args.output_dir, args.plot_name, args.report_name)


if __name__ == "__main__":
    main()

