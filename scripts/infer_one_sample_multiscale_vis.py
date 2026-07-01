#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单样本推理，并保存 VLM-FO1 编码阶段多尺度特征可视化。

模型加载与 prepare_inputs 对齐 scripts/eval_stage2_cot.py。
编码阶段特征来源：
  - vt_raw / vt_fpn / aux：全图多尺度特征图
  - vt_roi / aux_roi：ROIAlign 后（池化前 14×14）与池化后的 region 向量

可视化方式：特征图用通道 L2 范数；池化向量用 |x| 归一化后展成热力条带。
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import roi_align

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from eval_stage2_cot import (  # noqa: E402
    DEFAULT_VLM_FO1_BASE,
    _is_official_vlm_fo1_model_dir,
    _unwrap_hierarchical_fusion_module,
    add_task_constraints_to_user_text,
    load_checkpoint_model,
    maybe_enable_hierarchical_fusion,
)
from vlm_fo1.mm_utils import prepare_inputs  # noqa: E402
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import (  # noqa: E402
    visualize_hierarchical_attention,
)


DEFAULT_DATA_JSON = (
    "/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_train.json"
)


def _load_json_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"期望非空 JSON 数组: {path}")
    return data


def _find_sample(data: list, sample_id: str = None, sample_index: int = 0) -> tuple:
    """按 id 或 index 选取样本，返回 (item, index)。"""
    if sample_id is not None:
        matches = [(i, x) for i, x in enumerate(data) if str(x.get("id")) == str(sample_id)]
        if not matches:
            raise ValueError(f"未找到 id={sample_id!r}，数据集共 {len(data)} 条")
        if len(matches) > 1:
            idxs = [m[0] for m in matches]
            print(
                f"Warning: id={sample_id!r} 在数据集中出现 {len(matches)} 次，"
                f"使用首次出现 index={idxs[0]}（其余 index={idxs[1:]}）"
            )
        return matches[0][1], matches[0][0]
    if sample_index < 0 or sample_index >= len(data):
        raise IndexError(f"sample_index={sample_index} 超出范围 [0, {len(data) - 1}]")
    return data[sample_index], sample_index


def _extract_dialog_fields(item: dict):
    image_path = item.get("image")
    user_text = None
    assistant_text = None
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            user_text = conv.get("value", "").replace("<image>", "").strip()
            if user_text.startswith("\n"):
                user_text = user_text[1:]
        elif conv.get("from") == "assistant":
            assistant_text = conv.get("value", "")
    regions = item.get("regions", [])
    bbox_list = []
    region_id_to_index = {}
    for idx, region in enumerate(regions):
        bbox = region.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_list.append([float(v) for v in bbox])
            region_id = region.get("region_id", f"region{idx}")
            region_id_to_index[region_id] = idx
    task_id = item.get("task_id")
    return image_path, user_text, assistant_text, bbox_list, task_id, regions, region_id_to_index


def _feature_to_heatmap(feat: torch.Tensor) -> np.ndarray:
    """[B,C,H,W] 或 [C,H,W] → 归一化热图 [H,W]。"""
    if feat.dim() == 4:
        feat = feat[0]
    if feat.dim() != 3:
        raise ValueError(f"期望 3D/4D 特征，得到 shape={tuple(feat.shape)}")
    arr = feat.detach().float().cpu().numpy()
    hm = np.linalg.norm(arr, axis=0)
    hm = np.maximum(hm, 0.0)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    return hm.astype(np.float32)


def _resize_heatmap(hm: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    hm_img = Image.fromarray((hm * 255.0).astype(np.uint8), mode="L")
    hm_img = hm_img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(hm_img, dtype=np.float32) / 255.0


def _overlay_heatmap(base_rgb: Image.Image, hm: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = base_rgb.convert("RGB")
    w, h = base.size
    hm = _resize_heatmap(hm, w, h)
    try:
        import matplotlib.cm as cm

        colored = (cm.get_cmap("jet")(hm)[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        colored = np.stack([hm * 255, np.zeros_like(hm), (1.0 - hm) * 255], axis=-1).astype(np.uint8)
    heat_img = Image.fromarray(colored, mode="RGB")
    return Image.blend(base, heat_img, alpha=alpha)


def _draw_bboxes(img: Image.Image, bboxes, region_ids=None, highlight_index=None, width=2) -> Image.Image:
    if not bboxes:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    colors = [(0, 255, 0), (255, 165, 0), (0, 191, 255), (255, 0, 255)]
    for i, bb in enumerate(bboxes):
        if bb is None or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        color = (255, 0, 0) if highlight_index is not None and i == highlight_index else colors[i % len(colors)]
        lw = width + 1 if highlight_index is not None and i == highlight_index else width
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = region_ids[i] if region_ids and i < len(region_ids) else f"region{i}"
        draw.text((x1 + 2, max(y1 - 14, 0)), label, fill=color)
    return out


def _infer_highlight_region_index(user_text: str, region_id_to_index: dict):
    """从 user 文本中的 <regionN> 推断主要关注区域下标（取首个）。"""
    if not user_text:
        return None
    tags = re.findall(r"<region(\d+)>", user_text)
    if not tags:
        return None
    idx = int(tags[0])
    for rid, ridx in region_id_to_index.items():
        if rid == f"region{idx}":
            return ridx
    if 0 <= idx < max(region_id_to_index.values(), default=-1) + 1:
        return idx
    return None


def _save_feature_group(
    features,
    titles,
    group_name: str,
    base_image: Image.Image,
    out_dir: str,
    alpha: float,
):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("需要 matplotlib 才能保存可视化图") from e

    n = len(features)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for i, (feat, title) in enumerate(zip(features, titles)):
        hm = _feature_to_heatmap(feat)
        overlay = _overlay_heatmap(base_image, hm, alpha=alpha)
        row, col = divmod(i, cols)
        ax = axes[row, col]
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        stem = f"{group_name}_L{i + 1}"
        overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
        hm_path = os.path.join(out_dir, f"{stem}_heatmap.png")
        overlay.save(overlay_path)
        Image.fromarray((hm * 255).astype(np.uint8), mode="L").resize(
            base_image.size, resample=Image.BILINEAR
        ).save(hm_path)
        saved.append(
            {
                "level": i + 1,
                "title": title,
                "shape": list(feat.shape),
                "overlay": overlay_path,
                "heatmap": hm_path,
            }
        )

    for j in range(n, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis("off")

    grid_path = os.path.join(out_dir, f"{group_name}_grid.png")
    fig.suptitle(group_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append({"grid": grid_path})
    return saved


def _task_type_from_id(task_id):
    try:
        tid = int(task_id)
    except Exception:
        return "type1"
    if 0 <= tid <= 5:
        return f"type{tid}"
    return "type1"


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr.astype(np.float32), 0.0)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def _roi_spatial_heatmap(roi_feat: torch.Tensor) -> np.ndarray:
    """[C,H,W] → [H,W]。"""
    if roi_feat.dim() != 3:
        raise ValueError(f"期望 [C,H,W]，得到 {tuple(roi_feat.shape)}")
    arr = roi_feat.detach().float().cpu().numpy()
    return _normalize_map(np.linalg.norm(arr, axis=0))


def _vector_heatmap(vec: torch.Tensor, ncol: int = 64) -> np.ndarray:
    """[C] → 2D 热力图（按 |x| 归一化后 reshape）。"""
    v = vec.detach().float().abs().cpu().numpy()
    v = _normalize_map(v)
    c = v.shape[0]
    nrow = int(np.ceil(c / ncol))
    pad = nrow * ncol - c
    if pad > 0:
        v = np.pad(v, (0, pad))
    return v.reshape(nrow, ncol)


def _save_heatmap_image(hm: np.ndarray, path: str, cmap_name: str = "viridis"):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(2.5, hm.shape[1] / 8), max(2.0, hm.shape[0] / 8)))
        ax.imshow(hm, cmap=cmap_name, aspect="auto")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    except ImportError as e:
        raise RuntimeError("需要 matplotlib 才能保存 ROI 可视化") from e


def _crop_image_by_bbox(img: Image.Image, bbox, pad_ratio: float = 0.05) -> Image.Image:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = img.size
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1 = int(max(0, x1 - px))
    cy1 = int(max(0, y1 - py))
    cx2 = int(min(w, x2 + px))
    cy2 = int(min(h, y2 + py))
    return img.crop((cx1, cy1, cx2, cy2))


def _build_aux_concat_features(extractor, aux_multi_level_features):
    """与 HFREModule.__call__ 中 aux 分支一致。"""
    feats = aux_multi_level_features
    if extractor.pos_embedding_strategy in ("feature_map_based", "hybrid"):
        feats = extractor._apply_feature_map_position_embedding(feats)
    max_height = max(f.shape[2] for f in feats)
    max_width = max(f.shape[3] for f in feats)
    parts = []
    for level, feature in enumerate(feats):
        if level != 0:
            parts.append(
                F.interpolate(
                    feature.float(),
                    size=(max_height, max_width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        else:
            parts.append(feature.float())
    return torch.cat(parts, dim=1)


def _format_roi_boxes(boxes_xyxy: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
    """将 [N,4] xyxy 转为 torchvision roi_align 需要的 [N,5] (batch_idx, x1,y1,x2,y2)。"""
    if boxes_xyxy.dim() == 1:
        boxes_xyxy = boxes_xyxy.unsqueeze(0)
    n = boxes_xyxy.shape[0]
    batch_col = torch.full(
        (n, 1),
        float(batch_index),
        device=boxes_xyxy.device,
        dtype=boxes_xyxy.dtype,
    )
    return torch.cat([batch_col, boxes_xyxy], dim=1)


def _compute_vt_boxes(boxes, vt_images_size, aux_image_hw, device):
    current_h, current_w = aux_image_hw
    if isinstance(vt_images_size, torch.Tensor):
        original_h = float(vt_images_size[-2].item())
        original_w = float(vt_images_size[-1].item())
    else:
        original_h = float(vt_images_size[0])
        original_w = float(vt_images_size[1])
    scale_h = original_h / float(current_h)
    scale_w = original_w / float(current_w)
    scale = torch.tensor(
        [scale_w, scale_h, scale_w, scale_h],
        device=device,
        dtype=boxes.dtype,
    )
    return boxes * scale


def capture_roi_align_outputs(model, gen_kwargs, task_id, region_labels=None):
    """提取 VT-FPN / Aux 的 ROIAlign 与池化结果。"""
    core = model.get_model()
    extractor = getattr(core, "object_vp_extractor", None)
    if extractor is None:
        return None

    images = gen_kwargs["images"]
    images_aux = _prepare_aux_images(gen_kwargs.get("images_aux"))
    image_grid_thws = gen_kwargs.get("image_grid_thws")
    bbox_list = gen_kwargs.get("bbox_list")
    if images_aux is None or bbox_list is None or len(bbox_list) == 0:
        return None

    device = images[0].device
    boxes = bbox_list[0].float().to(device)
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    n_roi = boxes.shape[0]
    if region_labels is None:
        region_labels = [f"region{i}" for i in range(n_roi)]

    task_type = _task_type_from_id(task_id)
    roi_size = int(extractor.roi_output_size or 14)

    with torch.inference_mode():
        _, _, vt_multi_level = model.encode_images(images, image_grid_thws)
        aux_out_list = core.get_vision_tower_aux()(images_aux)
        aux_feats = list(aux_out_list[0]["image_features"])

        patch_size = core.get_vision_tower().config.patch_size
        vt_images_size = image_grid_thws[0][0][-2:] * patch_size
        aux_hw = (images_aux[0].shape[-2], images_aux[0].shape[-1])
        vt_boxes_xyxy = _compute_vt_boxes(boxes, vt_images_size, aux_hw, device)
        aux_roi_boxes = _format_roi_boxes(boxes)
        vt_roi_boxes = _format_roi_boxes(vt_boxes_xyxy)

        result = {
            "roi_output_size": roi_size,
            "region_labels": region_labels,
            "bboxes": boxes.detach().cpu().tolist(),
            "vt_fpn": None,
            "aux": None,
            "shapes": {},
        }

        if getattr(extractor, "use_vision_tower_region_feature", False):
            vt_input = vt_multi_level[0][-1]
            if getattr(extractor, "use_simpleFPN_for_vt", False):
                fpn_levels = list(extractor.simple_fpn(vt_input))
                level_indices = getattr(extractor, "fpn_level_indices", list(range(len(fpn_levels))))
                vt_strides = getattr(extractor, "vt_fpn_strides", [3.5, 7.0, 14.0, 28.0])
                pre_pool = []
                post_pool = []
                for rel_idx, src_idx in enumerate(level_indices):
                    if src_idx >= len(fpn_levels):
                        continue
                    feat = fpn_levels[src_idx]
                    stride = vt_strides[rel_idx] if rel_idx < len(vt_strides) else vt_strides[-1]
                    aligned = roi_align(
                        feat.float(),
                        vt_roi_boxes,
                        output_size=roi_size,
                        spatial_scale=1.0 / float(stride),
                    )
                    pre_pool.append(aligned)
                    post_pool.append(aligned.mean(dim=(2, 3)))
                fused = None
                if getattr(extractor, "use_hierarchical_fusion", False) and hasattr(extractor, "hierarchical_fusion"):
                    fused = extractor.hierarchical_fusion(post_pool, task_type=task_type)
                result["vt_fpn"] = {
                    "pre_pool": pre_pool,
                    "post_pool": post_pool,
                    "fused": fused,
                }
                result["shapes"]["vt_fpn_pre_pool"] = [list(x.shape) for x in pre_pool]
                result["shapes"]["vt_fpn_post_pool"] = [list(x.shape) for x in post_pool]
                if fused is not None:
                    result["shapes"]["vt_fpn_fused"] = list(fused.shape)

        concat_aux = _build_aux_concat_features(extractor, aux_feats)
        aux_aligned = roi_align(
            concat_aux,
            aux_roi_boxes,
            output_size=roi_size,
            spatial_scale=extractor.aux_vision_tower_spatial_scale,
        )
        aux_pooled = aux_aligned.mean(dim=(2, 3))
        result["aux"] = {
            "pre_pool": aux_aligned,
            "post_pool": aux_pooled,
        }
        result["shapes"]["aux_pre_pool"] = list(aux_aligned.shape)
        result["shapes"]["aux_post_pool"] = list(aux_pooled.shape)

        # 触发与 generate 相同的 region 编码路径，便于 hierarchical_attn 缓存
        model.encode_regions(
            images_aux,
            bbox_list,
            vt_multi_level,
            [vt_images_size],
            task_ids=gen_kwargs.get("task_ids"),
        )

    return result


def _save_vt_roi_visualizations(roi_data, base_image: Image.Image, out_dir: str):
    vt = roi_data.get("vt_fpn")
    if vt is None:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    labels = roi_data["region_labels"]
    pre_pool = vt["pre_pool"]
    post_pool = vt["post_pool"]
    fused = vt.get("fused")
    saved = {"regions": []}

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("需要 matplotlib 才能保存 ROI 可视化") from e

    n_levels = len(pre_pool)
    for ridx, label in enumerate(labels):
        region_dir = os.path.join(out_dir, label)
        os.makedirs(region_dir, exist_ok=True)
        crop = _crop_image_by_bbox(base_image, roi_data["bboxes"][ridx])
        crop_path = os.path.join(region_dir, f"{label}_crop.jpg")
        crop.save(crop_path)

        n_cols = n_levels + (1 if fused is not None else 0)
        fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * max(n_cols, 1), 6))
        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for lvl, aligned in enumerate(pre_pool):
            hm = _roi_spatial_heatmap(aligned[ridx])
            axes[0, lvl].imshow(hm, cmap="inferno", aspect="equal")
            axes[0, lvl].set_title(f"ROIAlign L{lvl + 1}\n{hm.shape[0]}×{hm.shape[1]}")
            axes[0, lvl].axis("off")
            pre_path = os.path.join(region_dir, f"{label}_prepool_L{lvl + 1}.png")
            _save_heatmap_image(hm, pre_path, cmap_name="inferno")

        for lvl, pooled in enumerate(post_pool):
            vec_hm = _vector_heatmap(pooled[ridx])
            axes[1, lvl].imshow(vec_hm, cmap="viridis", aspect="auto")
            axes[1, lvl].set_title(f"Pooled L{lvl + 1}\nC={pooled.shape[1]}")
            axes[1, lvl].axis("off")
            pool_path = os.path.join(region_dir, f"{label}_pooled_L{lvl + 1}.png")
            _save_heatmap_image(vec_hm, pool_path, cmap_name="viridis")

        next_col = n_levels
        if fused is not None:
            vec_hm = _vector_heatmap(fused[ridx])
            axes[0, next_col].imshow(vec_hm, cmap="magma", aspect="auto")
            axes[0, next_col].set_title(f"Fused\nC={fused.shape[1]}")
            axes[0, next_col].axis("off")
            axes[1, next_col].axis("off")
            fused_path = os.path.join(region_dir, f"{label}_fused.png")
            _save_heatmap_image(vec_hm, fused_path, cmap_name="magma")

        fig.suptitle(f"{label} | VT-FPN ROIAlign + Pool", fontsize=11)
        fig.tight_layout()
        panel_path = os.path.join(region_dir, f"{label}_panel.png")
        fig.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved["regions"].append(
            {
                "label": label,
                "crop": crop_path,
                "panel": panel_path,
                "pre_pool": [os.path.join(region_dir, f"{label}_prepool_L{i + 1}.png") for i in range(n_levels)],
                "pooled": [os.path.join(region_dir, f"{label}_pooled_L{i + 1}.png") for i in range(n_levels)],
                "fused": os.path.join(region_dir, f"{label}_fused.png") if fused is not None else None,
            }
        )

    grid_path = os.path.join(out_dir, "all_regions_pooled_L1-L4.png")
    fig, axes = plt.subplots(len(labels), n_levels, figsize=(3.0 * n_levels, 2.8 * len(labels)))
    if len(labels) == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_levels == 1:
        axes = np.expand_dims(axes, axis=1)
    for ridx, label in enumerate(labels):
        for lvl in range(n_levels):
            hm = _roi_spatial_heatmap(pre_pool[lvl][ridx])
            axes[ridx, lvl].imshow(hm, cmap="inferno", aspect="equal")
            axes[ridx, lvl].set_title(f"{label} | L{lvl + 1}")
            axes[ridx, lvl].axis("off")
    fig.suptitle("VT-FPN ROIAlign (pre-pool spatial heatmap)", fontsize=12)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["summary_grid"] = grid_path
    return saved


def _save_aux_roi_visualizations(roi_data, base_image: Image.Image, out_dir: str):
    aux = roi_data.get("aux")
    if aux is None:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    labels = roi_data["region_labels"]
    aligned = aux["pre_pool"]
    pooled = aux["post_pool"]
    saved = {"regions": []}

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("需要 matplotlib 才能保存 ROI 可视化") from e

    for ridx, label in enumerate(labels):
        region_dir = os.path.join(out_dir, label)
        os.makedirs(region_dir, exist_ok=True)
        crop = _crop_image_by_bbox(base_image, roi_data["bboxes"][ridx])
        crop_path = os.path.join(region_dir, f"{label}_crop.jpg")
        crop.save(crop_path)

        pre_hm = _roi_spatial_heatmap(aligned[ridx])
        pool_hm = _vector_heatmap(pooled[ridx], ncol=128)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(crop)
        axes[0].set_title("Crop")
        axes[0].axis("off")
        axes[1].imshow(pre_hm, cmap="inferno", aspect="equal")
        axes[1].set_title(f"ROIAlign {pre_hm.shape[0]}×{pre_hm.shape[1]}")
        axes[1].axis("off")
        axes[2].imshow(pool_hm, cmap="viridis", aspect="auto")
        axes[2].set_title(f"Pooled C={pooled.shape[1]}")
        axes[2].axis("off")
        fig.suptitle(f"{label} | Aux (DaViT concat) ROIAlign + Pool", fontsize=11)
        fig.tight_layout()
        panel_path = os.path.join(region_dir, f"{label}_panel.png")
        fig.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        pre_path = os.path.join(region_dir, f"{label}_prepool.png")
        pool_path = os.path.join(region_dir, f"{label}_pooled.png")
        _save_heatmap_image(pre_hm, pre_path, cmap_name="inferno")
        _save_heatmap_image(pool_hm, pool_path, cmap_name="viridis")
        saved["regions"].append(
            {
                "label": label,
                "crop": crop_path,
                "panel": panel_path,
                "pre_pool": pre_path,
                "pooled": pool_path,
            }
        )
    return saved


def _resolve_vt_fpn_levels(model, vt_raw_levels):
    core = model.get_model()
    extractor = getattr(core, "object_vp_extractor", None)
    if extractor is None or not getattr(extractor, "use_simpleFPN_for_vt", False):
        return None
    if not vt_raw_levels:
        return None
    last_level = vt_raw_levels[-1]
    with torch.inference_mode():
        fpn_out = extractor.simple_fpn(last_level)
    return list(fpn_out)


def _prepare_aux_images(images_aux):
    """与 omchat_qwen2_5_vl.encode_regions 一致：DaViT 需要 [1,C,H,W]。"""
    if images_aux is None:
        return None
    out = []
    for img in images_aux:
        if img.dim() == 3:
            out.append(img.unsqueeze(0))
        elif img.dim() == 4:
            out.append(img)
        else:
            raise ValueError(f"images_aux 期望 3D/4D tensor，得到 shape={tuple(img.shape)}")
    return out


def capture_encoding_features(model, gen_kwargs):
    """显式走 encode_images / encode_regions，与 generate 首步编码路径一致。"""
    core = model.get_model()
    images = gen_kwargs["images"]
    images_aux = gen_kwargs.get("images_aux")
    image_grid_thws = gen_kwargs.get("image_grid_thws")
    bbox_list = gen_kwargs.get("bbox_list")
    task_ids = gen_kwargs.get("task_ids")
    images_minibatch_aux = _prepare_aux_images(images_aux)

    captured = {
        "vt_raw": None,
        "vt_fpn": None,
        "aux": None,
        "shapes": {},
    }

    with torch.inference_mode():
        _, _, vt_multi_level = model.encode_images(images, image_grid_thws)
        if vt_multi_level and len(vt_multi_level) > 0:
            captured["vt_raw"] = list(vt_multi_level[0])
            captured["shapes"]["vt_raw"] = [list(x.shape) for x in captured["vt_raw"]]

        captured["vt_fpn"] = _resolve_vt_fpn_levels(model, captured["vt_raw"])
        if captured["vt_fpn"] is not None:
            captured["shapes"]["vt_fpn"] = [list(x.shape) for x in captured["vt_fpn"]]

        if images_minibatch_aux is not None and core.get_vision_tower_aux() is not None:
            aux_out_list = core.get_vision_tower_aux()(images_minibatch_aux)
            if aux_out_list:
                captured["aux"] = list(aux_out_list[0]["image_features"])
                captured["shapes"]["aux"] = [list(x.shape) for x in captured["aux"]]

    return captured


def main():
    parser = argparse.ArgumentParser(description="单样本推理 + 编码多尺度特征可视化")
    parser.add_argument(
        "--data_json",
        type=str,
        default=DEFAULT_DATA_JSON,
        help="数据集 JSON（含 image / regions / conversations）",
    )
    parser.add_argument("--sample_id", type=str, default=None, help="按样本 id 选取（优先于 sample_index）")
    parser.add_argument("--sample_index", type=int, default=0, help="按数组下标选取样本")
    parser.add_argument("--image", type=str, default=None, help="直接指定图像路径（绕过 data_json）")
    parser.add_argument("--user_text", type=str, default=None, help="用户问题文本（不含 <image>）")
    parser.add_argument("--task_id", type=int, default=None, help="覆盖 JSON 中的 task_id")
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/stage1_layerGated_taskAdaptive",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/stage2_layerGated_taskAdaptive",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result/infer_one_sample_multiscale_vis",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=896, help="最短边缩放；0 表示关闭")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument(
        "--no_hierarchical_fusion",
        action="store_true",
        help="关闭层级融合（默认开启）",
    )
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="task_adaptive",
        choices=["legacy", "task_adaptive", "gated"],
    )
    args = parser.parse_args()

    sample_meta = {}
    gt_text = None
    regions = []
    region_id_to_index = {}
    highlight_index = None

    if args.image:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"图像不存在: {args.image}")
        image_path = args.image
        user_text = args.user_text or "描述图中这个区域<region0>"
        bbox_list_for_model = None
        task_id = args.task_id if args.task_id is not None else 2
        sample_meta = {"id": os.path.basename(image_path), "image": image_path, "source": "cli"}
    else:
        data_path = args.data_json or DEFAULT_DATA_JSON
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        data = _load_json_dataset(data_path)
        item, picked_index = _find_sample(data, sample_id=args.sample_id, sample_index=args.sample_index)
        (
            image_path,
            user_text,
            gt_text,
            bbox_list_for_model,
            task_id,
            regions,
            region_id_to_index,
        ) = _extract_dialog_fields(item)
        if args.task_id is not None:
            task_id = args.task_id
        elif task_id is None:
            task_id = 2
        else:
            task_id = int(task_id)
        sample_meta = {
            "id": item.get("id"),
            "image": image_path,
            "source": data_path,
            "sample_index": picked_index,
            "task_id": task_id,
            "regions": regions,
        }
        highlight_index = _infer_highlight_region_index(user_text, region_id_to_index)

    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")

    user_text = add_task_constraints_to_user_text(user_text, task_id, language=args.language)

    pil_orig = Image.open(image_path).convert("RGB")
    w, h = pil_orig.size
    if bbox_list_for_model:
        vis_bboxes = bbox_list_for_model
    else:
        bbox_list_for_model = [[0.0, 0.0, float(w), float(h)]]
        vis_bboxes = bbox_list_for_model

    region_ids = [r.get("region_id", f"region{i}") for i, r in enumerate(regions)] if regions else None
    if args.output_dir == "eval_result/infer_one_sample_multiscale_vis" and sample_meta.get("id"):
        args.output_dir = os.path.join(args.output_dir, str(sample_meta["id"]))

    os.makedirs(args.output_dir, exist_ok=True)
    base_vis_path = os.path.join(args.output_dir, "input_image.jpg")
    _draw_bboxes(
        pil_orig,
        vis_bboxes,
        region_ids=region_ids,
        highlight_index=highlight_index,
    ).save(base_vis_path)

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
    model.eval()

    model_name_for_template = (
        args.base_model
        if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model)
        else DEFAULT_VLM_FO1_BASE
    )
    image_size = None if args.image_size == 0 else args.image_size

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

    gen_kwargs = prepare_inputs(
        model_name=model_name_for_template,
        model=model,
        image_processors=image_processors,
        tokenizer=tokenizer,
        messages=messages,
        device=args.device,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        path_feature_root=None,
        task_ids=[int(task_id)],
        image_size=image_size,
        verbose=True,
    )

    captured = capture_encoding_features(model, gen_kwargs)

    region_labels = None
    if regions:
        region_labels = [r.get("region_id", f"region{i}") for i, r in enumerate(regions)]
    elif bbox_list_for_model:
        region_labels = [f"region{i}" for i in range(len(bbox_list_for_model))]

    roi_outputs = capture_roi_align_outputs(
        model,
        gen_kwargs,
        task_id,
        region_labels=region_labels,
    )

    resized_pil = Image.open(base_vis_path).convert("RGB")
    if gen_kwargs.get("images_aux") is not None:
        aux_tensor = gen_kwargs["images_aux"][0].detach().float().cpu()
        aux_arr = aux_tensor.numpy()
        if aux_arr.ndim == 3:
            if aux_arr.shape[0] in (1, 3):
                chw = aux_arr
                if chw.shape[0] == 1:
                    chw = np.repeat(chw, 3, axis=0)
                chw = (chw - chw.min()) / max(chw.max() - chw.min(), 1e-6)
                resized_pil = Image.fromarray((chw.transpose(1, 2, 0) * 255).astype(np.uint8))

    vis_outputs = {}
    if captured["vt_raw"]:
        titles = [f"VT-L{i + 1}" for i in range(len(captured["vt_raw"]))]
        vis_outputs["vt_raw"] = _save_feature_group(
            captured["vt_raw"],
            titles,
            "vt_raw",
            resized_pil,
            os.path.join(args.output_dir, "vt_raw"),
            alpha=args.overlay_alpha,
        )
    if captured["vt_fpn"]:
        titles = [f"VT-FPN-L{i + 1}" for i in range(len(captured["vt_fpn"]))]
        vis_outputs["vt_fpn"] = _save_feature_group(
            captured["vt_fpn"],
            titles,
            "vt_fpn",
            resized_pil,
            os.path.join(args.output_dir, "vt_fpn"),
            alpha=args.overlay_alpha,
        )
    if captured["aux"]:
        titles = [f"Aux-L{i + 1}" for i in range(len(captured["aux"]))]
        vis_outputs["aux"] = _save_feature_group(
            captured["aux"],
            titles,
            "aux_davit",
            resized_pil,
            os.path.join(args.output_dir, "aux_davit"),
            alpha=args.overlay_alpha,
        )

    if roi_outputs:
        pil_for_roi = Image.open(image_path).convert("RGB")
        vis_outputs["vt_roi"] = _save_vt_roi_visualizations(
            roi_outputs,
            pil_for_roi,
            os.path.join(args.output_dir, "vt_roi"),
        )
        vis_outputs["aux_roi"] = _save_aux_roi_visualizations(
            roi_outputs,
            pil_for_roi,
            os.path.join(args.output_dir, "aux_roi"),
        )
        captured["shapes"].update(roi_outputs.get("shapes", {}))

    core = model.get_model()
    extractor = getattr(core, "object_vp_extractor", None)
    hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
    hf_inner = _unwrap_hierarchical_fusion_module(hf)
    attn_path = os.path.join(args.output_dir, "hierarchical_attn.png")
    if hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
        visualize_hierarchical_attention(hf_inner, save_path=attn_path)
        vis_outputs["hierarchical_attn"] = attn_path

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    in_len = int(gen_kwargs["inputs"].shape[1])
    output_text = tokenizer.decode(
        output_ids[0, in_len:],
        skip_special_tokens=False,
    ).strip()

    summary = {
        "sample": sample_meta,
        "task_id": int(task_id),
        "user_text": user_text,
        "ground_truth": gt_text,
        "prediction": output_text,
        "bbox_list": bbox_list_for_model,
        "region_id_to_index": region_id_to_index,
        "highlight_region_index": highlight_index,
        "feature_shapes": captured["shapes"],
        "roi_align": {
            "roi_output_size": roi_outputs.get("roi_output_size") if roi_outputs else None,
            "region_labels": roi_outputs.get("region_labels") if roi_outputs else None,
            "shapes": roi_outputs.get("shapes") if roi_outputs else None,
        },
        "visualizations": vis_outputs,
        "input_image": base_vis_path,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("单样本推理完成")
    print("=" * 60)
    print(f"  样本: {sample_meta.get('id')}")
    print(f"  图像: {image_path}")
    if gt_text is not None:
        print(f"  GT:   {gt_text}")
    print(f"  Pred: {output_text}")
    if captured["shapes"]:
        print(f"  特征 shape: {json.dumps(captured['shapes'], ensure_ascii=False)}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
