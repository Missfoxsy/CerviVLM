#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task1 单样本：模型加载与 infer_attMap_task1.sh / infer_stage2_task1_attention.py 一致；
可视化逻辑与 scripts/qwen-vl-visualize.py 一致（Monkey-patch Qwen2.5-VL 视觉塔各层 attn，
捕获自注意力 → JET 热图叠原图；默认导出全部层在一张总览 grid 中对比）。

注意：
- 仅对视觉塔做前向（不跑 LLM 生成），输入图与 stage2 JSON 中 image 字段一致。
- FlashAttention2 实现无法直接导出权重矩阵；请在本脚本运行前设置：
    export VLM_FO1_ATTN_IMPLEMENTATION=eager
  （shell 脚本已默认设置）

用法见 scripts/visualize_attMap_task1.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from functools import partial
from typing import Any, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(s))[:200]


def _load_json_list(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 顶层为 list，得到: {type(data)}")
    return data


def _find_task1_items(data: list, sample_id: str) -> list[tuple[int, dict]]:
    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "")) != str(sample_id):
            continue
        tid = item.get("task_id", None)
        try:
            if tid is None or int(tid) != 1:
                continue
        except (TypeError, ValueError):
            continue
        out.append((i, item))
    return out


def _parse_user_assistant(item: dict) -> tuple[str | None, str | None]:
    user_text, assistant_text = None, None
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            user_text = str(conv.get("value", "")).replace("<image>", "").strip()
            if user_text.startswith("\n"):
                user_text = user_text[1:]
        elif conv.get("from") == "assistant":
            assistant_text = str(conv.get("value", ""))
    return user_text, assistant_text


def _parse_layer_indices(s: str, num_blocks: int) -> List[int]:
    """解析 'all' / '7,15,23,31' / '0-31:1'；越界则裁剪。"""
    s = str(s).strip()
    if not s or s.lower() in ("all", "*"):
        return list(range(num_blocks))
    if s.lower() == "default4":
        return [num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks - 1]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part or "-" in part and part.count("-") == 1 and ":" in part:
            # 7-31:8
            rng, step = part.split(":")
            step = int(step) if step else 1
            a, b = rng.split("-")
            out.extend(range(int(a), int(b) + 1, step))
        elif "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted({i for i in out if 0 <= i < num_blocks})
    if not out:
        raise ValueError(f"解析 layers 后无有效层索引（共 {num_blocks} 层）: {s!r}")
    return out


def _get_qwen_vision_vit(model) -> Any:
    from vlm_fo1.model.multimodal_encoder.qwen2_5_vl_encoder import Qwen2_5_VlVisionTower

    core = model.get_model() if hasattr(model, "get_model") else model
    vt = core.get_vision_tower() if hasattr(core, "get_vision_tower") else None
    if vt is None:
        raise RuntimeError("当前模型无 vision_tower，无法做 Qwen ViT 注意力可视化。")
    if not isinstance(vt, Qwen2_5_VlVisionTower):
        raise RuntimeError(
            f"vision_tower 类型为 {type(vt).__name__}，本脚本仅支持 Qwen2_5_VlVisionTower。"
        )
    vit = vt.image_tower
    if not hasattr(vit, "blocks"):
        raise RuntimeError("image_tower 上找不到 blocks，结构可能与 Qwen2.5-VL 不一致。")
    return vt, vit


def _make_capture_patch(capture_box: list):
    """
    与 qwen-vl-visualize.py 相同思路：在标准 eager 注意力路径上保存 softmax 权重。
    适配 VLM 视觉塔 forward 签名 (hidden_states, cu_seqlens, position_embeddings=...)。
    """
    from vlm_fo1.model.multimodal_encoder.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = q.shape[-1]

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        capture_box[0] = attn_weights.detach()

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    return patched_forward


def _check_attn_patchable(attn_module) -> None:
    cls = attn_module.__class__.__name__
    if "FlashAttention" in cls:
        raise RuntimeError(
            f"视觉塔注意力为 {cls}，无法导出注意力矩阵。请设置环境变量 "
            "VLM_FO1_ATTN_IMPLEMENTATION=eager 后重新加载模型（visualize_attMap_task1.sh 已默认设置）。"
        )


def _run_vit_forward_capture_layer(
    vit,
    pixel_values,
    image_grid_thw,
    layer_idx: int,
    capture_box: list,
) -> None:
    import torch

    target = vit.blocks[layer_idx].attn
    _check_attn_patchable(target)
    original_forward = target.forward
    target.forward = partial(_make_capture_patch(capture_box), target)
    try:
        with torch.no_grad():
            vit(
                pixel_values.to(device=vit.device, dtype=vit.dtype),
                grid_thw=image_grid_thw.to(device=vit.device),
            )
    finally:
        target.forward = original_forward


def _build_spatial_layout(vit, image_grid_thw) -> dict:
    """
    Qwen ViT 在 forward 前用 window_index 打乱 token 顺序；热图必须先还原到
    (llm_grid_h, llm_grid_w) 自然栅格，再 resize 到原图。
    """
    import numpy as np
    import torch

    grid_thw = image_grid_thw.detach()
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)
    window_index, _ = vit.get_window_index(grid_thw)
    wi = window_index.cpu().numpy().astype(np.int64)
    t = int(grid_thw[0, 0].item())
    gh = int(grid_thw[0, 1].item())
    gw = int(grid_thw[0, 2].item())
    merge_size = int(vit.spatial_merge_size)
    llm_h = gh // merge_size
    llm_w = gw // merge_size
    n_groups = t * llm_h * llm_w
    if wi.shape[0] != n_groups:
        raise RuntimeError(
            f"window_index 长度 {wi.shape[0]} 与 grid {t}x{llm_h}x{llm_w}={n_groups} 不一致"
        )
    fullatt = set(int(x) for x in getattr(vit, "fullatt_block_indexes", []) or [])
    return {
        "window_index": wi,
        "llm_grid_h": llm_h,
        "llm_grid_w": llm_w,
        "spatial_merge_unit": int(vit.spatial_merge_unit),
        "fullatt_block_indexes": fullatt,
    }


def _attn_to_spatial_grid(
    attn_raw,
    layout: dict,
    *,
    saliency_mode: str = "received",
) -> "np.ndarray":
    """
    将捕获的 [heads, S, S] 注意力变为 llm 栅格上的 saliency（已 undo window_index）。
    - received: 各 key 收到的注意力均值（列均值，比整列求和更稳）
    - self: 各位置自注意力强度 mean_h attn[h,i,i]（更稀疏、少糊成一片）
    """
    import numpy as np
    import torch

    attn = attn_raw.detach().float().cpu()
    if attn.dim() == 4:
        attn = attn[0]
    if attn.dim() != 3:
        raise ValueError(f"期望注意力为 [H,S,S]，得到 shape={tuple(attn.shape)}")
    attn = attn.mean(dim=0)  # [S, S]

    mode = str(saliency_mode).strip().lower()
    if mode == "self":
        s = min(attn.shape[0], attn.shape[1])
        sal_perm = attn[:s, :s].diag()
    else:
        sal_perm = attn.mean(dim=0)  # 各 key 收到的平均注意力

    merge_unit = int(layout["spatial_merge_unit"])
    wi = layout["window_index"]
    n_groups = len(wi)
    seq_len = int(sal_perm.shape[0])
    if seq_len != n_groups * merge_unit:
        raise RuntimeError(
            f"序列长 {seq_len} != n_groups({n_groups})*merge_unit({merge_unit})，"
            "无法按 Qwen ViT 规则还原空间布局。"
        )

    group_sal = sal_perm.reshape(n_groups, merge_unit).mean(dim=1).numpy()
    llm_h, llm_w = int(layout["llm_grid_h"]), int(layout["llm_grid_w"])
    sal_natural = np.zeros(n_groups, dtype=np.float32)
    for perm_pos, nat_idx in enumerate(wi):
        nat = int(nat_idx)
        if 0 <= nat < n_groups:
            sal_natural[nat] = float(group_sal[perm_pos])

    return sal_natural.reshape(llm_h, llm_w)


def _normalize_heatmap(grid: "np.ndarray", p_low: float, p_high: float) -> "np.ndarray":
    import numpy as np

    g = grid.astype(np.float32)
    lo, hi = np.percentile(g, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-8
    g = np.clip(g, lo, hi)
    return (g - lo) / (hi - lo + 1e-8)


def _attn_tensor_to_heatmap(
    attn_raw,
    original_image_np,
    layout: dict,
    *,
    saliency_mode: str = "received",
    norm_p_low: float = 5.0,
    norm_p_high: float = 95.0,
    smooth_ksize: int = 0,
) -> Tuple[Any, Any]:
    """还原 window 顺序 → llm 栅格 → 原图尺寸；百分位归一化减轻「满屏红噪点」。"""
    import cv2
    import numpy as np

    grid = _attn_to_spatial_grid(attn_raw, layout, saliency_mode=saliency_mode)
    if smooth_ksize and int(smooth_ksize) >= 3:
        k = int(smooth_ksize) | 1
        grid = cv2.GaussianBlur(grid.astype(np.float32), (k, k), 0)

    h, w, _ = original_image_np.shape
    heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = _normalize_heatmap(heat, norm_p_low, norm_p_high)
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image_np, 0.6, colored, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return heat, overlay_rgb


def _save_layer_figure(
    *,
    original_image_np,
    overlay_rgb,
    layer_idx: int,
    prompt: str,
    output_path: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(original_image_np)
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(overlay_rgb)
    ax2.set_title(f"ViT Attention Heatmap (Layer {layer_idx + 1})")
    ax2.axis("off")
    title_prompt = (prompt[:80] + "…") if len(prompt) > 80 else prompt
    plt.suptitle(
        f"Qwen2.5-VL Vision Tower Attention (Layer {layer_idx + 1})\nPrompt: {title_prompt!r}",
        fontsize=14,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _save_all_layers_grid_figure(
    *,
    original_image_np,
    overlays: List[Tuple[int, Any]],
    prompt: str,
    output_path: str,
    grid_cols: int,
    dpi: int,
    show_original_first: bool = True,
) -> None:
    """将所有层的叠图热图排在一张 figure 中，便于逐层对比。"""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlays = sorted(overlays, key=lambda x: x[0])
    n_layers = len(overlays)
    if n_layers == 0:
        return

    panels: List[Tuple[str, Any]] = []
    if show_original_first:
        panels.append(("Original", original_image_np))
    for item in overlays:
        layer_idx = item[0]
        ov = item[1]
        is_fullatt = bool(item[2]) if len(item) > 2 else False
        tag = f"L{layer_idx + 1}"
        if is_fullatt:
            tag += "†"
        panels.append((tag, ov))

    n_panels = len(panels)
    cols = max(1, int(grid_cols))
    rows = (n_panels + cols - 1) // cols
    cell_w = 3.2
    cell_h = 3.0
    fig, axs = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows))
    axs_flat = np.atleast_1d(axs).ravel()

    for ax, (title, img) in zip(axs_flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axs_flat[n_panels:]:
        ax.axis("off")

    title_prompt = (prompt[:100] + "…") if len(prompt) > 100 else prompt
    fig.suptitle(
        f"Qwen2.5-VL Vision Tower — All Layers ({n_layers} layers, †=full-image attn)\n"
        f"{title_prompt!r}",
        fontsize=12,
        y=1.02,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def visualize_sample_qwen_vit(
    *,
    model,
    data_item: dict,
    out_dir: str,
    layers: List[int],
    dpi: int = 150,
    save_individual_layers: bool = False,
    grid_cols: int = 4,
    show_original_in_grid: bool = True,
    saliency_mode: str = "received",
    norm_p_low: float = 5.0,
    norm_p_high: float = 95.0,
    smooth_ksize: int = 0,
) -> dict:
    import numpy as np
    from PIL import Image

    sid = str(data_item.get("id", "unknown"))
    safe = _sanitize_filename(sid)
    image_path = data_item["image"]
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"样本 {sid} 图片不存在: {image_path!r}")

    user_text, _ = _parse_user_assistant(data_item)
    prompt = user_text or ""

    vt, vit = _get_qwen_vision_vit(model)
    pil_image = Image.open(image_path).convert("RGB")
    original_np = np.array(pil_image)

    proc = vt.image_processor
    inputs = proc(images=pil_image, videos=None, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    spatial_layout = _build_spatial_layout(vit, image_grid_thw)
    fullatt_set = spatial_layout["fullatt_block_indexes"]
    print(
        f"[Info] llm 栅格 {spatial_layout['llm_grid_h']}x{spatial_layout['llm_grid_w']}, "
        f"merge_unit={spatial_layout['spatial_merge_unit']}; "
        f"全图注意力层(†): {sorted(fullatt_set)}",
        flush=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    saved_paths: List[str] = []
    overlays_for_grid: List[Tuple[int, Any, bool]] = []

    for layer_idx in layers:
        print(
            f"[Qwen-ViT] 处理视觉层 {layer_idx + 1}/{len(vit.blocks)} "
            f"(index {layer_idx}, {len(overlays_for_grid) + 1}/{len(layers)}) …",
            flush=True,
        )
        capture_box: list = [None]
        _run_vit_forward_capture_layer(vit, pixel_values, image_grid_thw, layer_idx, capture_box)
        if capture_box[0] is None:
            print(f"[Warn] 层 {layer_idx} 未捕获到注意力，跳过。")
            continue

        _, overlay_rgb = _attn_tensor_to_heatmap(
            capture_box[0],
            original_np,
            spatial_layout,
            saliency_mode=saliency_mode,
            norm_p_low=norm_p_low,
            norm_p_high=norm_p_high,
            smooth_ksize=smooth_ksize,
        )
        is_fullatt = layer_idx in fullatt_set
        overlays_for_grid.append((layer_idx, overlay_rgb, is_fullatt))

        if save_individual_layers:
            out_png = os.path.join(out_dir, f"{safe}_qwen_vit_attention_layer_{layer_idx + 1}.png")
            _save_layer_figure(
                original_image_np=original_np,
                overlay_rgb=overlay_rgb,
                layer_idx=layer_idx,
                prompt=prompt,
                output_path=out_png,
                dpi=dpi,
            )
            saved_paths.append(os.path.abspath(out_png))
            print(f"[OK] 已保存单层: {out_png}")

    grid_path = os.path.join(out_dir, f"{safe}_qwen_vit_attention_all_layers_grid.png")
    if overlays_for_grid:
        _save_all_layers_grid_figure(
            original_image_np=original_np,
            overlays=overlays_for_grid,
            prompt=prompt,
            output_path=grid_path,
            grid_cols=grid_cols,
            dpi=dpi,
            show_original_first=show_original_in_grid,
        )
        print(f"[OK] 全层对比总图: {grid_path}（共 {len(overlays_for_grid)} 层）")
    else:
        grid_path = ""
        print("[Warn] 未成功捕获任何层，未生成总图。")

    meta = {
        "id": sid,
        "image": image_path,
        "user_text": user_text,
        "layers_visualized": [item[0] for item in overlays_for_grid],
        "attention_pngs": saved_paths,
        "all_layers_grid_png": os.path.abspath(grid_path) if grid_path else None,
        "num_vision_blocks": len(vit.blocks),
        "llm_grid": [spatial_layout["llm_grid_h"], spatial_layout["llm_grid_w"]],
        "fullatt_block_indexes": sorted(fullatt_set),
        "saliency_mode": saliency_mode,
        "attn_implementation_note": "需要 VLM_FO1_ATTN_IMPLEMENTATION=eager；热图已 undo window_index",
    }
    meta_path = os.path.join(out_dir, f"{safe}_qwen_vit_attn_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] 元数据: {meta_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Task1：infer 同款 VLM 加载 + qwen-vl-visualize 同款 Qwen 视觉塔逐层注意力图",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
    )
    parser.add_argument("--sample_id", type=str, default=None, help="逗号分隔多个 id（--list_task1_ids 时可省略）")
    parser.add_argument("--base_model", type=str, default="outputs/stage1_layerGated_taskAdaptive")
    parser.add_argument("--checkpoint", type=str, default="outputs/stage2_layerGated_taskAdaptive")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="attention_infer/task1_qwen_vit_layers")
    parser.add_argument("--use_hierarchical_fusion", action="store_true")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="task_adaptive",
        choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"],
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="visual.blocks 层索引（0 起）：all=全部层；或逗号/范围如 7,15 或 0-31:2",
    )
    parser.add_argument(
        "--save_individual_layers",
        action="store_true",
        help="除总览图外，另存每层单独的双栏 PNG（层数多时会较慢）",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="全层总览图的列数",
    )
    parser.add_argument(
        "--no_original_in_grid",
        action="store_true",
        help="总览图左上角不单独放原图",
    )
    parser.add_argument("--dpi", type=int, default=120, help="输出 DPI；全层总图建议 100–120 以控制文件大小")
    parser.add_argument(
        "--saliency_mode",
        type=str,
        default="received",
        choices=["received", "self"],
        help="received=各 patch 收到的注意力(列均值); self=对角自注意力(更稀疏)",
    )
    parser.add_argument("--norm_p_low", type=float, default=5.0, help="热图百分位归一化下界")
    parser.add_argument("--norm_p_high", type=float, default=95.0, help="热图百分位归一化上界")
    parser.add_argument(
        "--smooth_ksize",
        type=int,
        default=0,
        help="栅格高斯平滑核大小(奇数，0=不平滑)",
    )
    parser.add_argument(
        "--list_task1_ids",
        action="store_true",
        help="仅列出 task_id=1 的 id",
    )
    parser.add_argument("--list_limit", type=int, default=50)

    args = parser.parse_args()
    data = _load_json_list(args.test_json)

    if args.list_task1_ids:
        n = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                if int(item.get("task_id", -1)) != 1:
                    continue
            except (TypeError, ValueError):
                continue
            print(item.get("id", ""))
            n += 1
            if args.list_limit > 0 and n >= args.list_limit:
                break
        print(f"(共 {n} 条)")
        return

    if not args.sample_id or not str(args.sample_id).strip():
        raise SystemExit("请指定 --sample_id，或使用 --list_task1_ids 查看可用 id。")

    from eval_stage2_cot import (
        DEFAULT_VLM_FO1_BASE,
        _is_official_vlm_fo1_model_dir,
        load_checkpoint_model,
        maybe_enable_hierarchical_fusion,
    )

    attn_impl = os.environ.get("VLM_FO1_ATTN_IMPLEMENTATION", "sdpa")
    if attn_impl == "flash_attention_2":
        print(
            "[Warn] VLM_FO1_ATTN_IMPLEMENTATION=flash_attention_2，可能无法捕获注意力；"
            "建议 export VLM_FO1_ATTN_IMPLEMENTATION=eager",
            flush=True,
        )

    print("[Load] 按 infer_stage2_task1_attention 加载 VLM …", flush=True)
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
    model.eval()
    del tokenizer, image_processors

    _, vit = _get_qwen_vision_vit(model)
    layer_indices = _parse_layer_indices(args.layers, len(vit.blocks))
    print(f"[Info] 将可视化视觉塔层索引: {layer_indices}（共 {len(vit.blocks)} 层）", flush=True)

    sample_ids = [s.strip() for s in str(args.sample_id).split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    all_meta = []

    for sid in sample_ids:
        matches = _find_task1_items(data, sid)
        if not matches:
            raise SystemExit(f"未找到 task_id=1 且 id={sid!r} 的样本。")
        for _idx, item in matches:
            sub_out = os.path.join(args.output_dir, _sanitize_filename(sid))
            meta = visualize_sample_qwen_vit(
                model=model,
                data_item=item,
                out_dir=sub_out,
                layers=layer_indices,
                dpi=args.dpi,
                save_individual_layers=args.save_individual_layers,
                grid_cols=args.grid_cols,
                show_original_in_grid=not args.no_original_in_grid,
                saliency_mode=args.saliency_mode,
                norm_p_low=args.norm_p_low,
                norm_p_high=args.norm_p_high,
                smooth_ksize=args.smooth_ksize,
            )
            all_meta.append(meta)

    batch_path = os.path.join(args.output_dir, "batch_qwen_vit_attn_meta.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"[Done] 批量摘要: {batch_path}")


if __name__ == "__main__":
    main()
