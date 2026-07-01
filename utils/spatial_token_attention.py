# -*- coding: utf-8 -*-
"""
从 model.generate(return_dict_in_generate=True, output_attentions=True) 的结果中，
提取「当前解码步对视觉 token 的交叉注意力」，近似重排为二维热图并叠到原图上。

与 scripts/eval_stage2_attVisual.py 中逻辑一致，供推理脚本复用。
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw


def to_numpy_attention(attn_obj):
    """将不同形态 attention 转为 [heads, q, k] numpy。"""
    if attn_obj is None:
        return None
    if isinstance(attn_obj, torch.Tensor):
        arr = attn_obj.detach().float().cpu().numpy()
    elif isinstance(attn_obj, np.ndarray):
        arr = attn_obj.astype(np.float32)
    else:
        return None

    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return None
    return arr


def extract_step_attn_tensors(generation_output):
    """
    从 generate(return_dict_in_generate=True, output_attentions=True) 中提取 step attentions。
    返回 list[Tensor]，每个元素对应一个解码步的「最后一层」attention。
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
        if isinstance(step_obj, (tuple, list)) and len(step_obj) > 0:
            parsed.append(step_obj[-1])
        else:
            parsed.append(step_obj)
    return parsed


def guess_image_token_indices(inputs, tokenizer) -> list[int]:
    """从紧凑 input_ids 中识别 image / vision 相关下标（与展开后序列不一定对齐，仅作兜底）。"""
    if inputs is None:
        return []
    if isinstance(inputs, torch.Tensor):
        ids = inputs[0].detach().cpu().tolist()
    elif isinstance(inputs, np.ndarray):
        ids = inputs[0].tolist()
    else:
        return []

    # OmChat/Qwen-VL 紧凑序列里常用 IMAGE_TOKEN_INDEX=-200、REGION=-300；展开后才是大量 <|image_pad|>
    try:
        from vlm_fo1.constants import DEFAULT_REGION_INDEX, IMAGE_TOKEN_INDEX, QWEN2_5_VL_IMAGE_TOKEN_INDEX
    except Exception:
        IMAGE_TOKEN_INDEX = -200
        DEFAULT_REGION_INDEX = -300
        QWEN2_5_VL_IMAGE_TOKEN_INDEX = 151655

    special_ids = {int(IMAGE_TOKEN_INDEX), int(DEFAULT_REGION_INDEX), int(QWEN2_5_VL_IMAGE_TOKEN_INDEX)}
    for tok in (
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
        "</img>",
    ):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid >= 0:
                special_ids.add(int(tid))
        except Exception:
            continue

    return [i for i, tid in enumerate(ids) if int(tid) in special_ids]


def count_llm_image_tokens_from_grids(image_grid_thws, merge_length: int = 4) -> int:
    """
    与 Qwen2.5-VL processor 一致：每个占位 <|image_pad|> 数 = grid.prod() // merge_length。
    merge_length 默认 4（spatial_merge_size=2）。
    """
    if not image_grid_thws:
        return 0
    total = 0
    for g in image_grid_thws:
        if g is None:
            continue
        if isinstance(g, torch.Tensor):
            flat = g.detach().long().reshape(-1)
            if flat.numel() < 3:
                continue
            t, h, w = int(flat[0].item()), int(flat[1].item()), int(flat[2].item())
            prod = t * h * w
            total += int(prod) // int(max(1, merge_length))
    return max(0, total)


def pick_vision_attn_segment(attn_row: np.ndarray, n_vis: int) -> np.ndarray:
    """
    在完整 key 行 attn_row（长度 K）上截取长度 n_vis 的子向量。
    Qwen 展开后全局图 patch 多为连续块；用「总注意力质量」最大的滑动窗口定位。
    """
    attn_row = np.asarray(attn_row, dtype=np.float32)
    K = int(attn_row.shape[0])
    if K <= 0:
        return attn_row
    if n_vis <= 0:
        return attn_row
    n_vis = min(int(n_vis), K)
    if n_vis == K:
        return attn_row
    best_s, best_m = 0, -1.0
    for s in range(0, K - n_vis + 1):
        m = float(attn_row[s : s + n_vis].sum())
        if m > best_m:
            best_m, best_s = m, s
    return attn_row[best_s : best_s + n_vis]


def vector_to_heatmap(vec: np.ndarray, out_w: int, out_h: int):
    """将一维注意力向量重排为近似矩形网格，再 resize 到图像尺寸（不要求完全平方）。"""
    if vec is None or vec.size == 0:
        return None
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.maximum(vec, 0.0)
    s = float(vec.sum())
    if s > 0:
        vec = vec / s

    n = int(vec.size)
    # h*w >= n，尽量接近方形，避免 n=2,3 时 int(sqrt)==1 导致无法可视化
    h = max(1, int(np.floor(np.sqrt(n))))
    w = int(np.ceil(n / h))
    while h * w < n:
        w += 1
    usable = h * w
    if usable > n:
        vec = np.pad(vec, (0, usable - n), mode="constant", constant_values=0.0)
    else:
        vec = vec[:usable]
    grid = vec.reshape(h, w)
    gmin, gmax = float(grid.min()), float(grid.max())
    if gmax > gmin:
        grid = (grid - gmin) / (gmax - gmin)
    else:
        grid = np.zeros_like(grid, dtype=np.float32)

    hm = Image.fromarray(np.uint8(np.clip(grid * 255.0, 0, 255)), mode="L")
    hm = hm.resize((out_w, out_h), resample=Image.BILINEAR)
    return hm


def overlay_heatmap_on_image(image: Image.Image, heatmap_l: Image.Image, alpha: float = 0.45):
    """将灰度热图叠加到原图，输出 RGB。"""
    base = image.convert("RGB")
    if heatmap_l is None:
        return base
    hm = np.asarray(heatmap_l, dtype=np.uint8)
    r = hm
    g = np.minimum(255, hm * 2).astype(np.uint8)
    b = np.zeros_like(hm, dtype=np.uint8)
    color = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
    return Image.blend(base, color, alpha=max(0.0, min(1.0, float(alpha))))


def render_step_label(image: Image.Image, text: str):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((8, 8, 220, 32), fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return out


def export_dynamic_attention_from_generate(
    generation_output,
    generation_kwargs: dict,
    tokenizer,
    image: Image.Image,
    out_prefix: str,
    max_steps: int = 48,
    overlay_alpha: float = 0.45,
    vision_merge_length: int = 4,
):
    """
    导出逐步动态注意力与静态平均叠图：
    - {out_prefix}_step_XXXX.png
    - {out_prefix}_dynamic.gif
    - {out_prefix}_static_token_attn.png
    """
    step_attn_objs = extract_step_attn_tensors(generation_output)
    if not step_attn_objs:
        return False, "no attentions in generation output"

    inputs = generation_kwargs.get("inputs", None)
    image_grid_thws = generation_kwargs.get("image_grid_thws")
    n_vis_tokens = count_llm_image_tokens_from_grids(image_grid_thws, merge_length=int(vision_merge_length))

    if max_steps is None or int(max_steps) <= 0:
        max_steps = len(step_attn_objs)
    max_steps = min(int(max_steps), len(step_attn_objs))

    frames = []
    step_vecs = []
    img_w, img_h = image.size

    for i in range(max_steps):
        arr = to_numpy_attention(step_attn_objs[i])
        if arr is None:
            continue
        attn_last_q = arr[:, -1, :] if arr.shape[1] > 0 else None
        if attn_last_q is None or attn_last_q.shape[-1] <= 0:
            continue
        attn_token = attn_last_q.mean(axis=0).astype(np.float32, copy=False)

        if n_vis_tokens > 0:
            vec = pick_vision_attn_segment(attn_token, n_vis_tokens)
        else:
            img_token_indices = guess_image_token_indices(inputs, tokenizer)
            if not img_token_indices:
                continue
            valid_indices = [idx for idx in img_token_indices if idx < attn_token.shape[0]]
            if not valid_indices:
                continue
            vec = attn_token[np.array(valid_indices, dtype=np.int64)]

        if vec.size == 0:
            continue
        hm = vector_to_heatmap(vec, out_w=img_w, out_h=img_h)
        if hm is None:
            continue
        step_vecs.append(vec)
        overlay = overlay_heatmap_on_image(image, hm, alpha=overlay_alpha)
        overlay = render_step_label(overlay, f"step {i + 1}")
        overlay.save(f"{out_prefix}_step_{i + 1:04d}.png")
        frames.append(overlay)

    if not frames:
        dbg = ""
        if step_attn_objs:
            a0 = to_numpy_attention(step_attn_objs[0])
            if a0 is not None and a0.ndim >= 2 and a0.shape[-2] > 0:
                K = int(a0.shape[-1])
                dbg = f" (decode_step0 key_len={K}, n_img_tokens_from_grid={n_vis_tokens})"
        return False, "no valid dynamic frames generated" + dbg

    gif_path = f"{out_prefix}_dynamic.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
    )

    static_path = None
    try:
        max_len = min(v.shape[0] for v in step_vecs) if step_vecs else 0
        if max_len > 0:
            stack = np.stack([v[:max_len] for v in step_vecs], axis=0)
            avg_vec = stack.mean(axis=0)
            hm_static = vector_to_heatmap(avg_vec, out_w=img_w, out_h=img_h)
            if hm_static is not None:
                static_overlay = overlay_heatmap_on_image(image, hm_static, alpha=overlay_alpha)
                static_path = f"{out_prefix}_static_token_attn.png"
                static_overlay.save(static_path)
    except Exception:
        static_path = None

    return True, None
