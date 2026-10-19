#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task1 单样本：在 FO1 checkpoint 上复现 mllms_know 的相对注意力 rel_att（LLM 层对全局视觉 token）。

与 infer_attMap_task1.sh / visualize_attMap_task1.sh 使用相同的 JSON、checkpoint、层级融合配置；
不依赖 HuggingFace 官方 Qwen2.5-VL-3B-Instruct，也不安装 mllms_know 独立 transformers。

需要: export VLM_FO1_ATTN_IMPLEMENTATION=eager

输出（默认 attention_infer/task1_mllms_know_fo1/{sample_id}/）:
  {id}_rel_att_heatmap.png   — rel_att 热图叠原图
  {id}_rel_att_bbox.png      — 自适应 bbox（utils/mllms_know_bbox.py）
  {id}_rel_att_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

_HEAVY = None
DEFAULT_GENERAL_PROMPT_ZH = "请简要描述这张图像的整体内容。"
DEFAULT_GENERAL_PROMPT_EN = "Write a general description of the image."
DEFAULT_ATT_LAYER = 22


def _ensure_heavy_deps():
    global _HEAVY
    if _HEAVY is not None:
        return _HEAVY
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from eval_stage2_cot import (
        DEFAULT_VLM_FO1_BASE,
        _is_official_vlm_fo1_model_dir,
        add_task_constraints_to_user_text,
        get_task_type_from_item,
        load_checkpoint_model,
        maybe_enable_hierarchical_fusion,
    )
    from vlm_fo1.mm_utils import prepare_inputs
    from utils.mllms_know_bbox import bbox_from_att_image_adaptive
    from utils.spatial_token_attention import (
        count_llm_image_tokens_from_grids,
        overlay_heatmap_on_image,
        pick_vision_attn_segment,
        vector_to_heatmap,
    )

    _HEAVY = {
        "np": np,
        "torch": torch,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "DEFAULT_VLM_FO1_BASE": DEFAULT_VLM_FO1_BASE,
        "_is_official_vlm_fo1_model_dir": _is_official_vlm_fo1_model_dir,
        "add_task_constraints_to_user_text": add_task_constraints_to_user_text,
        "get_task_type_from_item": get_task_type_from_item,
        "load_checkpoint_model": load_checkpoint_model,
        "maybe_enable_hierarchical_fusion": maybe_enable_hierarchical_fusion,
        "prepare_inputs": prepare_inputs,
        "count_llm_image_tokens_from_grids": count_llm_image_tokens_from_grids,
        "pick_vision_attn_segment": pick_vision_attn_segment,
        "overlay_heatmap_on_image": overlay_heatmap_on_image,
        "vector_to_heatmap": vector_to_heatmap,
        "bbox_from_att_image_adaptive": bbox_from_att_image_adaptive,
    }
    return _HEAVY


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


def _list_task1_ids(data: list, limit: int) -> None:
    n = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id", None)
        try:
            if tid is None or int(tid) != 1:
                continue
        except (TypeError, ValueError):
            continue
        print(item.get("id", ""))
        n += 1
        if limit > 0 and n >= limit:
            break
    print(f"(共列出 {n} 条，limit={limit})")


def _att_shape_from_grid(image_grid_thws) -> tuple[int, int]:
    """与 mllms_know qwen2_5_methods: (image_grid_thw[1:] / 2) 一致（spatial_merge_size=2）。"""
    if not image_grid_thws:
        return 1, 1
    g = image_grid_thws[0]
    if hasattr(g, "detach"):
        flat = g.detach().long().reshape(-1)
    else:
        flat = g.reshape(-1)
    if flat.numel() < 3:
        n = max(1, int(flat.numel()))
        side = max(1, int(round(n**0.5)))
        return side, max(1, n // side)
    _t, h, w = int(flat[0]), int(flat[1]), int(flat[2])
    return max(1, h // 2), max(1, w // 2)


def _forward_attentions(model, generation_kwargs: dict, torch):
    fwd = dict(
        input_ids=generation_kwargs["inputs"],
        images=generation_kwargs.get("images"),
        images_aux=generation_kwargs.get("images_aux"),
        image_grid_thws=generation_kwargs.get("image_grid_thws"),
        bbox_list=generation_kwargs.get("bbox_list"),
        task_ids=generation_kwargs.get("task_ids"),
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    with torch.no_grad():
        return model(**fwd)


def rel_attention_fo1(
    model,
    question_kwargs: dict,
    general_kwargs: dict,
    att_layer: int,
    vision_merge_length: int,
    torch,
    np,
    count_llm_image_tokens_from_grids,
    pick_vision_attn_segment,
):
    n_vis = count_llm_image_tokens_from_grids(
        question_kwargs.get("image_grid_thws"),
        merge_length=int(vision_merge_length),
    )

    out_q = _forward_attentions(model, question_kwargs, torch)
    if not out_q.attentions or att_layer >= len(out_q.attentions):
        raise RuntimeError(
            f"问题前向未返回 attentions 或层索引越界: layer={att_layer}, "
            f"n_layers={0 if not out_q.attentions else len(out_q.attentions)}；"
            "请设置 VLM_FO1_ATTN_IMPLEMENTATION=eager。"
        )
    # 右填充时 prompt 末 token 为序列最后一位（与 mllms_know 的 -1 一致）
    q_idx = int(out_q.attentions[att_layer].shape[-2]) - 1
    row_q = (
        out_q.attentions[att_layer][0, :, q_idx, :]
        .mean(dim=0)
        .float()
        .detach()
        .cpu()
        .numpy()
    )
    del out_q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_g = _forward_attentions(model, general_kwargs, torch)
    row_g = (
        out_g.attentions[att_layer][0, :, q_idx, :]
        .mean(dim=0)
        .float()
        .detach()
        .cpu()
        .numpy()
    )
    del out_g
    att = pick_vision_attn_segment(row_q, n_vis)
    general_att = pick_vision_attn_segment(row_g, n_vis)
    general_att = np.maximum(general_att, 1e-8)
    rel = att / general_att

    h_llm, w_llm = _att_shape_from_grid(question_kwargs.get("image_grid_thws"))
    if rel.size != h_llm * w_llm:
        n = rel.size
        h_llm = max(1, int(np.floor(np.sqrt(n))))
        w_llm = max(1, int(np.ceil(n / h_llm)))
        while h_llm * w_llm < n:
            w_llm += 1
    rel_map = rel.reshape(h_llm, w_llm)
    return rel_map, {
        "att_layer": att_layer,
        "query_token_index": int(q_idx),
        "n_vision_tokens": int(rel.size),
        "att_shape": [int(h_llm), int(w_llm)],
        "vision_locator": "pick_vision_attn_segment",
    }


def _normalize_map_for_display(att_map, np):
    m = np.asarray(att_map, dtype=np.float32)
    m = np.maximum(m, 0.0)
    lo, hi = np.percentile(m, 5), np.percentile(m, 95)
    if hi > lo:
        m = (m - lo) / (hi - lo)
    else:
        m = np.zeros_like(m)
    return np.clip(m, 0.0, 1.0)


def _draw_bbox(image, bbox, ImageDraw):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
    return out


def _run_one_sample(
    *,
    model,
    tokenizer,
    image_processors,
    model_name: str,
    device: str,
    path_feature_root: str | None,
    language: str,
    data_item: dict,
    out_dir: str,
    general_prompt: str,
    rel_prompt: str | None,
    att_layer: int,
    vision_merge_length: int,
    bbox_size: int,
    overlay_alpha: float,
) -> dict:
    h = _ensure_heavy_deps()
    np = h["np"]
    torch = h["torch"]
    Image = h["Image"]
    ImageDraw = h["ImageDraw"]
    get_task_type_from_item = h["get_task_type_from_item"]
    add_task_constraints_to_user_text = h["add_task_constraints_to_user_text"]
    prepare_inputs = h["prepare_inputs"]
    overlay_heatmap_on_image = h["overlay_heatmap_on_image"]
    vector_to_heatmap = h["vector_to_heatmap"]
    bbox_from_att_image_adaptive = h["bbox_from_att_image_adaptive"]

    os.makedirs(out_dir, exist_ok=True)
    sid = str(data_item.get("id", "unknown"))
    safe = _sanitize_filename(sid)

    image_path = data_item["image"]
    image = Image.open(image_path).convert("RGB")
    user_text, assistant_text = _parse_user_assistant(data_item)
    if not user_text:
        raise ValueError(f"样本 {sid} 缺少 user 对话")

    task_type = get_task_type_from_item(data_item)
    if task_type != "type1":
        raise ValueError(f"样本 {sid} 的 task_type={task_type}，本脚本仅支持 task_id=1 (type1)")

    raw_task_id = data_item.get("task_id", None)
    try:
        effective_task_id = int(raw_task_id) if raw_task_id is not None else 1
    except (TypeError, ValueError):
        effective_task_id = 1

    user_text = add_task_constraints_to_user_text(user_text, effective_task_id, language=language)
    question_text = rel_prompt if rel_prompt else user_text

    gt_regions = data_item.get("regions", [])
    bbox_list_for_model = [r["bbox"] for r in gt_regions]

    def _messages(text: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": text},
                ],
                "bbox_list": bbox_list_for_model,
            }
        ]

    common = dict(
        model_name=model_name,
        model=model,
        image_processors=image_processors,
        tokenizer=tokenizer,
        device=device,
        max_tokens=1,
        temperature=0.0,
        do_sample=False,
        path_feature_root=path_feature_root,
        task_ids=[effective_task_id],
        verbose=False,
    )
    q_kwargs = prepare_inputs(messages=_messages(question_text), **common)
    g_kwargs = prepare_inputs(messages=_messages(general_prompt), **common)

    model.eval()
    rel_map, dbg = rel_attention_fo1(
        model,
        q_kwargs,
        g_kwargs,
        att_layer=att_layer,
        vision_merge_length=vision_merge_length,
        torch=torch,
        np=np,
        count_llm_image_tokens_from_grids=h["count_llm_image_tokens_from_grids"],
        pick_vision_attn_segment=h["pick_vision_attn_segment"],
    )

    bbox = bbox_from_att_image_adaptive(rel_map, image.size, bbox_size=bbox_size)
    disp = _normalize_map_for_display(rel_map, np)
    vec = disp.reshape(-1)
    hm = vector_to_heatmap(vec, out_w=image.size[0], out_h=image.size[1])
    heat_overlay = overlay_heatmap_on_image(image, hm, alpha=overlay_alpha)
    bbox_overlay = _draw_bbox(heat_overlay, bbox, ImageDraw)

    heat_path = os.path.join(out_dir, f"{safe}_rel_att_heatmap.png")
    bbox_path = os.path.join(out_dir, f"{safe}_rel_att_bbox.png")
    heat_overlay.save(heat_path)
    bbox_overlay.save(bbox_path)

    meta = {
        "id": sid,
        "image": image_path,
        "method": "mllms_know_rel_att_fo1",
        "question_prompt": question_text,
        "general_prompt": general_prompt,
        "user_text_with_constraints": user_text,
        "ground_truth": assistant_text,
        "rel_att_heatmap_png": heat_path,
        "rel_att_bbox_png": bbox_path,
        "adaptive_bbox_xyxy": list(bbox),
        "bbox_size_for_adaptive": bbox_size,
        "attn_implementation_env": os.environ.get("VLM_FO1_ATTN_IMPLEMENTATION"),
        **dbg,
    }
    meta_path = os.path.join(out_dir, f"{safe}_rel_att_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {sid} rel_att 热图: {heat_path}")
    print(f"     bbox 叠图: {bbox_path}")
    print(f"     meta: {meta_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="FO1 checkpoint + mllms_know 相对注意力 rel_att（Task1 单样本）"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
    )
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="outputs/stage1_layerGated_taskAdaptive")
    parser.add_argument("--checkpoint", type=str, default="outputs/stage2_layerGated_taskAdaptive")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_feature_root", type=str, default=None)
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--output_dir", type=str, default="attention_infer/task1_mllms_know_fo1")
    parser.add_argument("--att_layer", type=int, default=DEFAULT_ATT_LAYER)
    parser.add_argument(
        "--vision_merge_length",
        type=int,
        default=4,
        help="视觉 token 数 = grid.prod() // merge_length",
    )
    parser.add_argument(
        "--general_prompt",
        type=str,
        default=None,
        help="通用描述 prompt；默认按 --language 选中英文/中文模板",
    )
    parser.add_argument(
        "--rel_prompt",
        type=str,
        default=None,
        help="用于 rel_att 的问题文本；默认使用带 task 约束的 user 文本",
    )
    parser.add_argument("--bbox_size", type=int, default=224, help="mllms_know qwen2_5 默认 224")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--use_hierarchical_fusion", action="store_true")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="task_adaptive",
        choices=["legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"],
    )
    parser.add_argument("--list_task1_ids", action="store_true")
    parser.add_argument("--list_limit", type=int, default=50)

    args = parser.parse_args()
    data = _load_json_list(args.test_json)
    if args.list_task1_ids:
        _list_task1_ids(data, args.list_limit)
        return

    if not args.sample_id or not str(args.sample_id).strip():
        parser.error("请指定 --sample_id，或使用 --list_task1_ids")

    general_prompt = args.general_prompt
    if not general_prompt:
        general_prompt = (
            DEFAULT_GENERAL_PROMPT_ZH if args.language == "zh" else DEFAULT_GENERAL_PROMPT_EN
        )

    attn_impl = os.environ.get("VLM_FO1_ATTN_IMPLEMENTATION", "sdpa")
    if attn_impl != "eager":
        print(
            f"[Warn] VLM_FO1_ATTN_IMPLEMENTATION={attn_impl!r}，rel_att 需要 eager；"
            "建议: export VLM_FO1_ATTN_IMPLEMENTATION=eager"
        )

    sample_ids = [s.strip() for s in str(args.sample_id).split(",") if s.strip()]
    hp = _ensure_heavy_deps()
    load_checkpoint_model = hp["load_checkpoint_model"]
    maybe_enable_hierarchical_fusion = hp["maybe_enable_hierarchical_fusion"]
    _is_official_vlm_fo1_model_dir = hp["_is_official_vlm_fo1_model_dir"]
    DEFAULT_VLM_FO1_BASE = hp["DEFAULT_VLM_FO1_BASE"]

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
        args.base_model
        if args.base_model and _is_official_vlm_fo1_model_dir(args.base_model)
        else DEFAULT_VLM_FO1_BASE
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []
    for sid in sample_ids:
        matches = _find_task1_items(data, sid)
        if not matches:
            raise SystemExit(
                f"未找到 task_id=1 且 id={sid!r} 的样本。"
                f"提示: python {os.path.basename(__file__)} --list_task1_ids"
            )
        for json_index, item in matches:
            item = dict(item)
            item["_json_index"] = json_index
            sub_out = os.path.join(args.output_dir, _sanitize_filename(sid))
            summaries.append(
                _run_one_sample(
                    model=model,
                    tokenizer=tokenizer,
                    image_processors=image_processors,
                    model_name=model_name_for_template,
                    device=args.device,
                    path_feature_root=args.path_feature_root,
                    language=args.language,
                    data_item=item,
                    out_dir=sub_out,
                    general_prompt=general_prompt,
                    rel_prompt=args.rel_prompt,
                    att_layer=args.att_layer,
                    vision_merge_length=args.vision_merge_length,
                    bbox_size=args.bbox_size,
                    overlay_alpha=args.overlay_alpha,
                )
            )

    batch_path = os.path.join(args.output_dir, "batch_rel_att_meta.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"批量 meta: {batch_path}")


if __name__ == "__main__":
    main()
