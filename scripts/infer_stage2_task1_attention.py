#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 JSON 中的样本 id 对 task_id=1（type1 特征定位）做单条推理，并可选导出：
- 层级 ROI 融合注意力（--use_hierarchical_fusion，HFRE 内部 MHA）
- 叠在原图上的空间注意力（--save_spatial_attn：解码步对视觉 token 的交叉注意力，与 eval_stage2_attVisual 一致）

用法示例：
  python scripts/infer_stage2_task1_attention.py \\
    --test_json /home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json \\
    --sample_id identity_2919 \\
    --base_model outputs/stage1_layerGated_taskAdaptive \\
    --checkpoint outputs/stage2_layerGated_taskAdaptive \\
    --use_hierarchical_fusion \\
    --hierarchical_fusion_mode task_adaptive \\
    --save_spatial_attn \\
    --output_dir attention_infer/task1_single
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

# 重型依赖（transformers / vlm_fo1）延迟到首次推理时加载，便于仅使用 --list_task1_ids 浏览 id。
_HEAVY = None


def _ensure_heavy_deps():
    """导入 eval / torch / PIL 等；成功后在模块级缓存。"""
    global _HEAVY
    if _HEAVY is not None:
        return _HEAVY
    import torch
    from PIL import Image

    from eval_stage2_cot import (
        DEFAULT_VLM_FO1_BASE,
        _is_official_vlm_fo1_model_dir,
        _unwrap_hierarchical_fusion_module,
        add_task_constraints_to_user_text,
        get_task_type_from_item,
        load_checkpoint_model,
        maybe_enable_hierarchical_fusion,
    )
    from vlm_fo1.mm_utils import draw_bboxes_and_save, prepare_inputs
    from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import (
        visualize_hierarchical_attention,
    )
    from utils.data_utils import parse_grounding_output

    _HEAVY = {
        "torch": torch,
        "Image": Image,
        "DEFAULT_VLM_FO1_BASE": DEFAULT_VLM_FO1_BASE,
        "_is_official_vlm_fo1_model_dir": _is_official_vlm_fo1_model_dir,
        "_unwrap_hierarchical_fusion_module": _unwrap_hierarchical_fusion_module,
        "add_task_constraints_to_user_text": add_task_constraints_to_user_text,
        "get_task_type_from_item": get_task_type_from_item,
        "load_checkpoint_model": load_checkpoint_model,
        "maybe_enable_hierarchical_fusion": maybe_enable_hierarchical_fusion,
        "prepare_inputs": prepare_inputs,
        "draw_bboxes_and_save": draw_bboxes_and_save,
        "visualize_hierarchical_attention": visualize_hierarchical_attention,
        "parse_grounding_output": parse_grounding_output,
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
    """返回 (json_index, item) 列表，task_id=1 且 id 匹配。"""
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
    max_tokens: int,
    save_bbox_overlay: bool,
    hierarchical_enabled: bool,
    save_spatial_attn: bool = False,
    spatial_attn_max_steps: int = 48,
    spatial_overlay_alpha: float = 0.45,
    spatial_vision_merge_length: int = 4,
) -> dict:
    h = _ensure_heavy_deps()
    torch = h["torch"]
    Image = h["Image"]
    get_task_type_from_item = h["get_task_type_from_item"]
    add_task_constraints_to_user_text = h["add_task_constraints_to_user_text"]
    _unwrap_hierarchical_fusion_module = h["_unwrap_hierarchical_fusion_module"]
    prepare_inputs = h["prepare_inputs"]
    draw_bboxes_and_save = h["draw_bboxes_and_save"]
    visualize_hierarchical_attention = h["visualize_hierarchical_attention"]
    parse_grounding_output = h["parse_grounding_output"]

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

    core = model.get_model() if hasattr(model, "get_model") else model
    extractor = getattr(core, "object_vp_extractor", None)
    hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
    hf_inner = _unwrap_hierarchical_fusion_module(hf)
    if hierarchical_enabled and hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
        hf_inner.last_gates = None
        hf_inner.last_gates_raw = None

    gt_regions = data_item.get("regions", [])
    bbox_list_for_model = [r["bbox"] for r in gt_regions]
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

    generation_kwargs = prepare_inputs(
        model_name=model_name,
        model=model,
        image_processors=image_processors,
        tokenizer=tokenizer,
        messages=messages,
        device=device,
        max_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        path_feature_root=path_feature_root,
        task_ids=[effective_task_id],
        verbose=False,
    )
    generation_kwargs.pop("streamer", None)

    spatial_prefix = os.path.join(out_dir, f"{safe}_spatial_cross_attn")
    spatial_meta: dict = {
        "spatial_static_token_attn_png": None,
        "spatial_dynamic_gif": None,
        "spatial_export_error": None,
    }

    model.eval()
    with torch.no_grad():
        if save_spatial_attn:
            try:
                gen_out = model.generate(
                    **generation_kwargs,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                output_ids = gen_out.sequences
                from utils.spatial_token_attention import export_dynamic_attention_from_generate

                ok, err = export_dynamic_attention_from_generate(
                    gen_out,
                    generation_kwargs,
                    tokenizer,
                    image,
                    spatial_prefix,
                    max_steps=spatial_attn_max_steps,
                    overlay_alpha=spatial_overlay_alpha,
                    vision_merge_length=int(spatial_vision_merge_length),
                )
                if ok:
                    st = f"{spatial_prefix}_static_token_attn.png"
                    gf = f"{spatial_prefix}_dynamic.gif"
                    if os.path.isfile(st):
                        spatial_meta["spatial_static_token_attn_png"] = st
                    if os.path.isfile(gf):
                        spatial_meta["spatial_dynamic_gif"] = gf
                else:
                    spatial_meta["spatial_export_error"] = err
                    print(f"[Warn] 空间注意力未写出: {err}")
            except TypeError as e:
                spatial_meta["spatial_export_error"] = f"TypeError: {e}"
                print(f"[Warn] 当前模型不支持 output_attentions 导出，已降级为普通 generate: {e}")
                output_ids = model.generate(**generation_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    spatial_meta["spatial_export_error"] = f"OOM: {e}"
                    print("[Warn] 空间注意力推理 OOM，清空缓存后降级为普通 generate。")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    output_ids = model.generate(**generation_kwargs)
                else:
                    raise
        else:
            output_ids = model.generate(**generation_kwargs)

    inp = generation_kwargs["inputs"]
    output_text = tokenizer.decode(
        output_ids[0, inp.shape[1] :],
        skip_special_tokens=False,
    ).strip()

    attn_path = os.path.join(out_dir, f"{safe}_hierarchical_attn.png")
    if hierarchical_enabled and hf_inner is not None and getattr(extractor, "use_hierarchical_fusion", False):
        visualize_hierarchical_attention(hf_inner, save_path=attn_path)
    else:
        attn_path = ""
        print(
            "[Warn] 未导出层级注意力：请使用 --use_hierarchical_fusion，并确保训练/评估时同样启用了层级融合。"
        )

    gates_mean = None
    if hf_inner is not None and getattr(hf_inner, "last_gates", None) is not None:
        g = hf_inner.last_gates.detach().float().cpu()
        if g.numel() > 0:
            gates_mean = g.mean(dim=0).tolist()

    overlay_path = ""
    if save_bbox_overlay:
        pred_dict = parse_grounding_output(output_text)
        region_id_to_bbox = {}
        for r in gt_regions:
            rid = r.get("region_id", "")
            mat = re.match(r"region(\d+)", str(rid))
            if mat:
                region_id_to_bbox[int(mat.group(1))] = r["bbox"]
        pred_indices = set()
        for _, indices in pred_dict.items():
            pred_indices.update(indices)
        pred_bboxes = [region_id_to_bbox[i] for i in sorted(pred_indices) if i in region_id_to_bbox]
        fo1_bboxes = {"positive cervical cell": pred_bboxes} if pred_bboxes else {}
        overlay_path = os.path.join(out_dir, f"{safe}_pred_boxes.jpg")
        draw_bboxes_and_save(image=image, fo1_bboxes=fo1_bboxes, output_path=overlay_path, color="red")

    summary = {
        "id": sid,
        "image": image_path,
        "json_index_hint": data_item.get("_json_index"),
        "user_text": user_text,
        "ground_truth": assistant_text,
        "prediction": output_text,
        "hierarchical_attn_png": attn_path or None,
        "pred_bbox_overlay": overlay_path or None,
        "gate_mean_per_level": gates_mean,
        **spatial_meta,
    }
    meta_path = os.path.join(out_dir, f"{safe}_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] 样本 {sid} 推理完成。摘要: {meta_path}")
    if attn_path:
        print(f"      层级融合注意力: {attn_path}")
    if spatial_meta.get("spatial_static_token_attn_png"):
        print(f"      空间注意力(静态叠图): {spatial_meta['spatial_static_token_attn_png']}")
    if spatial_meta.get("spatial_dynamic_gif"):
        print(f"      空间注意力(动图): {spatial_meta['spatial_dynamic_gif']}")
    if overlay_path:
        print(f"      预测框叠加: {overlay_path}")
    return summary


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


def main():
    parser = argparse.ArgumentParser(description="Task1 单样本推理 + 层级注意力可视化（按 JSON id 选择）")
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
        help="与 eval_stage2_cot 相同格式的 JSON 列表",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="数据项中的 id 字段；多个用英文逗号分隔，如 identity_2919,identity_123",
    )
    parser.add_argument("--base_model", type=str, default="outputs/stage1_layerGated_taskAdaptive")
    parser.add_argument("--checkpoint", type=str, default="outputs/stage2_layerGated_taskAdaptive")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_feature_root", type=str, default=None)
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--output_dir", type=str, default="attention_infer/task1_by_id")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--save_bbox_overlay", action="store_true", help="根据预测解析 region 并在原图上画框")
    parser.add_argument("--use_hierarchical_fusion", action="store_true")
    parser.add_argument("--hierarchical_num_heads", type=int, default=8)
    parser.add_argument(
        "--hierarchical_fusion_mode",
        type=str,
        default="task_adaptive",
        choices=["legacy", "task_adaptive", "gated"],
    )
    parser.add_argument(
        "--list_task1_ids",
        action="store_true",
        help="仅列出数据集中 task_id=1 的 id（配合 --list_limit）",
    )
    parser.add_argument(
        "--save_spatial_attn",
        action="store_true",
        help="导出解码时对视觉 token 的交叉注意力，叠到原图（*_static_token_attn.png / *_dynamic.gif / step 帧）",
    )
    parser.add_argument("--spatial_attn_max_steps", type=int, default=48, help="空间注意力最多保留的 decode 步数")
    parser.add_argument("--spatial_overlay_alpha", type=float, default=0.45, help="热图与原图 blend 权重")
    parser.add_argument(
        "--spatial_vision_merge_length",
        type=int,
        default=4,
        help="视觉 token 数 = grid.prod() // merge_length；与 Qwen2.5-VL spatial_merge_size=2 对应为 4",
    )
    parser.add_argument("--list_limit", type=int, default=50, help="--list_task1_ids 时最多打印条数，0 表示不限制")

    args = parser.parse_args()

    data = _load_json_list(args.test_json)
    if args.list_task1_ids:
        _list_task1_ids(data, args.list_limit)
        return

    if not args.sample_id or not str(args.sample_id).strip():
        parser.error("请指定 --sample_id，或使用 --list_task1_ids 查看可用 id")

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
    all_summaries = []

    for sid in sample_ids:
        matches = _find_task1_items(data, sid)
        if not matches:
            task1_ids = [
                str(it.get("id"))
                for it in data
                if isinstance(it, dict)
                and it.get("task_id") is not None
                and str(it.get("task_id")) == "1"
            ]
            hint = ""
            if sid not in task1_ids and any(str(it.get("id")) == sid for it in data if isinstance(it, dict)):
                hint = f"（存在 id={sid} 但 task_id 不是 1）"
            elif sid not in [str(it.get("id")) for it in data if isinstance(it, dict)]:
                hint = "（JSON 中无此 id）"
            raise SystemExit(
                f"未找到 task_id=1 且 id={sid!r} 的样本。{hint}\n"
                f"提示: python {os.path.basename(__file__)} --test_json ... --list_task1_ids"
            )

        for json_index, item in matches:
            item = dict(item)
            item["_json_index"] = json_index
            sub_out = os.path.join(args.output_dir, _sanitize_filename(sid))
            summary = _run_one_sample(
                model=model,
                tokenizer=tokenizer,
                image_processors=image_processors,
                model_name=model_name_for_template,
                device=args.device,
                path_feature_root=args.path_feature_root,
                language=args.language,
                data_item=item,
                out_dir=sub_out,
                max_tokens=args.max_tokens,
                save_bbox_overlay=args.save_bbox_overlay,
                hierarchical_enabled=args.use_hierarchical_fusion,
                save_spatial_attn=args.save_spatial_attn,
                spatial_attn_max_steps=args.spatial_attn_max_steps,
                spatial_overlay_alpha=args.spatial_overlay_alpha,
                spatial_vision_merge_length=args.spatial_vision_merge_length,
            )
            all_summaries.append(summary)

    batch_path = os.path.join(args.output_dir, "batch_summaries.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"批量摘要: {batch_path}")


if __name__ == "__main__":
    main()