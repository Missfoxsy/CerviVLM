#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Probe intermediate features in VLM-FO1 pipeline.

This script:
1) Loads model/checkpoint with the same logic as eval_stage2_cot.py
2) Loads one sample from dataset json (must contain image + regions)
3) Registers forward hooks on key modules
4) Runs one short generation pass
5) Prints and saves feature shape summary
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_stage2_cot import add_task_constraints_to_user_text, load_checkpoint_model  # noqa: E402
from vlm_fo1.mm_utils import prepare_inputs  # noqa: E402


def _shape_tree(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    if isinstance(x, (list, tuple)):
        return [_shape_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_tree(v) for k, v in x.items()}
    return str(type(x).__name__)


def _extract_user_text(item: dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            t = conv.get("value", "").replace("<image>", "").strip()
            if t.startswith("\n"):
                t = t[1:]
            return t
    return ""


def _pick_sample(data: List[dict], task_id: int) -> dict:
    for item in data:
        try:
            if int(item.get("task_id", -1)) != int(task_id):
                continue
        except Exception:
            continue
        regions = item.get("regions", [])
        bboxes = [r.get("bbox") for r in regions if isinstance(r.get("bbox"), list) and len(r.get("bbox")) == 4]
        if item.get("image") and len(bboxes) > 0:
            return item
    raise RuntimeError(f"No valid sample found for task_id={task_id}")


def main():
    parser = argparse.ArgumentParser(description="Probe VLM-FO1 intermediate features")
    parser.add_argument("--base_model", type=str, default="outputs/cervical_stage1_align_lora")
    parser.add_argument("--checkpoint", type=str, default="outputs/stage2_cot_balance")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
    )
    parser.add_argument("--task_id", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--output_json", type=str, default="eval_result/probe/vlmfo1_probe_summary.json")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_json):
        raise FileNotFoundError(f"dataset_json not found: {args.dataset_json}")

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("dataset_json is empty or invalid")

    print(f"[Probe] loading model base={args.base_model} checkpoint={args.checkpoint} device={args.device}")
    tokenizer, model, image_processors = load_checkpoint_model(args.base_model, args.checkpoint, device=args.device)
    model.eval()

    sample = _pick_sample(data, args.task_id)
    image_path = sample["image"]
    user_text = _extract_user_text(sample)
    user_text = add_task_constraints_to_user_text(user_text, int(sample.get("task_id", -1)), args.language)
    bbox_list = [r["bbox"] for r in sample.get("regions", []) if isinstance(r.get("bbox"), list) and len(r.get("bbox")) == 4]

    print(f"[Probe] sample id={sample.get('id')} task_id={sample.get('task_id')} image={image_path}")
    print(f"[Probe] bbox_count={len(bbox_list)}")

    hook_cache: Dict[str, Any] = {}
    hook_handles = []

    def _reg(mod, name: str):
        if mod is None:
            return

        def _hook(_m, _inp, out):
            hook_cache[name] = _shape_tree(out)

        hook_handles.append(mod.register_forward_hook(_hook))

    core = model.get_model() if hasattr(model, "get_model") else model
    _reg(core.get_vision_tower() if hasattr(core, "get_vision_tower") else None, "vision_tower_out")
    _reg(core.get_vision_tower_aux() if hasattr(core, "get_vision_tower_aux") else None, "vision_tower_aux_out")
    _reg(getattr(core, "object_vp_extractor", None), "object_vp_extractor_out")
    _reg(getattr(core, "mm_projector", None), "mm_projector_out")
    _reg(getattr(core, "mm_projector_aux", None), "mm_projector_aux_out")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": user_text},
        ],
        "bbox_list": bbox_list,
    }]

    model_name_for_template = args.base_model if "vlm-fo1" in os.path.basename(os.path.normpath(args.base_model)).lower() \
        else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "VLM-FO1_Qwen2.5-VL-3B-v01")

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
    )

    prep_summary = {
        "inputs": _shape_tree(gen_kwargs.get("inputs")),
        "images": _shape_tree(gen_kwargs.get("images")),
        "images_aux": _shape_tree(gen_kwargs.get("images_aux")),
        "image_grid_thws": _shape_tree(gen_kwargs.get("image_grid_thws")),
        "bbox_list": _shape_tree(gen_kwargs.get("bbox_list")),
    }

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    output_text = tokenizer.decode(
        output_ids[0, gen_kwargs["inputs"].shape[1]:],
        skip_special_tokens=False,
    ).strip()

    for h in hook_handles:
        h.remove()

    summary = {
        "sample_meta": {
            "id": sample.get("id"),
            "task_id": sample.get("task_id"),
            "image": image_path,
            "bbox_count": len(bbox_list),
        },
        "prepare_inputs_summary": prep_summary,
        "hook_summary": hook_cache,
        "output_text_preview": output_text[:500],
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Probe Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print("===================================")
    print(f"[Probe] saved: {args.output_json}")


if __name__ == "__main__":
    main()

