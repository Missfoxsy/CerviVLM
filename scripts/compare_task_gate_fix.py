#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 task_ids 在 generate() 中透传修复前后，type0 vs type2 的层级门控差异。

核心对比（同一张图 / 同一 ROI，仅改 task_id）：
  - 修复前：task_ids 被丢弃 -> 无论传 0 还是 2，门控相同（均走 type1 默认）
  - 修复后：传 0 vs 2 -> 门控应明显不同（task_level_bias 生效）

辅助对比（按 JSON 分 type0/type2 桶聚合）：
  - 修复前：两桶 mean 往往接近（此前 gate_stats.json 的现象）
  - 修复后：type0 偏粗层、type2 偏细层

用法：
  CUDA_VISIBLE_DEVICES=6 python scripts/compare_task_gate_fix.py

  python scripts/compare_task_gate_fix.py \\
    --base_model outputs/stage1_layerGated_taskAdaptive \\
    --checkpoint outputs/stage2_layerGated_taskAdaptive \\
    --num_per_task 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_stage2_cot import (  # noqa: E402
    DEFAULT_VLM_FO1_BASE,
    _unwrap_hierarchical_fusion_module,
    add_task_constraints_to_user_text,
    get_task_type_from_item,
    load_checkpoint_model,
)
from vlm_fo1.mm_utils import prepare_inputs  # noqa: E402
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import (  # noqa: E402
    HierarchicalFeatureFusion,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Compare hierarchical gates before/after task_ids fix")
    p.add_argument("--base_model", type=str, default="outputs/stage1_layerGated_taskAdaptive")
    p.add_argument("--checkpoint", type=str, default="outputs/stage2_layerGated_taskAdaptive")
    p.add_argument(
        "--test_json",
        type=str,
        default="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--language", type=str, default="zh", choices=["zh", "en"])
    p.add_argument("--num_per_task", type=int, default=16, help="type0 / type2 各采样条数（分桶聚合）")
    p.add_argument("--num_same_sample", type=int, default=4, help="同图双 task_id 对比的样本数")
    p.add_argument("--max_new_tokens", type=int, default=4, help="generate 步数尽量少，仅触发 region 编码")
    p.add_argument("--hierarchical_fusion_mode", type=str, default="task_adaptive")
    p.add_argument("--output_json", type=str, default="eval_result/task_gate_fix_compare.json")
    return p.parse_args()


def _load_samples(test_json: str, num_per_task: int):
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    buckets = {"type0": [], "type2": []}
    for idx, item in enumerate(data):
        t = get_task_type_from_item(item)
        if t in buckets and len(buckets[t]) < num_per_task:
            buckets[t].append((idx, item))
        if all(len(v) >= num_per_task for v in buckets.values()):
            break
    for t in buckets:
        if len(buckets[t]) < num_per_task:
            raise RuntimeError(f"测试集 {test_json} 中 {t} 样本不足 {num_per_task} 条（仅找到 {len(buckets[t])}）")
    return buckets


def _effective_task_id(item: dict) -> int:
    raw = item.get("task_id", None)
    try:
        if raw is not None:
            return int(raw)
    except (TypeError, ValueError):
        pass
    return -1


def _gate_mean_str(mean: list) -> str:
    if not mean:
        return "(empty)"
    return " ".join(f"L{i}={mean[i]:.4f}" for i in range(len(mean)))


def _summarize(rows: list) -> dict:
    if not rows:
        return {"count": 0, "mean": [], "std": []}
    arr = np.asarray(rows, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).round(6).tolist(),
        "std": arr.std(axis=0).round(6).tolist(),
    }


def _pair_diff(a: dict, b: dict) -> list:
    if not a.get("mean") or not b.get("mean"):
        return []
    return [round(float(b["mean"][i]) - float(a["mean"][i]), 6) for i in range(len(a["mean"]))]


def _l2(v: list) -> float:
    return float(np.linalg.norm(v)) if v else 0.0


def _patch_prepare_drop_task_ids(model):
    """模拟修复前：generate() 不将 task_ids 传入 forward（旧版行为）。"""
    orig = model.prepare_inputs_for_generation

    def patched(self, *args, **kwargs):
        out = orig(*args, **kwargs)
        out.pop("task_ids", None)
        return out

    model.prepare_inputs_for_generation = types.MethodType(patched, model)
    return orig


def _patch_hierarchical_fusion_ignore_task_type():
    """
    模拟修复前：门控恒按 type1 条件计算（与 task_ids 未传入 forward 等价）。
    """
    orig = HierarchicalFeatureFusion.forward

    def patched(self, roi_feats_per_level, task_type="type1"):
        return orig(self, roi_feats_per_level, task_type="type1")

    HierarchicalFeatureFusion.forward = patched
    return orig


def _restore_hierarchical_fusion_forward(orig):
    HierarchicalFeatureFusion.forward = orig


def _restore_prepare(model, orig):
    model.prepare_inputs_for_generation = orig


def _run_one_sample(model, tokenizer, image_processors, item, device, language, max_new_tokens, task_id_override: int):
    """对单条样本 generate 一次，返回 ROI-level gate 行列表。"""
    core = model.get_model() if hasattr(model, "get_model") else model
    extractor = getattr(core, "object_vp_extractor", None)
    hf = getattr(extractor, "hierarchical_fusion", None) if extractor is not None else None
    hf_inner = _unwrap_hierarchical_fusion_module(hf)

    image_path = item["image"]
    _ = Image.open(image_path).convert("RGB")

    user_text = None
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            user_text = conv.get("value", "").replace("<image>", "").strip()
            if user_text.startswith("\n"):
                user_text = user_text[1:]
    if not user_text:
        return []

    user_text = add_task_constraints_to_user_text(user_text, task_id_override, language=language)
    gt_regions = item.get("regions", [])
    bbox_list_for_model = [r["bbox"] for r in gt_regions]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": user_text},
        ],
        "bbox_list": bbox_list_for_model,
    }]

    if hf_inner is not None:
        hf_inner.last_gates = None
        hf_inner.last_gates_raw = None

    gen_kw = prepare_inputs(
        model_name=DEFAULT_VLM_FO1_BASE,
        model=model,
        image_processors=image_processors,
        tokenizer=tokenizer,
        messages=messages,
        device=device,
        max_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        task_ids=[task_id_override],
        verbose=False,
    )
    _ = model.generate(**gen_kw)

    gates = getattr(hf_inner, "last_gates", None) if hf_inner is not None else None
    if gates is None or gates.numel() == 0:
        return []
    return gates.detach().float().cpu().tolist()


def _collect_bucket_gates(model, tokenizer, image_processors, samples, device, language, max_new_tokens):
    by_task = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for idx, item in tqdm(samples, desc="bucket-forward", leave=False):
            task_type = get_task_type_from_item(item)
            eff_tid = _effective_task_id(item)
            rows = _run_one_sample(
                model, tokenizer, image_processors, item, device, language, max_new_tokens, eff_tid
            )
            by_task[task_type].extend(rows)
    return {k: _summarize(v) for k, v in sorted(by_task.items())}


def _collect_same_sample_dual_task(model, tokenizer, image_processors, items, device, language, max_new_tokens):
    """
    同一样本分别传 task_id=0 与 task_id=2，比较门控 L2 差。
    """
    per_sample = []
    model.eval()
    with torch.inference_mode():
        for idx, item in tqdm(items, desc="same-sample-dual-task", leave=False):
            g0 = _run_one_sample(model, tokenizer, image_processors, item, device, language, max_new_tokens, 0)
            g2 = _run_one_sample(model, tokenizer, image_processors, item, device, language, max_new_tokens, 2)
            if not g0 or not g2:
                continue
            m0 = np.mean(g0, axis=0)
            m2 = np.mean(g2, axis=0)
            diff = (m2 - m0).tolist()
            per_sample.append({
                "sample_idx": idx,
                "task_id0_mean": m0.round(6).tolist(),
                "task_id2_mean": m2.round(6).tolist(),
                "task2_minus_task0": [round(float(x), 6) for x in diff],
                "l2": round(float(np.linalg.norm(diff)), 6),
            })
    l2_list = [s["l2"] for s in per_sample]
    return {
        "samples": per_sample,
        "mean_l2": round(float(np.mean(l2_list)), 6) if l2_list else 0.0,
        "max_l2": round(float(np.max(l2_list)), 6) if l2_list else 0.0,
    }


def main():
    args = _parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Task-adaptive gate compare: before vs after task_ids fix")
    print("=" * 72)
    print(f"base_model      : {args.base_model}")
    print(f"checkpoint      : {args.checkpoint}")
    print(f"num_per_task    : {args.num_per_task}")
    print(f"num_same_sample : {args.num_same_sample}")
    print("=" * 72)

    buckets = _load_samples(args.test_json, args.num_per_task)
    flat_samples = buckets["type0"] + buckets["type2"]
    same_sample_items = buckets["type0"][: args.num_same_sample]

    tokenizer, model, image_processors = load_checkpoint_model(
        args.base_model,
        args.checkpoint,
        device=args.device,
        prefetch_hierarchical_fusion=True,
        hierarchical_num_heads=8,
        hierarchical_fusion_mode=args.hierarchical_fusion_mode,
    )

    orig_prepare = _patch_prepare_drop_task_ids(model)
    orig_hf_fwd = _patch_hierarchical_fusion_ignore_task_type()
    print("\n[Phase A] BEFORE fix — same sample, task_id=0 vs 2 ...")
    same_before = _collect_same_sample_dual_task(
        model, tokenizer, image_processors, same_sample_items, args.device, args.language, args.max_new_tokens
    )
    print("[Phase B] BEFORE fix — bucket aggregate type0 vs type2 ...")
    bucket_before = _collect_bucket_gates(
        model, tokenizer, image_processors, flat_samples, args.device, args.language, args.max_new_tokens
    )
    _restore_hierarchical_fusion_forward(orig_hf_fwd)
    _restore_prepare(model, orig_prepare)

    print("[Phase C] AFTER fix — same sample, task_id=0 vs 2 ...")
    same_after = _collect_same_sample_dual_task(
        model, tokenizer, image_processors, same_sample_items, args.device, args.language, args.max_new_tokens
    )
    print("[Phase D] AFTER fix — bucket aggregate type0 vs type2 ...")
    bucket_after = _collect_bucket_gates(
        model, tokenizer, image_processors, flat_samples, args.device, args.language, args.max_new_tokens
    )

    b0, b2 = bucket_before.get("type0", {}), bucket_before.get("type2", {})
    a0, a2 = bucket_after.get("type0", {}), bucket_after.get("type2", {})
    bucket_diff_before = _pair_diff(b0, b2)
    bucket_diff_after = _pair_diff(a0, a2)

    report = {
        "config": vars(args),
        "same_sample_dual_task": {
            "before_fix": same_before,
            "after_fix": same_after,
        },
        "bucket_type0_vs_type2": {
            "before_fix": bucket_before,
            "after_fix": bucket_after,
            "type2_minus_type0_before": bucket_diff_before,
            "type2_minus_type0_after": bucket_diff_after,
            "bucket_l2_before": round(_l2(bucket_diff_before), 6),
            "bucket_l2_after": round(_l2(bucket_diff_after), 6),
        },
    }

    print("\n" + "=" * 72)
    print("1) Same sample: task_id=0 vs task_id=2 (direct test of task_adaptive)")
    print("=" * 72)
    print(f"  Before fix: mean L2={same_before['mean_l2']:.6f}  max L2={same_before['max_l2']:.6f}")
    print(f"  After fix : mean L2={same_after['mean_l2']:.6f}  max L2={same_after['max_l2']:.6f}")
    if same_before["samples"]:
        ex = same_before["samples"][0]
        print(f"  Example BEFORE sample_idx={ex['sample_idx']}: "
              f"id0={_gate_mean_str(ex['task_id0_mean'])} | id2={_gate_mean_str(ex['task_id2_mean'])}")
    if same_after["samples"]:
        ex = same_after["samples"][0]
        print(f"  Example AFTER  sample_idx={ex['sample_idx']}: "
              f"id0={_gate_mean_str(ex['task_id0_mean'])} | id2={_gate_mean_str(ex['task_id2_mean'])}")

    print("\n" + "=" * 72)
    print("2) Bucket aggregate: JSON type0 vs type2")
    print("=" * 72)
    for label, data in [("BEFORE", bucket_before), ("AFTER", bucket_after)]:
        print(f"\n--- {label} ---")
        for t in ("type0", "type2"):
            s = data.get(t, {})
            print(f"  {t}: count={s.get('count', 0)}  {_gate_mean_str(s.get('mean', []))}")
    print(f"\n  type2-type0 L2: before={_l2(bucket_diff_before):.6f}  after={_l2(bucket_diff_after):.6f}")

    print("\n" + "=" * 72)
    if same_before["mean_l2"] < 0.01 and same_after["mean_l2"] > 0.05:
        print("结论: 修复有效。修复前同图换 task_id 门控不变；修复后 type0/type2 门控明显分离。")
    elif same_after["mean_l2"] > max(same_before["mean_l2"] * 3.0, 0.03):
        print("结论: 修复有效，同图 task_id=0 vs 2 的门控差异显著增大。")
    else:
        print("结论: 请增大 num_same_sample / num_per_task 重试，或确认 checkpoint 含 task_adaptive 权重。")
    print(
        "\n说明: 按 JSON 分 type0/type2 桶的均值差异，可能来自不同任务样本的图像差异，"
        "不能单独证明 task_adaptive 是否生效；请以「同图双 task_id」对比为准。"
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {args.output_json}")


if __name__ == "__main__":
    main()
