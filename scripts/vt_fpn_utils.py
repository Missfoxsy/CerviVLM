# -*- coding: utf-8 -*-
"""VT SimpleFPN 层数消融：策略 A + legacy 层级注意力（无门控）。"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vlm_fo1.model.multimodal_visual_prompt_encoder.simple_fpn import SimpleFP

# 与 SimpleFP scale_factors 下标一一对应
FPN_ABLATION_PRESETS: Dict[int, Dict] = {
    1: {
        "indices": [2],
        "scale_factors": [1.0],
        "strides": [14.0],
    },
    2: {
        "indices": [1, 2],
        "scale_factors": [2.0, 1.0],
        "strides": [7.0, 14.0],
    },
    4: {
        "indices": [0, 1, 2, 3],
        "scale_factors": [4.0, 2.0, 1.0, 0.5],
        "strides": [3.5, 7.0, 14.0, 28.0],
    },
    6: {
        "indices": [0, 1, 2, 3, 4, 5],
        "scale_factors": [8.0, 4.0, 2.0, 1.0, 0.5, 0.25],
        "strides": [1.75, 3.5, 7.0, 14.0, 28.0, 56.0],
    },
}

FULL_6_SCALE_FACTORS = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
DEFAULT_4_SCALE_FACTORS = [4.0, 2.0, 1.0, 0.5]

VT_FPN_LEVEL_CHANNELS = 512
VT_FPN_OUT_DIM = 2048
VT_FPN_BASE_STRIDE = 14.0


def resolve_fpn_num_levels(model_config: dict) -> int:
    n = int(model_config.get("fpn_num_levels", 4))
    if n not in FPN_ABLATION_PRESETS:
        raise ValueError(
            f"Unsupported fpn_num_levels={n}. Supported: {sorted(FPN_ABLATION_PRESETS.keys())}"
        )
    return n


def get_object_vp_extractor(model):
    core = model.get_model() if hasattr(model, "get_model") else model
    return getattr(core, "object_vp_extractor", None)


def _strides_from_scale_factors(
    scale_factors: List[float], base_stride: float = VT_FPN_BASE_STRIDE
) -> List[float]:
    return [float(base_stride) / float(s) for s in scale_factors]


def _copy_simple_fpn_stages_by_scale(old_fpn: SimpleFP, new_fpn: SimpleFP) -> None:
    """按 scale_factor 对齐，将旧 SimpleFP 各 stage 权重复制到新 SimpleFP。"""
    old_scales = list(old_fpn.scale_factors)
    new_scales = list(new_fpn.scale_factors)
    copied = 0
    for old_idx, scale in enumerate(old_scales):
        if scale not in new_scales:
            continue
        new_idx = new_scales.index(scale)
        new_fpn.stages[new_idx].load_state_dict(old_fpn.stages[old_idx].state_dict())
        copied += 1
    print(f"[VT-FPN] Copied {copied}/{len(old_scales)} SimpleFP stages by scale alignment.")


def extend_simple_fpn_to_six_levels(extractor, dim: int = 1280, stride: int = 14) -> None:
    """将 VT SimpleFP 从 4 层扩展到 6 层，并迁移原 4 层预训练权重。"""
    if not hasattr(extractor, "simple_fpn"):
        raise RuntimeError("object_vp_extractor.simple_fpn not found")
    old_fpn = extractor.simple_fpn
    if list(getattr(old_fpn, "scale_factors", [])) == FULL_6_SCALE_FACTORS:
        return

    ref_param = next(old_fpn.parameters())
    new_fpn = SimpleFP(
        out_channels=VT_FPN_LEVEL_CHANNELS,
        scale_factors=FULL_6_SCALE_FACTORS,
        norm="LN",
        square_pad=0,
        dim=dim,
        stride=stride,
    )
    _copy_simple_fpn_stages_by_scale(old_fpn, new_fpn)
    extractor.simple_fpn = new_fpn.to(device=ref_param.device, dtype=ref_param.dtype)
    print(f"[VT-FPN] SimpleFP extended to 6 levels: scales={FULL_6_SCALE_FACTORS}")


def apply_vt_fpn_ablation(
    model,
    fpn_num_levels: int,
    vt_fpn_out_dim: int = VT_FPN_OUT_DIM,
    level_channels: int = VT_FPN_LEVEL_CHANNELS,
    use_hierarchical_fusion: bool = False,
) -> dict:
    """
    配置 VT FPN 层数消融。

    - use_hierarchical_fusion=True（legacy）：保留跨层注意力，输出由 HierarchicalFeatureFusion 产生；
    - use_hierarchical_fusion=False：子集 concat + Linear→2048（策略 A 无注意力路径）。
    """
    if fpn_num_levels not in FPN_ABLATION_PRESETS:
        raise ValueError(
            f"Unsupported fpn_num_levels={fpn_num_levels}. "
            f"Supported: {sorted(FPN_ABLATION_PRESETS.keys())}"
        )

    preset = FPN_ABLATION_PRESETS[fpn_num_levels]
    extractor = get_object_vp_extractor(model)
    if extractor is None:
        raise RuntimeError("object_vp_extractor not found on model")

    if not getattr(extractor, "use_simpleFPN_for_vt", False):
        raise RuntimeError("VT FPN ablation requires mm_use_simpleFPN_for_vt=true")

    if fpn_num_levels == 6:
        extend_simple_fpn_to_six_levels(extractor)

    ref_param = next(model.parameters())
    device = ref_param.device
    dtype = ref_param.dtype

    extractor.fpn_num_levels = int(fpn_num_levels)
    extractor.fpn_level_indices = list(preset["indices"])
    extractor.vt_fpn_strides = _strides_from_scale_factors(preset["scale_factors"])
    extractor.vt_fpn_out_dim = int(vt_fpn_out_dim)

    in_dim = len(extractor.fpn_level_indices) * int(level_channels)
    if use_hierarchical_fusion:
        if hasattr(extractor, "vt_level_fusion_proj"):
            del extractor.vt_level_fusion_proj
    elif in_dim != vt_fpn_out_dim:
        proj = nn.Linear(in_dim, vt_fpn_out_dim)
        extractor.vt_level_fusion_proj = proj.to(device=device, dtype=dtype)
    elif hasattr(extractor, "vt_level_fusion_proj"):
        del extractor.vt_level_fusion_proj

    meta = {
        "fpn_num_levels": int(fpn_num_levels),
        "fpn_level_indices": list(extractor.fpn_level_indices),
        "fpn_scale_factors": list(preset["scale_factors"]),
        "vt_fpn_strides": list(extractor.vt_fpn_strides),
        "vt_fpn_in_dim": in_dim,
        "vt_fpn_out_dim": int(vt_fpn_out_dim),
        "use_hierarchical_fusion": bool(use_hierarchical_fusion),
        "hierarchical_fusion_mode": "legacy" if use_hierarchical_fusion else None,
        "strategy": "legacy_hierarchical_attention" if use_hierarchical_fusion else "subset_concat_linear_2048",
    }
    print(
        f"[VT-FPN] num_levels={fpn_num_levels}, indices={meta['fpn_level_indices']}, "
        f"scales={meta['fpn_scale_factors']}, in_dim={in_dim}, out_dim={vt_fpn_out_dim}, "
        f"hierarchical_fusion={use_hierarchical_fusion}"
    )
    return meta


def save_fpn_ablation_meta(output_dir: str, meta: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fpn_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_fpn_ablation_meta(checkpoint_dir: str) -> Optional[dict]:
    path = os.path.join(checkpoint_dir, "fpn_ablation.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_fpn_num_levels_for_eval(
    checkpoint_dir: str,
    cli_fpn_num_levels: Optional[int] = None,
) -> int:
    if cli_fpn_num_levels is not None:
        return int(cli_fpn_num_levels)
    meta = load_fpn_ablation_meta(checkpoint_dir)
    if meta and "fpn_num_levels" in meta:
        return int(meta["fpn_num_levels"])
    return 4


def resolve_use_hierarchical_fusion_for_eval(
    checkpoint_dir: str,
    cli_use_hf: Optional[bool] = None,
) -> bool:
    if cli_use_hf is not None:
        return bool(cli_use_hf)
    meta = load_fpn_ablation_meta(checkpoint_dir)
    if meta and "use_hierarchical_fusion" in meta:
        return bool(meta["use_hierarchical_fusion"])
    return False


def peft_modules_to_save_for_fpn(model, model_config: dict) -> List[str]:
    """返回需写入 PEFT adapter 的 object_vp_extractor 子模块路径（仅无 HF 时的投影层）。"""
    if bool(model_config.get("use_hierarchical_fusion", False)):
        return []
    modules: List[str] = []
    fpn_num_levels = resolve_fpn_num_levels(model_config)
    extractor = get_object_vp_extractor(model)
    if extractor is not None and hasattr(extractor, "vt_level_fusion_proj"):
        in_dim = len(FPN_ABLATION_PRESETS[fpn_num_levels]["indices"]) * VT_FPN_LEVEL_CHANNELS
        if in_dim != VT_FPN_OUT_DIM:
            modules.append("model.object_vp_extractor.vt_level_fusion_proj")
    return modules
