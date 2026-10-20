# -*- coding: utf-8 -*-
"""PS3C 二分类（阳性/阴性）数据集：为无 regions 的样本注入整图 bbox，复用 CervicalDataset 管道。"""

import hashlib
import json
import os
from typing import Dict, List, Optional

from PIL import Image

from dataset.cervical_dataset import CervicalDataset

PS3C_CLASSES = ("阳性", "阴性")
# 与 Stage2 分类任务一致，便于 task_adaptive 门控
PS3C_TASK_ID = 4


def extract_label_from_item(item: dict) -> Optional[str]:
    """从 conversations 的 assistant 回复解析 GT 标签。"""
    for conv in item.get("conversations", []):
        if conv.get("from") == "assistant":
            return parse_ps3c_label(conv.get("value", ""))
    return None


def parse_ps3c_label(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip()
    for sep in ("<|im_end|>", "\n"):
        if sep in t:
            t = t.split(sep)[0].strip()
    if t in PS3C_CLASSES:
        return t
    pos = t.find("阳性")
    neg = t.find("阴性")
    if pos != -1 and (neg == -1 or pos <= neg):
        return "阳性"
    if neg != -1:
        return "阴性"
    return None


def enrich_ps3c_item(item: dict) -> dict:
    """为 PS3C 样本补充 regions（整图 bbox）与 task_id。"""
    out = dict(item)
    if "task_id" not in out:
        out["task_id"] = PS3C_TASK_ID

    regions = out.get("regions") or []
    if regions:
        return out

    image_path = out.get("image", "")
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"PS3C 图像不存在: {image_path}")

    with Image.open(image_path) as im:
        w, h = im.size
    out["regions"] = [{"region_id": "region0", "bbox": [0.0, 0.0, float(w), float(h)]}]
    return out


def _enriched_cache_path(json_path: str, cache_dir: Optional[str]) -> str:
    base = os.path.abspath(json_path)
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    root = cache_dir or os.path.join(os.path.dirname(base), ".cache")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"ps3c_enriched_{digest}.json")


def build_enriched_json(json_path: str, cache_dir: Optional[str] = None, force: bool = False) -> str:
    """将原始 PS3C JSON 转为带 regions 的缓存文件，返回缓存路径。"""
    cache_path = _enriched_cache_path(json_path, cache_dir)
    src_mtime = os.path.getmtime(json_path)
    if not force and os.path.isfile(cache_path):
        if os.path.getmtime(cache_path) >= src_mtime:
            return cache_path

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    enriched: List[Dict] = []
    for item in raw:
        enriched.append(enrich_ps3c_item(item))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    return cache_path


class PS3CClassificationDataset(CervicalDataset):
    """PS3C few-shot 分类数据，内部自动 enrich 后走 CervicalDataset。"""

    def __init__(self, json_path: str, enrich_cache_dir: Optional[str] = None, **kwargs):
        enriched_path = build_enriched_json(json_path, cache_dir=enrich_cache_dir)
        super().__init__(enriched_path, **kwargs)
