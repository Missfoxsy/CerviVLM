import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from build_path_features import compute_region_feature


# 9维特征定义（与 build_path_features.py 保持一致）
FEATURE_NAMES = [
    "mean_inst_area",
    "mean_inst_perimeter",
    "mean_circularity",
    "mean_eccentricity",
    "mean_nn_dist",
    "std_nn_dist",
    "crowding_index",
    "dark_mean",
    "dark_std",
]

TARGET_KEYWORDS = ["核深染", "核增大", "核不规则", "核拥挤"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether task_id=4 text descriptions match extracted pathology features."
    )
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to dataset JSON.")
    parser.add_argument(
        "--hover_pred_root",
        type=str,
        required=True,
        help="HoVer-Net prediction root (contains json/ and mat/).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=40,
        help="Number of random samples from task_id=4 keyword subset.",
    )
    parser.add_argument("--task_id", type=int, default=4, help="Target task id to filter.")
    parser.add_argument(
        "--text_source",
        type=str,
        default="assistant",
        choices=["assistant", "user", "both"],
        help="Where to search keywords for filtering and matching.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Optional path to save JSON report.",
    )
    return parser.parse_args()


def extract_assistant_text(item: Dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "assistant":
            return str(conv.get("value", ""))
    return ""


def extract_user_text(item: Dict) -> str:
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            return str(conv.get("value", ""))
    return ""


def parse_keywords(text: str) -> List[str]:
    return [kw for kw in TARGET_KEYWORDS if kw in text]


def _parse_region_index_from_id(sample_id: str) -> Optional[int]:
    if not sample_id:
        return None
    m = re.search(r"_region(\d+)_", sample_id)
    if m is None:
        m = re.search(r"_region(\d+)$", sample_id)
    return int(m.group(1)) if m else None


def _parse_region_index_from_user_text(user_text: str) -> Optional[int]:
    if not user_text:
        return None
    m = re.search(r"<region(\d+)>", user_text)
    return int(m.group(1)) if m else None


def resolve_target_region(item: Dict, user_text: str) -> Tuple[Optional[Dict], Optional[str], Optional[int]]:
    regions = item.get("regions", [])
    if not regions:
        return None, None, None

    sample_id = str(item.get("id", ""))
    idx_from_id = _parse_region_index_from_id(sample_id)
    idx_from_user = _parse_region_index_from_user_text(user_text)

    region_idx = idx_from_id if idx_from_id is not None else idx_from_user
    if region_idx is not None:
        target_region_id = f"region{region_idx}"
        # 优先按 region_id 精确匹配
        for rg in regions:
            if str(rg.get("region_id", "")) == target_region_id:
                return rg, "region_id_match", region_idx
        # 其次按 list 下标兜底（历史数据可能缺 region_id）
        if 0 <= region_idx < len(regions):
            return regions[region_idx], "region_index_fallback", region_idx

    # 最后兜底：仍返回首个区域，但标记来源，便于统计
    return regions[0], "fallback_first_region", region_idx


def load_hover_and_image_features(
    image_path: str,
    hover_pred_root: str,
    cache: Dict[str, Optional[Dict[str, object]]],
) -> Optional[Dict[str, object]]:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if image_name in cache:
        return cache[image_name]

    hover_json_path = os.path.join(hover_pred_root, "json", image_name + ".json")
    hover_mat_path = os.path.join(hover_pred_root, "mat", image_name + ".mat")

    if not os.path.exists(hover_json_path):
        cache[image_name] = None
        return None

    try:
        with open(hover_json_path, "r", encoding="utf-8") as f:
            hover_data = json.load(f)
    except Exception:
        cache[image_name] = None
        return None

    nuc_dict = hover_data.get("nuc", {})
    if not isinstance(nuc_dict, dict) or len(nuc_dict) == 0:
        cache[image_name] = None
        return None

    bboxes_list: List[List[float]] = []
    centroids_list: List[List[float]] = []
    ids_list: List[int] = []

    for inst_id, inst in nuc_dict.items():
        bb = inst.get("bbox")
        ct = inst.get("centroid")
        if bb is None or ct is None:
            continue
        if (
            not isinstance(bb, list)
            or len(bb) != 2
            or not isinstance(bb[0], list)
            or not isinstance(bb[1], list)
            or len(bb[0]) != 2
            or len(bb[1]) != 2
        ):
            continue
        if not isinstance(ct, (list, tuple)) or len(ct) != 2:
            continue

        x1, y1 = float(bb[0][0]), float(bb[0][1])
        x2, y2 = float(bb[1][0]), float(bb[1][1])
        row, col = float(ct[0]), float(ct[1])
        bboxes_list.append([x1, y1, x2, y2])
        centroids_list.append([col, row])  # [x, y]
        try:
            ids_list.append(int(inst_id))
        except Exception:
            ids_list.append(-1)

    if len(bboxes_list) == 0:
        cache[image_name] = None
        return None

    inst_map = None
    if os.path.exists(hover_mat_path):
        try:
            mat_data = sio.loadmat(hover_mat_path)
            if "inst_map" in mat_data and isinstance(mat_data["inst_map"], np.ndarray):
                inst_map = mat_data["inst_map"].astype(np.int64, copy=False)
        except Exception:
            inst_map = None

    pap_dark_channel = None
    if os.path.exists(image_path):
        try:
            image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
            rgb_max = image_rgb.max(axis=-1)
            pap_dark_channel = (1.0 - rgb_max).astype(np.float32, copy=False)
        except Exception:
            pap_dark_channel = None

    result = {
        "nuc_boxes": torch.tensor(bboxes_list, dtype=torch.float32),
        "nuc_centroids": torch.tensor(centroids_list, dtype=torch.float32),
        "nuc_ids": torch.tensor(ids_list, dtype=torch.long),
        "inst_map": inst_map,
        "pap_dark_channel": pap_dark_channel,
    }
    cache[image_name] = result
    return result


def percentile(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 0.0
    # 严格分位：只统计严格小于当前值的比例（不把“相等”计入）
    return float(np.mean(values < value))


def keyword_score(keyword: str, feat: np.ndarray, all_feats: np.ndarray) -> float:
    # feat: [9], all_feats: [N, 9]
    if keyword == "核深染":
        return percentile(all_feats[:, 7], feat[7])
    if keyword == "核增大":
        return percentile(all_feats[:, 0], feat[0])
    if keyword == "核拥挤":
        return percentile(all_feats[:, 6], feat[6])
    if keyword == "核不规则":
        ecc = percentile(all_feats[:, 3], feat[3])
        circ_low = 1.0 - percentile(all_feats[:, 2], feat[2])
        return 0.5 * (ecc + circ_low)
    return 0.0


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 过滤 task_id 且含目标关键词
    candidates = []
    for item in data:
        if int(item.get("task_id", -1)) != int(args.task_id):
            continue

        assistant_text = extract_assistant_text(item)
        user_text = extract_user_text(item)
        if args.text_source == "assistant":
            text_for_keywords = assistant_text
        elif args.text_source == "user":
            text_for_keywords = user_text
        else:
            text_for_keywords = f"{user_text}\n{assistant_text}"

        hit_keywords = parse_keywords(text_for_keywords)
        if not hit_keywords:
            continue
        if not item.get("regions"):
            continue
        if not item.get("image"):
            continue
        candidates.append((item, user_text, assistant_text, text_for_keywords, hit_keywords))

    if not candidates:
        print(f"No task_id={args.task_id} samples with target keywords found in {args.text_source} text.")
        return

    k = min(args.sample_size, len(candidates))
    sampled = random.sample(candidates, k=k)

    # 2) 为抽样样本构建 path_features
    parsed_cache: Dict[str, Optional[Dict[str, object]]] = {}
    rows = []
    all_feature_rows = []
    skip_counters = {
        "invalid_bbox": 0,
        "hover_or_image_missing": 0,
        "region_resolve_fallback": 0,
    }

    for item, user_text, assistant_text, text_for_keywords, hit_keywords in sampled:
        image_path = item["image"]
        target_region, region_resolve_mode, parsed_region_idx = resolve_target_region(item, user_text)
        if region_resolve_mode == "fallback_first_region":
            skip_counters["region_resolve_fallback"] += 1
        bbox = target_region.get("bbox", None) if target_region is not None else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            skip_counters["invalid_bbox"] += 1
            continue

        parsed = load_hover_and_image_features(image_path, args.hover_pred_root, parsed_cache)
        if parsed is None:
            skip_counters["hover_or_image_missing"] += 1
            continue

        feat = compute_region_feature(
            region_box=torch.tensor(bbox, dtype=torch.float32),
            nuc_boxes=parsed["nuc_boxes"],
            nuc_centroids=parsed["nuc_centroids"],
            nuc_ids=parsed["nuc_ids"],
            inst_map=parsed["inst_map"],
            pap_dark_channel=parsed["pap_dark_channel"],
        ).cpu().numpy()

        all_feature_rows.append(feat)
        rows.append(
            {
                "sample_id": item.get("id", ""),
                "image": image_path,
                "target_region_id": target_region.get("region_id", None) if target_region is not None else None,
                "region_resolve_mode": region_resolve_mode,
                "parsed_region_idx": parsed_region_idx,
                "bbox": bbox,
                "assistant_text": assistant_text,
                "user_text": user_text,
                "text_for_keywords": text_for_keywords,
                "keywords": hit_keywords,
                "feature_vector": feat.tolist(),
            }
        )

    if not rows:
        print("No valid sampled rows after feature extraction (check hover paths / image paths).")
        return

    feat_mat = np.stack(all_feature_rows, axis=0)  # [N, 9]
    zero_vector_flags = [np.allclose(np.array(r["feature_vector"], dtype=np.float32), 0.0) for r in rows]
    non_zero_indices = [i for i, is_zero in enumerate(zero_vector_flags) if not is_zero]

    # 3) 文本关键词与特征一致性验证（去除零向量样本）
    # 规则：对应关键词 score >= 0.60 视为匹配
    threshold = 0.60
    total_checks = 0
    passed_checks = 0
    per_keyword_total = {kw: 0 for kw in TARGET_KEYWORDS}
    per_keyword_pass = {kw: 0 for kw in TARGET_KEYWORDS}
    if len(non_zero_indices) > 0:
        feat_mat_non_zero = feat_mat[non_zero_indices]
    else:
        feat_mat_non_zero = np.zeros((0, feat_mat.shape[1]), dtype=np.float32)

    for idx, row in enumerate(rows):
        feat = np.array(row["feature_vector"], dtype=np.float32)
        if zero_vector_flags[idx]:
            row["skipped_in_eval_due_to_zero_vector"] = True
            row["keyword_checks"] = []
            row["all_keywords_match"] = False
            continue

        checks = []
        for kw in row["keywords"]:
            sc = keyword_score(kw, feat, feat_mat_non_zero)
            ok = sc >= threshold
            checks.append({"keyword": kw, "score": float(sc), "match": bool(ok)})
            total_checks += 1
            per_keyword_total[kw] += 1
            if ok:
                passed_checks += 1
                per_keyword_pass[kw] += 1
        row["skipped_in_eval_due_to_zero_vector"] = False
        row["keyword_checks"] = checks
        row["all_keywords_match"] = all(c["match"] for c in checks) if checks else False

    # 4) 输出统计
    overall_match_rate = (passed_checks / total_checks) if total_checks > 0 else 0.0
    eval_rows = [r for r in rows if not r.get("skipped_in_eval_due_to_zero_vector", False)]
    sample_all_match_rate = float(np.mean([r["all_keywords_match"] for r in eval_rows])) if eval_rows else 0.0
    zero_vector_rate = float(np.mean(zero_vector_flags))

    per_keyword_match_rate = {}
    for kw in TARGET_KEYWORDS:
        if per_keyword_total[kw] > 0:
            per_keyword_match_rate[kw] = float(per_keyword_pass[kw] / per_keyword_total[kw])
        else:
            per_keyword_match_rate[kw] = None

    summary = {
        "dataset_json": args.dataset_json,
        "hover_pred_root": args.hover_pred_root,
        "task_id": args.task_id,
        "text_source": args.text_source,
        "candidate_count": len(candidates),
        "sample_requested": args.sample_size,
        "sampled_count": len(rows),
        "evaluated_non_zero_count": len(eval_rows),
        "excluded_zero_count": int(np.sum(zero_vector_flags)),
        "skip_counters": skip_counters,
        "threshold": threshold,
        "overall_keyword_match_rate": overall_match_rate,
        "sample_all_match_rate": sample_all_match_rate,
        "zero_vector_rate": zero_vector_rate,
        "per_keyword_total": per_keyword_total,
        "per_keyword_pass": per_keyword_pass,
        "per_keyword_match_rate": per_keyword_match_rate,
        "feature_names": FEATURE_NAMES,
    }

    print("\n=== Task-4 Path Feature Validation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== Example Rows (first 10) ===")
    for row in rows[:10]:
        print(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "keywords": row["keywords"],
                    "checks": row["keyword_checks"],
                    "feature_vector": row["feature_vector"],
                    "text_for_keywords": row["text_for_keywords"],
                    "assistant_text": row["assistant_text"],
                },
                ensure_ascii=False,
            )
        )

    if args.output_report:
        report = {
            "summary": summary,
            "rows": rows,
        }
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()

