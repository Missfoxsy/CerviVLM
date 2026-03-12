import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from skimage.measure import perimeter, regionprops


def bbox_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    计算一个 box 与一组 boxes 的 IoU。

    Args:
        box: Tensor[4] = [x1, y1, x2, y2]
        boxes: Tensor[N, 4]
    """
    # [4] -> [1,4]
    box = box.unsqueeze(0)

    x1 = torch.max(box[:, 0], boxes[:, 0])
    y1 = torch.max(box[:, 1], boxes[:, 1])
    x2 = torch.min(box[:, 2], boxes[:, 2])
    y2 = torch.min(box[:, 3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_box = (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    union = area_box + area_boxes - inter
    iou = inter / union.clamp(min=1e-6)
    return iou  # [N]


def compute_region_feature(
    region_box: torch.Tensor,
    nuc_boxes: torch.Tensor,
    nuc_centroids: torch.Tensor,
    nuc_ids: torch.Tensor,
    inst_map: np.ndarray = None,
    pap_dark_channel: np.ndarray = None,
) -> torch.Tensor:
    """
    根据一个 ROI bbox 与该图像中的所有核实例信息，构造该 ROI 的病理统计向量。

    当前实现仅保留新增特征：
      - 实例形态（面积/周长/圆度/偏心率）
      - 空间拥挤（最近邻距离均值/标准差 + crowding index）
      - 核深染（巴氏染色场景下，基于核像素亮度反转得到 dark_mean/dark_std）
    """
    base_dim = 9
    if nuc_boxes.numel() == 0:
        # 没有任何核时返回零向量
        return torch.zeros(base_dim, dtype=torch.float32)

    # 使用质心判断核是否落在 ROI 内
    x1, y1, x2, y2 = region_box
    # 注意：在 build_path_features_for_dataset 中，我们已经将质心转换为 [x, y]
    cx = nuc_centroids[:, 0]
    cy = nuc_centroids[:, 1]
    inside_mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    if inside_mask.sum() == 0:
        return torch.zeros(base_dim, dtype=torch.float32)

    nuc_boxes_roi = nuc_boxes[inside_mask]
    nuc_ids_roi = nuc_ids[inside_mask]

    # ---- 实例级形态特征（优先使用 inst_map） ----
    mean_inst_area = 0.0
    mean_inst_perimeter = 0.0
    mean_circularity = 0.0
    mean_eccentricity = 0.0

    # ---- 空间拥挤特征 ----
    mean_nn_dist = 0.0
    std_nn_dist = 0.0
    crowding_index = 0.0

    centroids_roi = nuc_centroids[inside_mask].to(torch.float32)
    if centroids_roi.shape[0] >= 2:
        pts = centroids_roi.cpu().numpy()
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        nn = np.min(dist, axis=1)
        mean_nn_dist = float(nn.mean())
        std_nn_dist = float(nn.std())
        crowding_index = float(1.0 / (mean_nn_dist + 1e-6))

    # ---- 核深染特征（巴氏染色下使用亮度反转） ----
    dark_mean = 0.0
    dark_std = 0.0

    if inst_map is not None:
        h_img, w_img = inst_map.shape[:2]
        rx1 = max(0, int(np.floor(float(x1))))
        ry1 = max(0, int(np.floor(float(y1))))
        rx2 = min(w_img, int(np.ceil(float(x2))))
        ry2 = min(h_img, int(np.ceil(float(y2))))

        if rx2 > rx1 and ry2 > ry1:
            inst_roi = inst_map[ry1:ry2, rx1:rx2]
            ids_np = nuc_ids_roi.cpu().numpy().astype(np.int64)

            inst_areas = []
            inst_perims = []
            inst_circularities = []
            inst_eccs = []
            nuclei_mask_roi = np.zeros_like(inst_roi, dtype=bool)

            for inst_id in ids_np:
                if inst_id <= 0:
                    continue
                mask = (inst_roi == inst_id)
                area = int(mask.sum())
                if area <= 0:
                    continue

                nuclei_mask_roi |= mask
                inst_areas.append(float(area))

                try:
                    p = float(perimeter(mask, neighborhood=8))
                except TypeError:
                    p = float(perimeter(mask))
                inst_perims.append(p)
                circ = float((4.0 * np.pi * area) / (p * p + 1e-6))
                inst_circularities.append(circ)

                rp = regionprops(mask.astype(np.uint8))
                if len(rp) > 0:
                    inst_eccs.append(float(rp[0].eccentricity))

            if inst_areas:
                mean_inst_area = float(np.mean(inst_areas))
            if inst_perims:
                mean_inst_perimeter = float(np.mean(inst_perims))
            if inst_circularities:
                mean_circularity = float(np.mean(inst_circularities))
            if inst_eccs:
                mean_eccentricity = float(np.mean(inst_eccs))

            if pap_dark_channel is not None:
                dark_roi = pap_dark_channel[ry1:ry2, rx1:rx2]
                if dark_roi.shape == inst_roi.shape:
                    dark_vals = dark_roi[nuclei_mask_roi]
                    if dark_vals.size > 0:
                        dark_mean = float(np.mean(dark_vals))
                        dark_std = float(np.std(dark_vals))

    feat = torch.tensor(
        [
            mean_inst_area,
            mean_inst_perimeter,
            mean_circularity,
            mean_eccentricity,
            mean_nn_dist,
            std_nn_dist,
            crowding_index,
            dark_mean,
            dark_std,
        ],
        dtype=torch.float32,
    )
    return feat


def build_path_features_for_dataset(
    dataset_json: str,
    hover_pred_root: str,
    output_root: str,
):
    """
    根据标注 JSON 和 HoVer-Net 推理结果，为每个图像的每个 ROI 生成病理统计特征。

    输出：
      在 output_root 下为每个图像写一个 .pt 文件：
        {
          "bboxes": Tensor[M, 4],
          "features": Tensor[M, D_raw]
        }
      其中 M 为该图像在所有样本中累计出现的 ROI 数量。
    """
    os.makedirs(output_root, exist_ok=True)

    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按图像聚合 ROI
    image_to_regions: Dict[str, List[Tuple[List[float], Dict]]] = defaultdict(list)
    for item in data:
        image_path = item.get("image")
        if not image_path:
            continue
        for region in item.get("regions", []):
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            image_to_regions[image_path].append((bbox, region))

    print(f"[build_path_features] Total images in JSON: {len(image_to_regions)}")

    # 逐图像读取 HoVer-Net json 结果并构建特征
    for image_path, regions in image_to_regions.items():
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        hover_json_path = os.path.join(hover_pred_root, "json", image_name + ".json")
        hover_mat_path = os.path.join(hover_pred_root, "mat", image_name + ".mat")

        if not os.path.exists(hover_json_path):
            print(f"[build_path_features] Warning: hover json not found for image {image_name}: {hover_json_path}")
            continue

        with open(hover_json_path, "r", encoding="utf-8") as f:
            hover_data = json.load(f)

        # HoVer-Net 官方 json 结构:
        # {
        #   "mag": ...,
        #   "nuc": {
        #       "1": {"bbox": [[x1, y1], [x2, y2]], "centroid": [row, col], "type": int, ...},
        #       "2": {...},
        #       ...
        #   }
        # }
        nuc_dict = hover_data.get("nuc", {})
        if not isinstance(nuc_dict, dict) or len(nuc_dict) == 0:
            print(f"[build_path_features] Warning: empty or invalid 'nuc' field in {hover_json_path}")
            continue

        bboxes_list = []
        centroids_list = []
        ids_list = []
        for inst_id, inst in nuc_dict.items():
            bb = inst.get("bbox")
            ct = inst.get("centroid")
            if bb is None or ct is None:
                continue
            # bbox: [[x1, y1], [x2, y2]]
            if (
                not isinstance(bb, list)
                or len(bb) != 2
                or not isinstance(bb[0], list)
                or not isinstance(bb[1], list)
                or len(bb[0]) != 2
                or len(bb[1]) != 2
            ):
                continue
            # centroid: [row, col] -> 转为 [x, y] = [col, row]
            if not isinstance(ct, (list, tuple)) or len(ct) != 2:
                continue
            x1, y1 = float(bb[0][0]), float(bb[0][1])
            x2, y2 = float(bb[1][0]), float(bb[1][1])
            row, col = float(ct[0]), float(ct[1])

            bboxes_list.append([x1, y1, x2, y2])
            centroids_list.append([col, row])

            try:
                ids_list.append(int(inst_id))
            except Exception:
                ids_list.append(-1)

        if len(bboxes_list) == 0:
            print(f"[build_path_features] Warning: no valid nuclei parsed in {hover_json_path}")
            continue

        bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
        centroids = torch.tensor(centroids_list, dtype=torch.float32)
        ids = torch.tensor(ids_list, dtype=torch.long)

        # 优先从 mat 读取实例图，用于实例级形态与核深染统计
        inst_map = None
        if os.path.exists(hover_mat_path):
            try:
                mat_data = sio.loadmat(hover_mat_path)
                if "inst_map" in mat_data and isinstance(mat_data["inst_map"], np.ndarray):
                    inst_map = mat_data["inst_map"].astype(np.int64, copy=False)
            except Exception as e:
                print(f"[build_path_features] Warning: failed loading mat for {image_name}: {e}")

        # 从原图计算巴氏染色下的“深染”通道：dark = 1 - Value(HSV)
        pap_dark_channel = None
        if os.path.exists(image_path):
            try:
                image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
                rgb_max = image_rgb.max(axis=-1)
                pap_dark_channel = (1.0 - rgb_max).astype(np.float32, copy=False)
            except Exception as e:
                print(f"[build_path_features] Warning: failed to compute Pap dark channel for {image_name}: {e}")

        roi_boxes = []
        roi_feats = []
        for bbox, _ in regions:
            box_tensor = torch.tensor(bbox, dtype=torch.float32)
            feat = compute_region_feature(
                region_box=box_tensor,
                nuc_boxes=bboxes,
                nuc_centroids=centroids,
                nuc_ids=ids,
                inst_map=inst_map,
                pap_dark_channel=pap_dark_channel,
            )
            roi_boxes.append(box_tensor)
            roi_feats.append(feat)

        if not roi_boxes:
            continue

        roi_boxes = torch.stack(roi_boxes, dim=0)
        roi_feats = torch.stack(roi_feats, dim=0)

        out_path = os.path.join(output_root, image_name + ".pt")
        torch.save({"bboxes": roi_boxes, "features": roi_feats}, out_path)
        print(f"[build_path_features] Saved {roi_boxes.shape[0]} regions for image {image_name} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build pathology-aware region features from HoVer-Net predictions.")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to cervical dataset JSON (stage2_train.json / stage2_test.json).")
    parser.add_argument("--hover_pred_root", type=str, required=True, help="Root directory of HoVer-Net predictions (contains json/, mat/, overlay/ ...).")
    parser.add_argument("--output_root", type=str, required=True, help="Directory to save aggregated region-level pathology features.")
    args = parser.parse_args()

    build_path_features_for_dataset(
        dataset_json=args.dataset_json,
        hover_pred_root=args.hover_pred_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()


