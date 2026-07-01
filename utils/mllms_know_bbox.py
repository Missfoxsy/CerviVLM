# -*- coding: utf-8 -*-
"""mllms_know/utils.py 中 bbox_from_att_image_adaptive 的副本（仅 numpy，避免包名冲突）。"""

from __future__ import annotations

import numpy as np


def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    从注意力图生成自适应 bbox（与 mllms_know/utils.py 一致）。
    image_size: (width, height)
    """
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]
    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]
        block_num = (
            min(int(bbox_size * ratio / block_size[0]), att_map.shape[1]),
            min(int(bbox_size * ratio / block_size[1]), att_map.shape[0]),
        )
        if att_map.shape[1] - block_num[0] < 1 and att_map.shape[0] - block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            continue
        block_nums.append((block_num[0], block_num[1]))

        sliding_att = np.zeros((att_map.shape[0] - block_num[1] + 1, att_map.shape[1] - block_num[0] + 1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        for x in range(att_map.shape[1] - block_num[0] + 1):
            for y in range(att_map.shape[0] - block_num[1] + 1):
                att = att_map[y : y + block_num[1], x : x + block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)

        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] - 1])
        if max_att_pos[0] < sliding_att.shape[1] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] + 1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1] - 1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1] + 1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)

    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]

    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)

    x_center = selected_bbox_size // 2 if x_center < selected_bbox_size // 2 else x_center
    y_center = selected_bbox_size // 2 if y_center < selected_bbox_size // 2 else y_center
    x_center = image_size[0] - selected_bbox_size // 2 if x_center > image_size[0] - selected_bbox_size // 2 else x_center
    y_center = image_size[1] - selected_bbox_size // 2 if y_center > image_size[1] - selected_bbox_size // 2 else y_center

    x1 = max(0, x_center - selected_bbox_size // 2)
    y1 = max(0, y_center - selected_bbox_size // 2)
    x2 = min(image_size[0], x_center + selected_bbox_size // 2)
    y2 = min(image_size[1], y_center + selected_bbox_size // 2)
    return x1, y1, x2, y2
