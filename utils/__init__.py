"""
数据处理工具模块
"""

from .data_utils import (
    convert_conversations_to_messages,
    validate_bbox,
    normalize_bbox,
    denormalize_bbox,
    extract_region_ids_from_text,
    map_region_ids_to_indices,
    calculate_iou,
    match_bboxes,
    parse_grounding_output,
    filter_regions_by_confidence,
    get_image_size
)

__all__ = [
    'convert_conversations_to_messages',
    'validate_bbox',
    'normalize_bbox',
    'denormalize_bbox',
    'extract_region_ids_from_text',
    'map_region_ids_to_indices',
    'calculate_iou',
    'match_bboxes',
    'parse_grounding_output',
    'filter_regions_by_confidence',
    'get_image_size'
]






















