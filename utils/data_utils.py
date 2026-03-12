"""
数据处理工具函数

提供消息格式转换、边界框验证、区域ID映射等辅助功能。
"""
import re
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image


def convert_conversations_to_messages(
    conversations: List[Dict],
    image_path: str,
    regions: List[Dict]
) -> Tuple[List[Dict], str]:
    """
    将JSON格式的conversations转换为VLM-FO1所需的messages格式
    
    Args:
        conversations: 对话列表，包含user和assistant消息
        image_path: 图像文件路径
        regions: 区域列表，包含region_id和bbox
        
    Returns:
        (messages, assistant_text) 元组
        - messages: VLM-FO1格式的消息列表
        - assistant_text: 助手回复文本
    """
    messages = []
    user_text = None
    assistant_text = None
    
    # 提取用户和助手消息
    for conv in conversations:
        if conv['from'] == 'user':
            user_text = conv['value'].replace('<image>', '').strip()
            if user_text.startswith('\n'):
                user_text = user_text[1:]
        elif conv['from'] == 'assistant':
            assistant_text = conv['value']
    
    if user_text is None or assistant_text is None:
        return [], ""
    
    # 构建边界框列表
    bbox_list = []
    region_id_to_index = {}
    for idx, region in enumerate(regions):
        bbox = region['bbox']
        if len(bbox) == 4:
            bbox_list.append(bbox)
            region_id = region.get('region_id', f'region{idx}')
            region_id_to_index[region_id] = idx
    
    # 构建用户消息
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_path}
            },
            {
                "type": "text",
                "text": user_text
            }
        ],
        "bbox_list": bbox_list
    }
    
    messages.append(user_message)
    
    return messages, assistant_text


def validate_bbox(bbox: List[float], image_width: int, image_height: int) -> bool:
    """
    验证边界框的有效性
    
    Args:
        bbox: 边界框 [x_min, y_min, x_max, y_max]
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        边界框是否有效
    """
    if len(bbox) != 4:
        return False
    
    x_min, y_min, x_max, y_max = bbox
    
    # 检查坐标顺序
    if x_min >= x_max or y_min >= y_max:
        return False
    
    # 检查坐标范围
    if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
        return False
    
    # 检查最小尺寸（避免过小的边界框）
    if (x_max - x_min) < 1 or (y_max - y_min) < 1:
        return False
    
    return True


def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    归一化边界框坐标到[0, 1]范围
    
    Args:
        bbox: 边界框 [x_min, y_min, x_max, y_max]
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        归一化后的边界框
    """
    x_min, y_min, x_max, y_max = bbox
    
    # 确保坐标在有效范围内
    x_min = max(0, min(image_width, x_min))
    y_min = max(0, min(image_height, y_min))
    x_max = max(0, min(image_width, x_max))
    y_max = max(0, min(image_height, y_max))
    
    # 归一化
    normalized = [
        x_min / image_width,
        y_min / image_height,
        x_max / image_width,
        y_max / image_height
    ]
    
    return normalized


def denormalize_bbox(
    normalized_bbox: List[float],
    image_width: int,
    image_height: int
) -> List[float]:
    """
    将归一化的边界框坐标转换回像素坐标
    
    Args:
        normalized_bbox: 归一化的边界框 [x_min, y_min, x_max, y_max] (范围[0, 1])
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        像素坐标的边界框
    """
    x_min, y_min, x_max, y_max = normalized_bbox
    
    return [
        x_min * image_width,
        y_min * image_height,
        x_max * image_width,
        y_max * image_height
    ]


def extract_region_ids_from_text(text: str) -> Set[str]:
    """
    从文本中提取区域ID（如region0, region1等）
    
    Args:
        text: 包含区域ID的文本，例如 "<ground>阳性宫颈细胞</ground><object><region0></object>"
        
    Returns:
        区域ID集合
    """
    pattern = r'<region(\d+)>'
    matches = re.findall(pattern, text)
    return set([f'region{match}' for match in matches])


def map_region_ids_to_indices(
    text: str,
    region_id_to_index: Dict[str, int]
) -> Dict[str, int]:
    """
    将文本中的区域ID映射到索引
    
    Args:
        text: 包含区域ID的文本
        region_id_to_index: 区域ID到索引的映射字典
        
    Returns:
        区域ID到索引的映射字典（从文本中提取的）
    """
    region_ids = extract_region_ids_from_text(text)
    mapping = {}
    
    for region_id in region_ids:
        if region_id in region_id_to_index:
            mapping[region_id] = region_id_to_index[region_id]
    
    return mapping


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个边界框的IoU（Intersection over Union）
    
    Args:
        bbox1: 第一个边界框 [x_min, y_min, x_max, y_max]
        bbox2: 第二个边界框 [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU值（0到1之间）
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集区域
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def match_bboxes(
    pred_bboxes: List[List[float]],
    gt_bboxes: List[List[float]],
    iou_threshold: float = 0.5
) -> Tuple[List[int], List[int], List[float]]:
    """
    匹配预测边界框和真实边界框
    
    Args:
        pred_bboxes: 预测的边界框列表
        gt_bboxes: 真实的边界框列表
        iou_threshold: IoU阈值，用于判断是否匹配
        
    Returns:
        (matched_pred_indices, matched_gt_indices, iou_scores) 元组
        - matched_pred_indices: 匹配的预测框索引
        - matched_gt_indices: 匹配的真实框索引
        - iou_scores: 对应的IoU分数
    """
    matched_pred = []
    matched_gt = []
    iou_scores = []
    
    used_gt = set()
    
    # 为每个预测框找到最佳匹配的真实框
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_bbox in enumerate(gt_bboxes):
            if gt_idx in used_gt:
                continue
            
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_pred.append(pred_idx)
            matched_gt.append(best_gt_idx)
            iou_scores.append(best_iou)
            used_gt.add(best_gt_idx)
    
    return matched_pred, matched_gt, iou_scores


def parse_grounding_output(text: str) -> Dict[str, List[int]]:
    """
    解析grounding输出文本，提取标签和对应的区域索引
    
    Args:
        text: 模型输出文本，例如 "<ground>阳性宫颈细胞</ground><object><region0></object>"
              或 "<ground>阳性宫颈细胞</ground><objects><region0></objects>"
        
    Returns:
        字典，键为标签，值为区域索引列表
    """
    result = {}
    
    # 匹配 <ground>标签</ground><object(s)><region0></object(s)> 格式
    # 同时支持<object>和<objects>
    pattern = r'<ground>(.*?)</ground><objects?>(.*?)</objects?>'
    matches = re.findall(pattern, text)
    
    for label_text, region_text in matches:
        label_text = label_text.strip()
        # 提取区域索引
        region_pattern = r'<region(\d+)>'
        region_matches = re.findall(region_pattern, region_text)
        region_indices = [int(idx) for idx in region_matches]
        
        if label_text not in result:
            result[label_text] = []
        result[label_text].extend(region_indices)
    
    return result


def filter_regions_by_confidence(
    regions: List[Dict],
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    根据置信度过滤区域
    
    Args:
        regions: 区域列表，每个区域包含confidence字段
        confidence_threshold: 置信度阈值
        
    Returns:
        过滤后的区域列表
    """
    filtered = [
        region for region in regions
        if region.get('confidence', 1.0) >= confidence_threshold
    ]
    return filtered


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    获取图像尺寸
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        (width, height) 元组
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return (0, 0)

