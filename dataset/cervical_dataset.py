import json
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from vlm_fo1.mm_utils import (
    load_image,
    make_message_context,
    tokenizer_image_region_token,
    tokenizer_image_token,
    resize_shortest_edge_images_and_bboxes,
    adjust_bbox
)
from vlm_fo1.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_REGION_INDEX,
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_REGION_TOKEN,
    DEFAULT_REGION_FEATURE_TOKEN
)

# 任务输出格式约束：中/英两套文案，由 CervicalDataset(language=...) 选用
TASK_CONSTRAINT_TEXTS = {
    "zh": {
        "type2_keywords": [
            "描述该区域的病理属性", "不要输出任何 <ground>", "<objects>", "<region*>"
        ],
        "type34_keywords": ["只输出一个类别名称", "不要输出解释"],
        "type5_keywords": [
            "请按以下三部分结构回答", "【细胞来源分析】", "【病理特征观察】", "【诊断结果】"
        ],
        "task2_suffix": (
            "请直接用一句话描述该区域的病理属性（例如阳性/阴性、细胞类型、来源、核特征等）。"
            "不要输出任何 <ground>、<objects>、<region*> 等标签或坐标。"
        ),
        "task34_suffix": (
            "请从问题给出的候选类别中选择，并且只输出一个类别名称，不要输出解释、标点或其他内容。"
        ),
        "task5_suffix": (
            "请按以下三部分结构回答：【细胞来源分析】描述该切片来自何种组织或部位；"
            "【病理特征观察】描述该区域细胞的病理特征；"
            "【诊断结果】给出细胞类型（从问题中的候选类别中选择）并简要说明诊断依据。"
        ),
    },
    "en": {
        "type2_keywords": [
            "Describe the pathological attributes", "Do not output any <ground>", "<objects>", "<region*>"
        ],
        "type34_keywords": ["output only one category name", "Do not output explanation"],
        "type5_keywords": [
            "Answer in the following three parts", "[Cell origin analysis]", "[Pathological features]", "[Diagnosis]"
        ],
        "task2_suffix": (
            "Please describe the pathological features of this region along the following six aspects: "
            "check whether there is nuclear hyperchromasia, nuclear enlargement, nuclear irregularity, nuclear crowding, "
            "high nuclear-to-cytoplasm ratio, and koilocytosis. "
            "Do not output any <ground>, <objects>, <region*> tags or coordinates."
        ),
        "task34_suffix": (
            "Please choose one category from the given options and output only the category name, "
            "with no explanation, punctuation or other content."
        ),
        "task5_suffix": (
            "Please answer in the following three parts: "
            "<Cell origin analysis> Describe the tissue or site of origin of this slide; "
            "<Pathological features> Describe the pathological features of the cells in this region; "
            "<Diagnosis> Give the cell type (choose from the options in the question) and briefly state the diagnostic basis."
        ),
    },
}


class CervicalDataset(Dataset):
    """
    宫颈细胞定位数据集类
    
    处理JSON格式的数据，包含图像路径、边界框和对话格式。
    输出格式符合VLM-FO1的要求。
    """
    
    def __init__(
        self,
        json_path: str,
        tokenizer: AutoTokenizer,
        image_processors: Tuple,
        model_name: str = "qwen2.5-vl",
        max_regions: int = 100,
        image_size: Optional[int] = None,
        is_train: bool = True,
        verbose: bool = True,
        path_feature_root: Optional[str] = None,
        language: str = "zh",
    ):
        """
        Args:
            json_path: JSON数据文件路径
            tokenizer: Tokenizer对象
            image_processors: (primary_image_processor, auxiliary_image_processor)元组
            model_name: 模型名称，用于确定token格式
            max_regions: 最大区域数量
            image_size: 图像尺寸（可选）
            is_train: 是否为训练模式
            verbose: 是否打印调试信息
            path_feature_root: path feature 根目录（可选）
            language: 约束文案语言，"zh" 中文（默认），"en" 英文
        """
        self.tokenizer = tokenizer
        self.primary_image_processor, self.auxiliary_image_processor = image_processors
        self.model_name = model_name.lower()
        self.max_regions = max_regions
        self.image_size = image_size
        self.is_train = is_train
        self.verbose = verbose
        self.path_feature_root = path_feature_root
        self.language = (language or "zh").strip().lower()
        if self.language not in TASK_CONSTRAINT_TEXTS:
            self.language = "zh"
        self._constraint = TASK_CONSTRAINT_TEXTS[self.language]
        
        # 加载数据
        if self.verbose:
            print(f"[DEBUG] Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if self.verbose:
            print(f"[DEBUG] Loaded {len(raw_data)} raw samples from JSON file")
        
        # 为了加快 sample1000 等已清洗数据集的启动速度，这里临时跳过 _validate_data 的逐条合法性检查，
        # 直接使用原始 JSON 作为 self.data。若后续需要对大规模原始数据做清洗，可以再启用 _validate_data。
        self.data = raw_data
        
        if self.verbose:
            print(f"[DEBUG] Validation skipped, using all {len(self.data)} samples as-is")
    
    def _validate_data(self, data: List[Dict]) -> List[Dict]:
        """验证和过滤无效数据"""
        valid_data = []
        invalid_reasons = {
            'missing_fields': 0,
            'image_not_found': 0,
            'invalid_regions_count': 0,
            'invalid_conversations': 0,
            'invalid_bboxes': 0
        }
        
        for idx, item in enumerate(data):
            # 检查必要字段
            if 'image' not in item or 'conversations' not in item or 'regions' not in item:
                invalid_reasons['missing_fields'] += 1
                if self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}: Missing required fields")
                continue
            
            # 检查图像文件是否存在
            image_path = item['image']
            if not os.path.exists(image_path):
                invalid_reasons['image_not_found'] += 1
                if self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}: Image not found: {image_path}")
                continue
            
            # 检查区域数量
            if len(item['regions']) == 0 or len(item['regions']) > self.max_regions:
                invalid_reasons['invalid_regions_count'] += 1
                if self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}: Invalid regions count: {len(item['regions'])}")
                continue
            
            # 检查对话格式
            if len(item['conversations']) < 2:
                invalid_reasons['invalid_conversations'] += 1
                if self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}: Invalid conversations count: {len(item['conversations'])}")
                continue
            
            # 验证边界框格式
            valid_regions = []
            for region_idx, region in enumerate(item['regions']):
                bbox = region.get('bbox', [])
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        # 确保所有值都是数字
                        bbox_float = [float(x) for x in bbox]
                        # 验证边界框有效性
                        if bbox_float[0] < bbox_float[2] and bbox_float[1] < bbox_float[3]:
                            valid_regions.append(region)
                        elif self.verbose and idx < 5:
                            print(f"[DEBUG] Sample {idx}, Region {region_idx}: Invalid bbox coordinates: {bbox_float}")
                    except (ValueError, TypeError, IndexError) as e:
                        if self.verbose and idx < 5:
                            print(f"[DEBUG] Sample {idx}, Region {region_idx}: Bbox conversion error: {e}, bbox={bbox}")
                elif self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}, Region {region_idx}: Invalid bbox format: {type(bbox)}, len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}")
            
            if len(valid_regions) == 0:
                invalid_reasons['invalid_bboxes'] += 1
                if self.verbose and idx < 5:
                    print(f"[DEBUG] Sample {idx}: No valid regions after validation")
                continue
            
            # 更新regions为有效区域
            item['regions'] = valid_regions
            valid_data.append(item)
        
        if self.verbose:
            print(f"[DEBUG] Validation statistics:")
            print(f"  - Missing fields: {invalid_reasons['missing_fields']}")
            print(f"  - Image not found: {invalid_reasons['image_not_found']}")
            print(f"  - Invalid regions count: {invalid_reasons['invalid_regions_count']}")
            print(f"  - Invalid conversations: {invalid_reasons['invalid_conversations']}")
            print(f"  - Invalid bboxes: {invalid_reasons['invalid_bboxes']}")
        
        return valid_data

    def _convert_conversations_to_messages(self, item: Dict) -> List[Dict]:
        """
        将JSON中的conversations转换为VLM-FO1所需的messages格式
        
        Args:
            item: 数据项，包含conversations和regions
            
        Returns:
            messages列表，符合VLM-FO1格式
        """
        messages = []
        
        # 提取用户和助手消息
        user_msg = None
        assistant_msg = None
        
        for conv in item['conversations']:
            if conv['from'] == 'user':
                user_msg = conv['value']
            elif conv['from'] == 'assistant':
                assistant_msg = conv['value']
        
        if user_msg is None or assistant_msg is None:
            return []
        
        # 提取文本（移除<image>标签）
        user_text = user_msg.replace('<image>', '').strip()
        if user_text.startswith('\n'):
            user_text = user_text[1:]

        # ------------------------------------------------------------------
        # 关键：为不同任务增加“输出格式约束”，降低多任务训练时的任务混淆
        # - Type2（区域描述）：强约束只输出描述文本，禁止 <ground>/<objects>/<region*> 模板
        # - Type3/4（分类）：强约束只输出一个候选类别字符串
        # - Type5（区域细胞类型+诊断依据）：强约束按三部分模板输出
        # 说明：不改动原始 JSON，仅在训练/评估数据管道中增强 prompt。
        # ------------------------------------------------------------------
        task_id_raw = item.get("task_id", None)
        task_id = None
        try:
            if task_id_raw is not None:
                task_id = int(task_id_raw)
        except Exception:
            task_id = None

        # 检查是否已经包含约束（避免重复添加），使用当前语言的关键词
        c = self._constraint
        has_type2_constraint = any(k in user_text for k in c["type2_keywords"])
        has_type34_constraint = any(k in user_text for k in c["type34_keywords"])
        has_type5_constraint = any(k in user_text for k in c["type5_keywords"])

        if task_id == 2 and not has_type2_constraint:
            user_text = user_text + "\n" + c["task2_suffix"]
        elif task_id in (3, 4) and not has_type34_constraint:
            user_text = user_text + "\n" + c["task34_suffix"]
        elif task_id == 5 and not has_type5_constraint:
            user_text = user_text + "\n" + c["task5_suffix"]

        # 构建边界框列表
        bbox_list = []
        region_id_to_index = {}
        for idx, region in enumerate(item['regions']):
            bbox = region['bbox']
            # 确保边界框格式正确 [x_min, y_min, x_max, y_max]
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
                    "image_url": {"url": item['image']}
                },
                {
                    "type": "text",
                    "text": user_text
                }
            ],
            "bbox_list": bbox_list
        }
        
        messages.append(user_message)
        
        # 助手消息将在训练时单独处理
        self._assistant_text = assistant_msg
        self._region_id_to_index = region_id_to_index
        
        return messages
    
    def _prepare_labels(
        self,
        messages: List[Dict],
        assistant_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备输入token和标签
        
        Args:
            messages: 消息列表
            assistant_text: 助手回复文本
            
        Returns:
            (input_ids, labels) 元组
        """
        # 构建完整对话的prompt
        prompt = ""
        input_tokens = []
        
        # 处理Qwen2.5-VL的特殊token
        if 'qwen2.5-vl' in self.model_name or 'qwen2_5_vl' in self.model_name:
            global DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            DEFAULT_IM_START_TOKEN = "<|vision_start|>"
            DEFAULT_IM_END_TOKEN = "<|vision_end|>"
        
        # 构建用户消息的token
        for message in messages:
            inp, context_tokens, _, bbox_list = make_message_context(
                self.tokenizer, message, chat_format="chatml"
            )
            prompt += inp
            input_tokens.extend(context_tokens)
        
        # 添加系统提示（如果没有）
        if "system" not in prompt:
            system_content = "system\nYou are a helpful assistant."
            system_prompt = "<|im_start|>" + system_content + "<|im_end|>" + "\n"
            prompt = system_prompt + prompt
            system_tokens = [151644] + self.tokenizer(system_content).input_ids + [151645] + self.tokenizer("\n").input_ids
            input_tokens = system_tokens + input_tokens
        
        # 添加助手提示
        assistant_prompt = "<|im_start|>" + "assistant" + "\n"
        prompt += assistant_prompt
        assistant_tokens = [151644] + self.tokenizer("assistant\n").input_ids
        input_tokens.extend(assistant_tokens)
        
        # Tokenize助手回复
        assistant_tokens_full = self.tokenizer(assistant_text).input_ids
        # 移除BOS token（如果存在）
        if len(assistant_tokens_full) > 0 and assistant_tokens_full[0] == self.tokenizer.bos_token_id:
            assistant_tokens_full = assistant_tokens_full[1:]
        
        # 添加结束token
        assistant_tokens_full.append(151645)  # <|im_end|>
        
        # 构建完整的input_ids
        full_input_ids = input_tokens + assistant_tokens_full
        
        # 构建labels：用户输入部分设为IGNORE_INDEX，只计算助手回复的损失
        labels = [IGNORE_INDEX] * len(input_tokens) + assistant_tokens_full
        
        return torch.tensor(full_input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个数据样本
        
        Returns:
            包含以下键的字典：
            - input_ids: tokenized输入序列
            - labels: 标签序列
            - images: 主图像tensor
            - images_aux: 辅助图像tensor（如果有）
            - bbox_list: 边界框列表
            - image_grid_thws: 图像网格尺寸
        """
        # 使用循环而不是递归，避免无限递归
        max_attempts = min(100, len(self.data))  # 最多尝试100次或数据集大小
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                item = self.data[idx]
                
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] __getitem__({idx}): Processing sample {idx}")
                
                # 转换消息格式
                messages = self._convert_conversations_to_messages(item)
                if len(messages) == 0:
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: Failed to convert conversations to messages")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"All samples invalid after {attempt} attempts")
                    continue
                
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] Sample {idx}: Converted messages successfully, bbox_list length: {len(messages[0].get('bbox_list', []))}")
                
                assistant_text = self._assistant_text
                
                # 加载图像
                image_path = item['image']
                if not os.path.exists(image_path):
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: Image file not found: {image_path}")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"All image paths invalid after {attempt} attempts")
                    continue
                
                try:
                    image = load_image(image_path)
                    if self.verbose and attempt == 0:
                        print(f"[DEBUG] Sample {idx}: Loaded image, size: {image.size}")
                except Exception as e:
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: Error loading image: {e}")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"Error loading images after {attempt} attempts: {e}")
                    continue
                
                # 获取边界框列表（原始坐标，后续会用于 path_feature 对齐）
                bbox_list = messages[0]['bbox_list']
                bbox_list_original = [bbox.copy() for bbox in bbox_list]
                
                # 验证和过滤边界框格式
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] Sample {idx}: Original bbox_list length: {len(bbox_list)}")
                
                valid_bbox_list = []
                valid_bbox_indices = []
                for bbox_idx, bbox in enumerate(bbox_list):
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        try:
                            bbox_float = [float(x) for x in bbox]
                            if bbox_float[0] < bbox_float[2] and bbox_float[1] < bbox_float[3]:
                                valid_bbox_list.append(bbox_float)
                                valid_bbox_indices.append(bbox_idx)
                            elif self.verbose and attempt == 0 and bbox_idx < 3:
                                print(f"[DEBUG] Sample {idx}, Bbox {bbox_idx}: Invalid coordinates (x1>=x2 or y1>=y2): {bbox_float}")
                        except (ValueError, TypeError, IndexError) as e:
                            if self.verbose and attempt == 0 and bbox_idx < 3:
                                print(f"[DEBUG] Sample {idx}, Bbox {bbox_idx}: Conversion error: {e}, bbox={bbox}")
                            continue
                    elif self.verbose and attempt == 0 and bbox_idx < 3:
                        print(f"[DEBUG] Sample {idx}, Bbox {bbox_idx}: Invalid format: type={type(bbox)}, len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}")
                
                if len(valid_bbox_list) == 0:
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: No valid bboxes after initial validation (original: {len(bbox_list)})")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"No valid bboxes after {attempt} attempts")
                    continue
                
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] Sample {idx}: Valid bboxes after initial validation: {len(valid_bbox_list)}/{len(bbox_list)}")
                
                bbox_list = valid_bbox_list
                # 仅保留与当前有效 bbox 一一对应的原始坐标，确保后续 path_features 对齐。
                bbox_list_original = [bbox_list_original[i] for i in valid_bbox_indices]
                
                # 调整图像大小和边界框
                try:
                    if self.verbose and attempt == 0:
                        print(f"[DEBUG] Sample {idx}: Resizing image and bboxes, input bbox_list length: {len(bbox_list)}")
                    
                    if self.image_size is not None:
                        images, bbox_list_resized = resize_shortest_edge_images_and_bboxes(
                            [image], [bbox_list],
                            candidate_sizes=[self.image_size],
                            max_size=2048
                        )
                    else:
                        images, bbox_list_resized = resize_shortest_edge_images_and_bboxes(
                            [image], [bbox_list],
                            max_size=2048
                        )
                    image = images[0]
                    
                    if self.verbose and attempt == 0:
                        print(f"[DEBUG] Sample {idx}: After resize, bbox_list_resized type: {type(bbox_list_resized)}, length: {len(bbox_list_resized) if isinstance(bbox_list_resized, list) else 'N/A'}")
                        if isinstance(bbox_list_resized, list) and len(bbox_list_resized) > 0:
                            print(f"[DEBUG] Sample {idx}: First element type: {type(bbox_list_resized[0])}, length: {len(bbox_list_resized[0]) if isinstance(bbox_list_resized[0], list) else 'N/A'}")
                    
                    # 处理返回格式：resize_shortest_edge_images_and_bboxes返回格式分析
                    # 当我们传入[bbox_list]（其中bbox_list是[[x1,y1,x2,y2], ...]）时：
                    # - bbox_tensor.shape是(1, N, 4)，不是2维
                    # - 返回transformed_bbox_lists，即[[[x1',y1',x2',y2'], ...]]
                    # 所以需要解包一层
                    if isinstance(bbox_list_resized, list):
                        if len(bbox_list_resized) > 0:
                            # 检查第一个元素
                            if isinstance(bbox_list_resized[0], list):
                                # 检查第一个元素的第一个元素是否是数字（边界框坐标）
                                if len(bbox_list_resized[0]) > 0:
                                    if isinstance(bbox_list_resized[0][0], (int, float)):
                                        # bbox_list_resized[0]是边界框列表：[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
                                        bbox_list = bbox_list_resized[0]
                                        if self.verbose and attempt == 0:
                                            print(f"[DEBUG] Sample {idx}: bbox_list format: unpacked nested list, length: {len(bbox_list)}")
                                    elif isinstance(bbox_list_resized[0][0], list):
                                        # 进一步嵌套：[[[x1,y1,x2,y2], ...]]，需要解包两层
                                        # 如果只有一个bbox，bbox_list_resized[0][0]是[x1,y1,x2,y2]，需要包装成[[x1,y1,x2,y2]]
                                        if len(bbox_list_resized[0]) == 1:
                                            # 检查是否是单个bbox的坐标列表（长度为4）
                                            if len(bbox_list_resized[0][0]) == 4 and all(isinstance(x, (int, float)) for x in bbox_list_resized[0][0]):
                                                bbox_list = [bbox_list_resized[0][0]]  # 包装成列表
                                            else:
                                                bbox_list = bbox_list_resized[0][0]  # 已经是bbox列表
                                        else:
                                            bbox_list = bbox_list_resized[0]  # 多个bbox
                                        if self.verbose and attempt == 0:
                                            print(f"[DEBUG] Sample {idx}: bbox_list format: double nested unpacked, length: {len(bbox_list)}")
                                    else:
                                        # 可能是边界框列表，但格式不确定，尝试使用第一个元素
                                        bbox_list = bbox_list_resized[0]
                                        if self.verbose and attempt == 0:
                                            print(f"[DEBUG] Sample {idx}: bbox_list format: using first element, length: {len(bbox_list)}")
                                else:
                                    # 空列表
                                    bbox_list = []
                                    if self.verbose:
                                        print(f"[DEBUG] Sample {idx}: bbox_list_resized[0] is empty")
                            elif len(bbox_list_resized) == 4 and all(isinstance(x, (int, float)) for x in bbox_list_resized):
                                # 单个边界框：[x1,y1,x2,y2]，包装成列表
                                bbox_list = [bbox_list_resized]
                                if self.verbose and attempt == 0:
                                    print(f"[DEBUG] Sample {idx}: bbox_list format: single bbox wrapped, length: {len(bbox_list)}")
                            else:
                                if self.verbose:
                                    print(f"[DEBUG] Sample {idx}: Unexpected format: type={type(bbox_list_resized)}, len={len(bbox_list_resized)}, first_elem_type={type(bbox_list_resized[0]) if len(bbox_list_resized) > 0 else 'N/A'}, first_elem_len={len(bbox_list_resized[0]) if len(bbox_list_resized) > 0 and isinstance(bbox_list_resized[0], list) else 'N/A'}")
                                bbox_list = []
                        else:
                            if self.verbose:
                                print(f"[DEBUG] Sample {idx}: bbox_list_resized is empty list")
                            bbox_list = []
                    else:
                        if self.verbose:
                            print(f"[DEBUG] Sample {idx}: bbox_list_resized is not a list: {type(bbox_list_resized)}")
                        bbox_list = []
                except Exception as e:
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: Error in resize: {e}")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"Error in resize after {attempt} attempts: {e}")
                    continue
                
                # 验证边界框格式
                validated_bbox_list = []
                for bbox in bbox_list:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        try:
                            bbox_float = [float(x) for x in bbox]
                            if bbox_float[0] < bbox_float[2] and bbox_float[1] < bbox_float[3]:
                                validated_bbox_list.append(bbox_float)
                            elif self.verbose and attempt == 0:
                                print(f"[DEBUG] Sample {idx}: Invalid bbox after resize: {bbox_float}")
                        except (ValueError, TypeError, IndexError) as e:
                            if self.verbose and attempt == 0:
                                print(f"[DEBUG] Sample {idx}: Bbox validation error: {e}, bbox={bbox}")
                            continue
                    elif self.verbose and attempt == 0:
                        print(f"[DEBUG] Sample {idx}: Invalid bbox format after resize: type={type(bbox)}, len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}")
                
                if len(validated_bbox_list) == 0:
                    if self.verbose:
                        print(f"[DEBUG] Sample {idx}: No valid bboxes after resize validation (before: {len(bbox_list)})")
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"No valid bboxes after resize after {attempt} attempts")
                    continue
                
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] Sample {idx}: Valid bboxes after resize validation: {len(validated_bbox_list)}/{len(bbox_list)}")
                
                bbox_list = validated_bbox_list
                
                # 限制区域数量
                if len(bbox_list) > self.max_regions:
                    bbox_list = bbox_list[:self.max_regions]
                    bbox_list_original = bbox_list_original[:self.max_regions]
                
                # 预处理图像
                primary_images = []
                primary_image_grid_thws = []
                try:
                    processed_data = self.primary_image_processor.preprocess(image, return_tensors="pt")
                    primary_images.append(processed_data['pixel_values'])
                    primary_image_grid_thws.append(processed_data['image_grid_thw'])
                except Exception as e:
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"Error preprocessing image after {attempt} attempts: {e}")
                    continue
                
                # 处理辅助图像（如果模型支持区域特征）
                images_aux = None
                if self.auxiliary_image_processor is not None:
                    try:
                        origin_image_size = image.size
                        aux_images = [image]
                        images_aux = [
                            self.auxiliary_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
                            for img in aux_images
                        ]
                        
                        # 调整边界框到辅助图像尺寸
                        if bbox_list and len(bbox_list) > 0:
                            # 确保bbox_list格式正确
                            validated_bboxes = []
                            for bbox in bbox_list:
                                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                    try:
                                        bbox_float = [float(x) for x in bbox]
                                        validated_bboxes.append(bbox_float)
                                    except (ValueError, TypeError):
                                        continue
                            
                            if len(validated_bboxes) > 0:
                                resize_h, resize_w = images_aux[0].shape[-2:]
                                original_w, original_h = origin_image_size
                                adjusted_bbox_list = adjust_bbox(
                                    validated_bboxes, original_h, original_w, resize_h, resize_w
                                )
                                bbox_list = adjusted_bbox_list
                            else:
                                idx = (idx + 1) % len(self.data)
                                if idx == original_idx and attempt > 0:
                                    raise ValueError(f"No valid bboxes for aux after {attempt} attempts")
                                continue
                    except Exception as e:
                        idx = (idx + 1) % len(self.data)
                        if idx == original_idx and attempt > 0:
                            raise ValueError(f"Error processing aux image after {attempt} attempts: {e}")
                        continue
                
                # 准备输入和标签
                try:
                    input_ids, labels = self._prepare_labels(messages, assistant_text)
                except Exception as e:
                    idx = (idx + 1) % len(self.data)
                    if idx == original_idx and attempt > 0:
                        raise ValueError(f"Error preparing labels after {attempt} attempts: {e}")
                    continue
                
                # 构建返回字典
                task_raw = item.get('task_id', item.get('task_type', -1))
                try:
                    task_id = int(task_raw)
                except Exception:
                    task_id = -1

                result = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'images': primary_images[0],
                    'image_grid_thws': primary_image_grid_thws[0],
                    'task_id': task_id,
                    'assistant_text': assistant_text,
                }
                
                if images_aux is not None:
                    result['images_aux'] = images_aux[0]
                    result['bbox_list'] = torch.tensor(bbox_list, dtype=torch.float32)
                else:
                    result['bbox_list'] = None
                
                # 不再加载病理特征；所有任务仅依赖视觉特征
                
                # 成功获取数据，返回结果
                if self.verbose and attempt == 0:
                    print(f"[DEBUG] Sample {idx}: Successfully processed, final bbox_list length: {len(bbox_list)}")
                return result
                
            except Exception as e:
                # 如果发生其他错误，尝试下一个样本
                if self.verbose:
                    print(f"[DEBUG] Sample {idx}: Exception occurred (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {e}")
                idx = (idx + 1) % len(self.data)
                if idx == original_idx and attempt >= max_attempts - 1:
                    # 如果已经尝试了所有样本，抛出异常
                    raise RuntimeError(f"Failed to get valid sample after {max_attempts} attempts. Last error: {e}")
                continue
        
        # 如果循环结束还没有返回，说明所有样本都无效
        raise RuntimeError(f"Failed to get valid sample after {max_attempts} attempts from index {original_idx}")
