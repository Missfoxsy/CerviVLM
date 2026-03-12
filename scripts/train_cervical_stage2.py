#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞定位第二阶段微调训练脚本

基于第一阶段微调后的模型继续微调。
"""

import os
import sys
import yaml
import argparse
import re
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

# LoRA支持
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will be disabled.")

# Wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.constants import IGNORE_INDEX


TYPE3_CELL_TYPE_CLASSES = [
    "ASCUS",
    "ASC_H",
    "EpitheliumNormal",
    "GlandTumor",
    "HPV",
    "HSIL",
    "LSIL",
]

TYPE4_SOURCE_CLASSES = [
    "上皮成团",
    "宫颈鳞状上皮中层",
    "宫颈鳞状上皮底层",
    "宫颈鳞状上皮表层",
    "腺上皮成团",
]


class CervicalDataCollator:
    """数据整理器，用于批处理数据"""
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, features):
        """
        整理一批数据
        
        Args:
            features: 数据样本列表
            
        Returns:
            整理后的批次字典
        """
        batch = {}
        
        # 处理input_ids和labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # 填充序列
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lbls in zip(input_ids, labels):
            pad_length = max_length - len(ids)
            
            # 填充input_ids
            padded_ids = torch.cat([ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
            padded_input_ids.append(padded_ids)
            
            # 填充labels
            padded_lbls = torch.cat([lbls, torch.full((pad_length,), IGNORE_INDEX)])
            padded_labels.append(padded_lbls)
            
            # 创建attention mask
            attention_mask = torch.ones(len(ids), dtype=torch.long)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
            attention_masks.append(attention_mask)
        
        # 不要在collate_fn中使用.to(device)，因为DataLoader worker进程不能重新初始化CUDA
        # Trainer会自动将数据移动到正确的设备
        batch['input_ids'] = torch.stack(padded_input_ids)
        batch['labels'] = torch.stack(padded_labels)
        batch['attention_mask'] = torch.stack(attention_masks)
        
        # 处理图像
        images = [f['images'] for f in features]
        batch['images'] = torch.stack(images)
        
        # 处理image_grid_thws
        image_grid_thws = [f['image_grid_thws'] for f in features]
        batch['image_grid_thws'] = image_grid_thws
        
        # 处理辅助图像和边界框（如果存在）
        if 'images_aux' in features[0] and features[0]['images_aux'] is not None:
            images_aux = [f['images_aux'] for f in features]
            batch['images_aux'] = torch.stack(images_aux)
        
        if 'bbox_list' in features[0] and features[0]['bbox_list'] is not None:
            bbox_lists = []
            for f in features:
                if f['bbox_list'] is not None:
                    bbox_lists.append(f['bbox_list'])
                else:
                    bbox_lists.append(torch.tensor([], dtype=torch.float32))
            batch['bbox_list'] = bbox_lists
        
        # 病理特征（按样本存 list[Tensor]，与 bbox_list 对齐）
        # 不能只看首样本，避免首样本为 None 时整批丢失病理特征。
        if any(('path_features' in f and f['path_features'] is not None) for f in features):
            path_features_batch = []
            for f in features:
                path_features_batch.append(f.get('path_features', None))
            batch['path_features'] = path_features_batch

        # task_id / 助手答案文本（用于阶段B辅助损失）
        batch['task_ids'] = [f.get('task_id', -1) for f in features]
        batch['assistant_texts'] = [f.get('assistant_text', "") for f in features]
        
        return batch


class CervicalTrainer(Trainer):
    """自定义Trainer，处理VLM-FO1的特殊输入格式"""

    def __init__(
        self,
        *args,
        tokenizer=None,
        phase="full",
        region_loss_weight=0.0,
        cls_loss_weight=0.0,
        region_loss_weight_by_task=None,
        cls_loss_weight_by_task=None,
        aux_log_every_steps=50,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.phase = phase
        self.region_loss_weight = float(region_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.region_loss_weight_by_task = region_loss_weight_by_task or {}
        self.cls_loss_weight_by_task = cls_loss_weight_by_task or {}
        self.aux_log_every_steps = max(1, int(aux_log_every_steps))

        # 运行期统计（用于每N step打印 batch 均值）
        self._aux_forward_counter = 0
        self._aux_region_sum = 0.0
        self._aux_cls_sum = 0.0
        self._aux_region_count = 0
        self._aux_cls_count = 0
        self._aux_diag_enabled = True
        self._input_diag_printed = False

    @staticmethod
    def _find_subseq_positions(seq_ids, pat_ids):
        if len(pat_ids) == 0 or len(seq_ids) == 0 or len(pat_ids) > len(seq_ids):
            return []
        pos = []
        plen = len(pat_ids)
        for i in range(len(seq_ids) - plen + 1):
            if seq_ids[i:i + plen] == pat_ids:
                pos.extend(range(i, i + plen))
        return pos

    def _build_region_token_weight_mask(self, labels, task_ids, assistant_texts):
        """
        构建用于 L_region 的 token weight mask：
        - 仅 task_id in {0,1}
        - 从 assistant_text 中提取 <regionX>，定位其在答案 token 序列中的位置
        """
        bsz, seqlen = labels.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.float32, device=labels.device)
        if self.processing_class is None:
            return mask

        for b in range(bsz):
            try:
                tid = int(task_ids[b])
            except Exception:
                tid = -1
            if tid not in (0, 1):
                continue
            w = float(self.region_loss_weight_by_task.get(str(tid), self.region_loss_weight))
            if w <= 0.0:
                continue

            ans_text = assistant_texts[b] if b < len(assistant_texts) else ""
            if not ans_text:
                continue
            region_mentions = re.findall(r"<region\d+>", ans_text)
            if not region_mentions:
                continue

            ans_pos = torch.where(labels[b] != IGNORE_INDEX)[0]
            if ans_pos.numel() == 0:
                continue
            ans_ids = labels[b, ans_pos].tolist()

            hit_local = set()
            for mention in region_mentions:
                pat = self.processing_class(mention, add_special_tokens=False).input_ids
                for p in self._find_subseq_positions(ans_ids, pat):
                    hit_local.add(p)

            for p in hit_local:
                if 0 <= p < ans_pos.numel():
                    mask[b, ans_pos[p]] = w
        return mask

    def _build_cls_token_weight_mask(self, labels, task_ids, assistant_texts):
        """
        构建用于 L_cls 的 token weight mask：
        - 仅 task_id in {3,4}
        - 在 assistant_text 中匹配类别名，定位到答案 token 序列
        """
        bsz, seqlen = labels.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.float32, device=labels.device)
        if self.processing_class is None:
            return mask

        for b in range(bsz):
            try:
                tid = int(task_ids[b])
            except Exception:
                tid = -1
            if tid not in (3, 4):
                continue
            w = float(self.cls_loss_weight_by_task.get(str(tid), self.cls_loss_weight))
            if w <= 0.0:
                continue

            ans_text = assistant_texts[b] if b < len(assistant_texts) else ""
            if not ans_text:
                continue

            candidates = TYPE3_CELL_TYPE_CLASSES if tid == 3 else TYPE4_SOURCE_CLASSES
            target = None
            # 优先精确匹配，其次包含匹配
            stripped = ans_text.strip()
            if stripped in candidates:
                target = stripped
            else:
                for c in candidates:
                    if c in ans_text:
                        target = c
                        break
            if target is None:
                continue

            ans_pos = torch.where(labels[b] != IGNORE_INDEX)[0]
            if ans_pos.numel() == 0:
                continue
            ans_ids = labels[b, ans_pos].tolist()
            pat = self.processing_class(target, add_special_tokens=False).input_ids
            local_positions = self._find_subseq_positions(ans_ids, pat)
            if len(local_positions) == 0:
                continue
            for p in local_positions:
                if 0 <= p < ans_pos.numel():
                    mask[b, ans_pos[p]] = w
        return mask

    @staticmethod
    def _masked_shift_ce(logits, labels, token_weight_mask):
        """
        在 causal shift 后，对 token_weight_mask>0 的位置计算加权 CE。
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = token_weight_mask[:, 1:].contiguous()

        # 多模态前向时，logits 序列长度可能大于文本 labels（视觉 token 插入后展开）。
        # 这里按尾部右对齐裁到共同长度，确保 CE 的输入/目标一一对应。
        if shift_logits.size(1) != shift_labels.size(1):
            common_len = min(shift_logits.size(1), shift_labels.size(1), shift_weights.size(1))
            if common_len <= 0:
                return None
            shift_logits = shift_logits[:, -common_len:, :].contiguous()
            shift_labels = shift_labels[:, -common_len:].contiguous()
            shift_weights = shift_weights[:, -common_len:].contiguous()

        valid_mask = (shift_weights > 0) & (shift_labels != IGNORE_INDEX)
        if valid_mask.sum().item() == 0:
            return None

        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view_as(shift_labels)
        weights = shift_weights[valid_mask]
        vals = per_tok[valid_mask]
        return (vals * weights).sum() / weights.sum().clamp(min=1e-6)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        计算损失
        
        Args:
            model: 模型对象
            inputs: 输入字典
            return_outputs: 是否返回输出
            
        Returns:
            损失值（如果return_outputs=True，返回(loss, outputs)）
        """
        # 首个 step 打印输入键，确认 task_ids/assistant_texts 未被 Trainer 预处理链路裁掉
        if not self._input_diag_printed:
            try:
                print(
                    f"[InputDiag] keys={sorted(list(inputs.keys()))}, "
                    f"has_task_ids={'task_ids' in inputs}, has_assistant_texts={'assistant_texts' in inputs}"
                )
            except Exception:
                pass
            self._input_diag_printed = True

        # 提取输入
        input_ids = inputs.pop('input_ids')
        labels = inputs.pop('labels')
        attention_mask = inputs.pop('attention_mask')
        images = inputs.pop('images')
        image_grid_thws = inputs.pop('image_grid_thws')
        
        # 准备模型输入
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'image_grid_thws': image_grid_thws,
            'labels': labels,
        }
        
        # 添加辅助图像和边界框（如果存在）
        if 'images_aux' in inputs:
            model_inputs['images_aux'] = inputs.pop('images_aux')
        if 'bbox_list' in inputs:
            model_inputs['bbox_list'] = inputs.pop('bbox_list')
        if 'path_features' in inputs:
            model_inputs['path_features'] = inputs.pop('path_features')
        task_ids = inputs.pop('task_ids', None)
        assistant_texts = inputs.pop('assistant_texts', None)
        
        # 前向传播
        outputs = model(**model_inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # 阶段B辅助损失：L_region + L_cls
        # 仅在 full 阶段启用，且需要 logits 可用。
        if (
            self.phase == "full"
            and hasattr(outputs, "logits")
            and outputs.logits is not None
            and task_ids is not None
            and assistant_texts is not None
        ):
            logits = outputs.logits
            aux_total = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            region_loss_val = None
            cls_loss_val = None

            region_mask_hit = 0
            cls_mask_hit = 0
            if self.region_loss_weight > 0.0 or len(self.region_loss_weight_by_task) > 0:
                region_mask = self._build_region_token_weight_mask(labels, task_ids, assistant_texts)
                region_mask_hit = int((region_mask[:, 1:] > 0).sum().item())
                region_loss = self._masked_shift_ce(logits, labels, region_mask)
                if region_loss is not None:
                    aux_total = aux_total + region_loss
                    region_loss_val = float(region_loss.detach().cpu())

            if self.cls_loss_weight > 0.0 or len(self.cls_loss_weight_by_task) > 0:
                cls_mask = self._build_cls_token_weight_mask(labels, task_ids, assistant_texts)
                cls_mask_hit = int((cls_mask[:, 1:] > 0).sum().item())
                cls_loss = self._masked_shift_ce(logits, labels, cls_mask)
                if cls_loss is not None:
                    aux_total = aux_total + cls_loss
                    cls_loss_val = float(cls_loss.detach().cpu())

            loss = loss + aux_total

            # 每 N step 打印辅助损失的 batch 均值
            self._aux_forward_counter += 1
            if region_loss_val is not None:
                self._aux_region_sum += region_loss_val
                self._aux_region_count += 1
            if cls_loss_val is not None:
                self._aux_cls_sum += cls_loss_val
                self._aux_cls_count += 1

            if self._aux_forward_counter % self.aux_log_every_steps == 0:
                region_mean = self._aux_region_sum / self._aux_region_count if self._aux_region_count > 0 else 0.0
                cls_mean = self._aux_cls_sum / self._aux_cls_count if self._aux_cls_count > 0 else 0.0
                print(
                    f"[AuxLoss][step={self.state.global_step}] "
                    f"L_region(mean/{self._aux_region_count})={region_mean:.6f}, "
                    f"L_cls(mean/{self._aux_cls_count})={cls_mean:.6f}"
                )
                if self._aux_diag_enabled:
                    try:
                        task_counter = Counter(int(t) for t in task_ids)
                    except Exception:
                        task_counter = Counter()
                    print(
                        f"[AuxDiag][step={self.state.global_step}] "
                        f"task_ids={dict(task_counter)}, "
                        f"region_mask_tokens={region_mask_hit}, cls_mask_tokens={cls_mask_hit}, "
                        f"region_loss_nonempty={int(region_loss_val is not None)}, "
                        f"cls_loss_nonempty={int(cls_loss_val is not None)}"
                    )
                self._aux_region_sum = 0.0
                self._aux_cls_sum = 0.0
                self._aux_region_count = 0
                self._aux_cls_count = 0
        
        return (loss, outputs) if return_outputs else loss


def load_stage1_model(base_model_path, stage1_model_path, device="cuda"):
    """
    加载第一阶段微调后的模型
    
    Args:
        base_model_path: 基础模型路径
        stage1_model_path: 第一阶段模型路径（包含LoRA权重）
        device: 设备
        
    Returns:
        tokenizer, model, image_processors
    """
    print(f"\n加载基础模型: {base_model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        base_model_path,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    
    # 检查是否有LoRA权重
    adapter_config_path = os.path.join(stage1_model_path, "adapter_config.json")
    adapter_model_path = os.path.join(stage1_model_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        if PEFT_AVAILABLE:
            print(f"加载第一阶段LoRA权重: {stage1_model_path}")
            model = PeftModel.from_pretrained(model, stage1_model_path)
            print("第一阶段LoRA权重加载成功")
            
            # 合并LoRA权重到基础模型，以便继续微调
            print("合并LoRA权重到基础模型...")
            model = model.merge_and_unload()
            print("LoRA权重已合并，可以继续微调")
        else:
            print("Warning: LoRA权重存在但PEFT不可用，使用基础模型")
    else:
        print(f"Warning: 未找到LoRA权重文件，使用基础模型")
        print(f"  查找路径: {adapter_config_path}")
        print(f"  查找路径: {adapter_model_path}")
    
    return tokenizer, model, image_processors


def setup_lora(model, lora_config):
    """
    设置LoRA配置并应用到模型
    
    Args:
        model: 模型对象
        lora_config: LoRA配置字典
        
    Returns:
        应用LoRA后的模型
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is not installed. Please install it with: pip install peft")
    
    # 确定目标模块
    target_modules = lora_config.get('target_modules', None)
    if target_modules is None:
        # 默认的LoRA目标模块（针对Qwen2.5-VL）
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",")]
    
    # 创建LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    # 应用LoRA到模型
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _enable_trainable_modules(model, trainable_modules):
    """
    启用指定模块参数训练。
    匹配规则：参数名等于目标、以“目标.”开头、或包含“.目标.”完整片段（避免子串误匹配）。
    兼容别名：
      - fusion_layer -> mm_projector_path
      - mm_projector -> mm_projector_aux（ROI token 投影）
    """
    alias_to_target = {
        "fusion_layer": "mm_projector_path",
        "mm_projector": "mm_projector_aux",
    }

    def _match_param_name(param_name: str, target: str) -> bool:
        if param_name == target:
            return True
        if param_name.startswith(target + "."):
            return True
        # 片段边界匹配，避免 "mm_projector_path" 误命中 "mm_projector_aux"
        return f".{target}." in f".{param_name}."

    resolved_targets = []
    for m in trainable_modules:
        m = str(m)
        resolved_targets.append(alias_to_target.get(m, m))

    enabled = 0
    enabled_names = []
    unmatched = set(resolved_targets)
    for name, param in model.named_parameters():
        for target in resolved_targets:
            if _match_param_name(name, target):
                unmatched.discard(target)
                if not param.requires_grad:
                    param.requires_grad = True
                enabled += param.numel()
                enabled_names.append(name)
                break

    if len(unmatched) > 0:
        print(f"Warning: trainable_modules not found: {sorted(list(unmatched))}")
        print("Available module prefixes:")
        prefixes = set()
        for n, _ in model.named_parameters():
            parts = n.split(".")
            if len(parts) >= 2:
                prefixes.add(".".join(parts[:2]))
        for p in sorted(prefixes):
            if any(u in p for u in unmatched):
                print(f"  - {p}")

    return enabled, enabled_names


def run_startup_sanity_check(train_dataset, data_collator, max_samples=8):
    """
    启动即检：在 trainer.train() 之前，快速验证 task_id/assistant_text/path_features 链路是否正常。
    仅打印诊断信息，不中断训练。
    """
    print("\n[StartupCheck] 开始数据链路快速检查...")
    try:
        ds = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
        n = min(max_samples, len(ds))

        # 检查原始JSON里 task 字段
        raw_counter = Counter()
        raw_task_id_missing = 0
        for i in range(n):
            item = ds.data[i] if hasattr(ds, "data") else None
            if not isinstance(item, dict):
                continue
            if item.get("task_id", None) is not None:
                try:
                    raw_counter[int(item.get("task_id"))] += 1
                except Exception:
                    raw_counter[str(item.get("task_id"))] += 1
            elif item.get("task_type", None) is not None:
                try:
                    raw_counter[int(item.get("task_type"))] += 1
                except Exception:
                    raw_counter[str(item.get("task_type"))] += 1
            else:
                raw_task_id_missing += 1
                raw_counter[-1] += 1

        print(f"[StartupCheck] raw task分布(前{n}条): {dict(raw_counter)}, 缺失task字段={raw_task_id_missing}")

        # 检查 __getitem__ + collator 后传给 Trainer 的字段
        batch_n = min(2, len(train_dataset))
        probe_samples = [train_dataset[i] for i in range(batch_n)]
        probe_batch = data_collator(probe_samples)

        task_ids = probe_batch.get("task_ids", [])
        task_counter = Counter()
        for t in task_ids:
            try:
                task_counter[int(t)] += 1
            except Exception:
                task_counter[str(t)] += 1

        assistant_texts = probe_batch.get("assistant_texts", [])
        empty_assistant = sum(1 for t in assistant_texts if not t)
        path_features = probe_batch.get("path_features", None)
        path_non_none = 0
        if isinstance(path_features, list):
            path_non_none = sum(1 for pf in path_features if pf is not None)

        print(
            f"[StartupCheck] collator task_ids={dict(task_counter)}, "
            f"assistant_text空样本={empty_assistant}/{len(assistant_texts)}, "
            f"path_features非空={path_non_none}/{len(path_features) if isinstance(path_features, list) else 0}"
        )
    except Exception as e:
        print(f"[StartupCheck] Warning: 启动即检失败: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train VLM-FO1 Stage 2 for cervical cell localization')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config_stage2.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    seed = config.get('misc', {}).get('seed', 42)
    set_seed(seed)
    
    # 设备配置
    device = config.get('hardware', {}).get('device', 'cuda')
    mixed_precision = config.get('hardware', {}).get('mixed_precision', 'bf16')
    
    # 分布式训练：设置每个进程使用对应的GPU
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        print(f"Process using GPU: {local_rank}")
    
    print("=" * 50)
    print("VLM-FO1 宫颈细胞定位第二阶段微调")
    print("=" * 50)
    
    # 加载基础模型（不再依赖第一阶段微调权重，直接从base_model开始，和一阶段一致）
    base_model_path = config['model']['base_model']
    print(f"\n加载模型: {base_model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        base_model_path,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置模型为训练模式
    model.train()
    
    # 启用gradient checkpointing以节省显存
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    elif hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # 检查训练阶段与是否使用LoRA
    training_config = config['training']
    phase = training_config.get('phase', 'full')  # "full" / "paffe_align" / "feature_fusion_warmup"
    aux_loss_cfg = training_config.get('aux_loss', {})
    region_loss_weight = float(aux_loss_cfg.get('region_loss_weight', 0.0))
    cls_loss_weight = float(aux_loss_cfg.get('cls_loss_weight', 0.0))
    region_loss_weight_by_task = aux_loss_cfg.get('region_loss_weight_by_task', {}) or {}
    cls_loss_weight_by_task = aux_loss_cfg.get('cls_loss_weight_by_task', {}) or {}
    aux_log_every_steps = int(aux_loss_cfg.get('log_every_steps', 50))
    use_lora = config.get('lora', {}).get('enabled', False) and phase == 'full'
    
    # 检查是否需要从阶段A的checkpoint加载（阶段A保存的是完整模型，不是LoRA）
    stage_a_checkpoint = None
    if args.resume_from_checkpoint and use_lora:
        # 检查是否是阶段A的checkpoint（完整模型）
        checkpoint_path = args.resume_from_checkpoint
        # 如果是目录，检查是否有adapter_config.json（LoRA checkpoint的标志）
        if os.path.isdir(checkpoint_path):
            adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                # 没有adapter_config.json，说明是完整模型checkpoint（阶段A）
                stage_a_checkpoint = checkpoint_path
                print(f"\n检测到阶段A的完整模型checkpoint: {stage_a_checkpoint}")
                print("将在应用LoRA之前加载阶段A的模型权重（包括PA-FFE模块）")
    
    if phase in ('paffe_align', 'feature_fusion_warmup'):
        # Pathology 融合预热阶段（Early Fusion）：
        # - 冻结主干
        # - 仅训练 trainable_modules 指定的融合相关模块
        #   默认：path_encoder + fusion_layer(mm_projector_path) + mm_projector(mm_projector_aux)
        print("\n使用 Pathology 融合预热阶段训练（feature_fusion_warmup/paffe_align）")
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        default_modules = ["path_encoder", "fusion_layer", "mm_projector"] \
            if phase == "feature_fusion_warmup" else ["path_encoder", "fusion_layer"]
        trainable_modules = training_config.get("trainable_modules", default_modules)
        enabled, enabled_names = _enable_trainable_modules(model, trainable_modules)

        freeze_cfg = config.get("freeze", {})
        if phase == "feature_fusion_warmup":
            if not freeze_cfg.get("vision_tower", True):
                print("Warning: feature_fusion_warmup 建议 freeze.vision_tower=true")
            if not freeze_cfg.get("language_model", True):
                print("Warning: feature_fusion_warmup 建议 freeze.language_model=true")

        if enabled > 0:
            print(f"Enabled trainable_modules={trainable_modules}, params={enabled}")
        else:
            print("Warning: no trainable params matched trainable_modules (will rely on lazy init after warmup).")

    elif use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA is enabled but PEFT is not installed. Please install it with: pip install peft")
        
        print("\n使用LoRA微调模式（第二阶段）")
        
        # ========= 关键：如果指定了阶段A的checkpoint，先加载完整模型权重 =========
        if stage_a_checkpoint is not None:
            print(f"\n加载阶段A的模型权重: {stage_a_checkpoint}")
            try:
                # 使用transformers的from_pretrained加载完整模型
                # 这会加载包括PA-FFE模块（path_mlp, path_gate）在内的所有权重
                from transformers import AutoModelForCausalLM
                # 注意：这里需要先获取模型的类型，然后加载
                # 由于我们的模型是自定义的，直接使用load_state_dict可能更安全
                checkpoint_state = torch.load(
                    os.path.join(stage_a_checkpoint, "pytorch_model.bin"),
                    map_location="cpu"
                ) if os.path.exists(os.path.join(stage_a_checkpoint, "pytorch_model.bin")) else None
                
                if checkpoint_state is None:
                    # 尝试加载safetensors格式
                    try:
                        from safetensors.torch import load_file
                        import json
                        index_file = os.path.join(stage_a_checkpoint, "model.safetensors.index.json")
                        if os.path.exists(index_file):
                            with open(index_file, 'r') as f:
                                index = json.load(f)
                            checkpoint_state = {}
                            # 去重weight_file，避免重复加载
                            weight_files = set(index.get("weight_map", {}).values())
                            for weight_file in weight_files:
                                weight_path = os.path.join(stage_a_checkpoint, weight_file)
                                if os.path.exists(weight_path):
                                    checkpoint_state.update(load_file(weight_path))
                                else:
                                    print(f"Warning: 权重文件不存在: {weight_path}")
                    except ImportError:
                        print("Warning: safetensors未安装，无法加载safetensors格式的checkpoint")
                    except Exception as e:
                        print(f"Warning: 加载safetensors格式checkpoint失败: {e}")
                
                if checkpoint_state:
                    # 过滤掉不需要的键（如optimizer state等）
                    model_state = {}
                    for k, v in checkpoint_state.items():
                        if not k.startswith("trainer_state") and not k.startswith("optimizer"):
                            model_state[k] = v
                    
                    # 加载模型权重
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                    if missing_keys:
                        print(f"Warning: 加载阶段A checkpoint时缺失的键: {len(missing_keys)} 个（可能是LoRA相关，正常）")
                    if unexpected_keys:
                        print(f"Warning: 加载阶段A checkpoint时意外的键: {len(unexpected_keys)} 个")
                    print("阶段A模型权重加载成功（包括PA-FFE模块）")
                else:
                    print(f"Warning: 无法从 {stage_a_checkpoint} 加载模型权重，将使用基础模型")
            except Exception as e:
                print(f"Warning: 加载阶段A checkpoint失败: {e}，将使用基础模型")
                import traceback
                traceback.print_exc()
        
        lora_config = config.get('lora', {})
        freeze_config = config.get('freeze', {})
        
        # 如果使用LoRA，先设置视觉编码器和投影层的requires_grad（如果需要）
        # 视觉编码器
        if not freeze_config.get('vision_tower', True):
            vision_tower = model.get_vision_tower()
            if vision_tower is not None:
                for param in vision_tower.parameters():
                    param.requires_grad = True
                print("Vision tower parameters enabled for training")
            
            vision_tower_aux = model.get_vision_tower_aux()
            if vision_tower_aux is not None:
                for param in vision_tower_aux.parameters():
                    param.requires_grad = True
                print("Auxiliary vision tower parameters enabled for training")
        
        # 对语言模型部分应用LoRA（第二阶段）
        print("应用LoRA到语言模型（第二阶段）...")
        model = setup_lora(model, lora_config)

        # full 阶段可通过 trainable_modules 控制融合路径训练模块；
        # 若未配置，默认训练：mm_projector_aux + path_encoder + fusion_layer(mm_projector_path)。
        full_trainable_modules = training_config.get("trainable_modules", ["mm_projector", "path_encoder", "fusion_layer"])
        full_trainable_modules = [m for m in full_trainable_modules if str(m) != "lora_layers"]
        enabled, _ = _enable_trainable_modules(model, full_trainable_modules)
        if enabled > 0:
            print(f"Enabled fusion trainable_modules={full_trainable_modules}, params={enabled}")
        else:
            print("Warning: full阶段未匹配到融合路径可训练参数（可能尚未懒初始化，将在warmup后出现）。")
    else:
        # 全参数微调模式
        print("\n使用全参数微调模式（第二阶段）")
        print("注意：第二阶段全参数微调会更新第一阶段的所有参数")
    
    # 准备数据集
    print("\n准备数据集...")
    data_config = config['data']
    train_json = data_config['train_json']
    test_json = data_config.get('test_json', None)
    image_size = data_config.get('image_size')
    max_regions = data_config.get('max_regions', 100)
    path_feature_root = data_config.get('path_feature_root', None)
    
    # 创建训练数据集
    print(f"加载训练集: {train_json}")
    train_dataset = CervicalDataset(
        json_path=train_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=base_model_path,
        max_regions=max_regions,
        image_size=image_size,
        is_train=True,
        verbose=False,
        path_feature_root=path_feature_root,
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建验证/测试数据集
    if test_json and os.path.exists(test_json):
        print(f"加载测试集: {test_json}")
        val_dataset = CervicalDataset(
            json_path=test_json,
            tokenizer=tokenizer,
            image_processors=image_processors,
            model_name=base_model_path,
            max_regions=max_regions,
            image_size=image_size,
            is_train=False,
            verbose=False,
            path_feature_root=path_feature_root,
        )
        print(f"测试集大小: {len(val_dataset)}")
    else:
        print("警告: 未找到测试集文件，将使用训练集的一部分作为验证集")
        # 如果没有测试集，使用训练集的一部分作为验证集
        train_dataset_full = train_dataset
        dataset_size = len(train_dataset_full)
        val_size = int(dataset_size * 0.1)  # 使用10%作为验证集
        train_size = dataset_size - val_size
        
        indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 创建Subset数据集
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
        val_dataset = torch.utils.data.Subset(train_dataset_full, val_indices)
        
        print(f"从训练集划分验证集: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
    
    # 创建数据整理器
    data_collator = CervicalDataCollator(tokenizer=tokenizer, device=device)
    run_startup_sanity_check(train_dataset, data_collator)

    # ------------------------------------------------------------------
    # Pathology 融合模块（path_encoder/mm_projector_path）是懒初始化。
    # 为确保其参数在 Trainer/optimizer 构建前可见，这里在需要时先做一次 warmup forward。
    # ------------------------------------------------------------------
    need_path_warmup = (path_feature_root is not None) and (phase in ("paffe_align", "feature_fusion_warmup", "full"))
    if need_path_warmup:
        try:
            warm_sample = train_dataset[0] if not isinstance(train_dataset, torch.utils.data.Subset) else train_dataset.dataset[train_dataset.indices[0]]
            warm_batch = data_collator([warm_sample])
            warm_inputs = {
                "input_ids": warm_batch["input_ids"].to(device),
                "attention_mask": warm_batch["attention_mask"].to(device),
                "labels": warm_batch["labels"].to(device),
                "images": warm_batch["images"].to(device),
                "image_grid_thws": warm_batch["image_grid_thws"],
            }
            if "images_aux" in warm_batch and warm_batch["images_aux"] is not None:
                warm_inputs["images_aux"] = warm_batch["images_aux"].to(device)
            if "bbox_list" in warm_batch and warm_batch["bbox_list"] is not None:
                warm_inputs["bbox_list"] = [
                    b.to(device) if b is not None else None
                    for b in warm_batch["bbox_list"]
                ]
            if "path_features" in warm_batch and warm_batch["path_features"] is not None:
                warm_inputs["path_features"] = [
                    p.to(device) if p is not None else None
                    for p in warm_batch["path_features"]
                ]

            model.eval()
            with torch.no_grad():
                _ = model(**warm_inputs)
            model.train()
            print("Pathology warmup forward done (path_encoder/mm_projector_path initialized).")
        except Exception as e:
            print(f"Warning: Pathology warmup forward failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # 训练配置
    output_dir = training_config['output_dir']
    
    # 检查是否有checkpoint
    # 注意：如果指定了阶段A的checkpoint（完整模型），已经在上面加载了，这里不应该再传给Trainer
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        # 如果是阶段A的checkpoint（完整模型），且当前是LoRA模式，则不作为resume_from_checkpoint
        # 因为阶段A的checkpoint已经在应用LoRA之前加载了
        if stage_a_checkpoint is None or not use_lora:
            checkpoint = args.resume_from_checkpoint
        elif os.path.isdir(output_dir):
            checkpoint = get_last_checkpoint(output_dir)
    
    # 配置wandb（如果启用）
    report_to = config.get('misc', {}).get('report_to', 'tensorboard')
    if report_to == 'wandb' and WANDB_AVAILABLE:
        wandb_config = config.get('wandb', {})
        wandb_project = wandb_config.get('project', 'vlm-fo1-cervical')
        wandb_entity = wandb_config.get('entity', None)
        wandb_name = wandb_config.get('name', None) or config.get('misc', {}).get('run_name', 'cervical_cell_finetune_stage2')
        wandb_tags = wandb_config.get('tags', [])
        wandb_notes = wandb_config.get('notes', '')
        
        # 初始化wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            tags=wandb_tags,
            notes=wandb_notes,
            config={
                'model': config.get('model', {}),
                'data': config.get('data', {}),
                'training': config.get('training', {}),
                'lora': config.get('lora', {}),
            }
        )
        print(f"Wandb initialized: project={wandb_project}, name={wandb_name}")
    elif report_to == 'wandb' and not WANDB_AVAILABLE:
        print("Warning: wandb is requested but not installed. Falling back to tensorboard.")
        report_to = 'tensorboard'
    
    # 创建训练参数
    # 确保学习率是浮点数类型
    learning_rate = training_config['learning_rate']
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    # 允许在配置文件中覆盖 eval/save 策略与 DDP timeout，避免阶段A频繁/超长 eval 导致 NCCL watchdog 超时
    eval_strategy = training_config.get("eval_strategy", "steps")  # "steps" | "epoch" | "no"
    save_strategy = training_config.get("save_strategy", "steps")
    ddp_timeout = training_config.get("ddp_timeout", 1800)  # seconds
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=learning_rate,
        warmup_steps=training_config['warmup_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        logging_steps=training_config['logging_steps'],
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_config.get('greater_is_better', False),
        bf16=mixed_precision == 'bf16',
        fp16=mixed_precision == 'fp16',
        dataloader_num_workers=config.get('hardware', {}).get('dataloader_num_workers', 4),
        dataloader_pin_memory=config.get('hardware', {}).get('dataloader_pin_memory', True),
        dataloader_drop_last=True,  # 丢弃最后一个不完整的batch，避免OOM
        gradient_checkpointing=True,  # 启用gradient checkpointing节省显存
        report_to=report_to,
        run_name=config.get('misc', {}).get('run_name', 'cervical_cell_finetune_stage2'),
        max_grad_norm=config.get('misc', {}).get('max_grad_norm', 1.0),
        eval_strategy=eval_strategy,  # 使用eval_strategy替代evaluation_strategy
        save_strategy=save_strategy,
        logging_dir=os.path.join(output_dir, 'logs'),
        # PA-FFE 对齐/融合阶段会存在在部分 step 中某些参数未参与前向的情况，
        # 为避免DDP报 "Expected to have finished reduction..." 错误，这里显式开启 unused 参数检测。
        ddp_find_unused_parameters=True,
        ddp_timeout=ddp_timeout,
        # 自定义辅助损失需要这些字段，禁止 Trainer 自动列裁剪。
        remove_unused_columns=False,
        # PEFT 下显式声明 label_names，确保字段保留策略稳定。
        label_names=["labels", "task_ids", "assistant_texts"],
    )
    
    # 创建Trainer
    trainer = CervicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # 使用processing_class替代tokenizer
        tokenizer=tokenizer,
        phase=phase,
        region_loss_weight=region_loss_weight,
        cls_loss_weight=cls_loss_weight,
        region_loss_weight_by_task=region_loss_weight_by_task,
        cls_loss_weight_by_task=cls_loss_weight_by_task,
        aux_log_every_steps=aux_log_every_steps,
    )

    # 训练前打印可训练参数统计，便于核查分阶段策略是否按预期生效
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = (trainable_params / total_params) if total_params > 0 else 0.0
    print(f"\n{'=' * 50}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {trainable_ratio:.2%}")
    print(f"{'=' * 50}\n")
    
    # 开始训练
    print("\n开始第二阶段训练...")
    print("=" * 50)
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存最终模型
    print("\n保存最终模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 如果使用LoRA，保存LoRA权重
    if use_lora and hasattr(trainer.model, 'save_pretrained'):
        lora_output_dir = os.path.join(output_dir, "lora")
        os.makedirs(lora_output_dir, exist_ok=True)
        trainer.model.save_pretrained(lora_output_dir)
        print(f"LoRA权重已保存到: {lora_output_dir}")
    
    print("\n第二阶段训练完成！")
    print(f"模型保存在: {output_dir}")
    
    # 结束wandb运行
    if report_to == 'wandb' and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished.")


if __name__ == '__main__':
    main()

