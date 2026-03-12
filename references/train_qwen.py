# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from qwenvl.train.custom_trainer import CustomTrainer

# LoRA 支持
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will be disabled.")

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model, use_lora=False):
    """
    设置模型的可训练参数。
    如果 use_lora=True，则使用 LoRA 而不是全参数微调。
    """
    if use_lora and PEFT_AVAILABLE:
        # LoRA 模式下，不需要手动设置 requires_grad
        # LoRA 会自动处理
        return
    
    # 全参数微调模式
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def setup_lora(model, model_args):
    """
    设置 LoRA 配置并应用到模型。
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is not installed. Please install it with: pip install peft")
    
    # 确定目标模块
    if model_args.lora_target_modules:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    else:
        # 默认的 LoRA 目标模块（针对 Qwen2.5-VL）
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 创建 LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    # 应用 LoRA 到模型
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    if local_rank == 0 or local_rank is None:
        model.print_trainable_parameters()
    
    return model


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # 设置 LoRA 或全参数微调
    if model_args.use_lora:
        rank0_print("Using LoRA for fine-tuning")
        # LoRA 模式下，先设置视觉和 MLP 的 requires_grad（如果需要）
        if model_args.tune_mm_vision:
            for n, p in model.visual.named_parameters():
                p.requires_grad = True
        if model_args.tune_mm_mlp:
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = True
        
        # 只对 LLM 部分应用 LoRA
        if model_args.tune_mm_llm:
            model = setup_lora(model, model_args)
    else:
        rank0_print("Using full parameter fine-tuning")
        set_model(model_args, model, use_lora=False)
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 使用自定义Trainer，添加IoU损失和分类损失
    # 损失权重可以通过训练参数设置，这里使用默认值
    use_custom_loss = getattr(model_args, 'use_custom_loss', False)
    if use_custom_loss:
        iou_loss_weight = getattr(model_args, 'iou_loss_weight', 1.0)
        bbox_loss_weight = getattr(model_args, 'bbox_loss_weight', 1.0)
        use_hybrid_bbox_head = getattr(model_args, 'use_hybrid_bbox_head', False)
        hybrid_bbox_loss_weight = getattr(model_args, 'hybrid_bbox_loss_weight', 0.5)
        
        rank0_print(f"Using CustomTrainer with additional losses:")
        rank0_print(f"  IoU loss weight: {iou_loss_weight}")
        rank0_print(f"  Bbox loss weight: {bbox_loss_weight}")
        if use_hybrid_bbox_head:
            rank0_print(f"  Hybrid bbox head: Enabled")
            rank0_print(f"  Hybrid bbox loss weight: {hybrid_bbox_loss_weight}")
            rank0_print(f"  Note: Using ground truth labels for training, hybrid head provides additional supervision")
        else:
            rank0_print(f"  Hybrid bbox head: Disabled")
        
        trainer = CustomTrainer(
            model=model, 
            processing_class=tokenizer, 
            args=training_args,
            iou_loss_weight=iou_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            use_hybrid_bbox_head=use_hybrid_bbox_head,
            hybrid_bbox_loss_weight=hybrid_bbox_loss_weight,
            **data_module
        )
    else:
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # 如果使用 LoRA，保存 LoRA 权重
    if model_args.use_lora and hasattr(trainer.model, "save_pretrained"):
        lora_output_dir = os.path.join(training_args.output_dir, "lora")
        os.makedirs(lora_output_dir, exist_ok=True)
        trainer.model.save_pretrained(lora_output_dir)
        rank0_print(f"LoRA weights saved to {lora_output_dir}")


if __name__ == "__main__":
    # 使用 eager 实现以避免与新版 Transformers 的 FlashAttention 掩码逻辑不兼容
    train(attn_implementation="eager")
