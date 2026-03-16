#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞 Stage-1 对齐训练脚本（不使用 path_feature）

数据集：
- 训练集: /home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_train.json
- 验证集: /home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json

任务：
- task_id = 0：定位阴性宫颈细胞（带 <ground>/<objects>/<regionX>）
- task_id = 1：根据描述“找出阴性细胞”进行定位
- task_id = 2：对指定 <regionX> 区域做病理形态描述（caption）
"""

import os
import sys
import argparse
from collections import Counter

import torch
from transformers import TrainingArguments, Trainer, set_seed

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from scripts.train_cervical_stage2 import setup_lora, load_config, _enable_trainable_modules
from vlm_fo1.constants import IGNORE_INDEX


class CervicalDataCollatorStage1:
    """Stage-1 专用数据整理器（不含 path_features）"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}

        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, lbls in zip(input_ids, labels):
            pad_length = max_length - len(ids)

            padded_ids = torch.cat(
                [ids, torch.full((pad_length,), self.tokenizer.pad_token_id)]
            )
            padded_input_ids.append(padded_ids)

            padded_lbls = torch.cat(
                [lbls, torch.full((pad_length,), IGNORE_INDEX)]
            )
            padded_labels.append(padded_lbls)

            attn = torch.ones(len(ids), dtype=torch.long)
            attn = torch.cat([attn, torch.zeros(pad_length, dtype=torch.long)])
            attention_masks.append(attn)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["labels"] = torch.stack(padded_labels)
        batch["attention_mask"] = torch.stack(attention_masks)

        images = [f["images"] for f in features]
        batch["images"] = torch.stack(images)

        image_grid_thws = [f["image_grid_thws"] for f in features]
        batch["image_grid_thws"] = image_grid_thws

        if "images_aux" in features[0] and features[0]["images_aux"] is not None:
            images_aux = [f["images_aux"] for f in features]
            batch["images_aux"] = torch.stack(images_aux)

        if "bbox_list" in features[0] and features[0]["bbox_list"] is not None:
            bbox_lists = []
            for f in features:
                if f["bbox_list"] is not None:
                    bbox_lists.append(f["bbox_list"])
                else:
                    bbox_lists.append(torch.tensor([], dtype=torch.float32))
            batch["bbox_list"] = bbox_lists
        else:
            batch["bbox_list"] = None

        # task_id / assistant_text 目前仅用于统计，不参与额外 loss
        batch["task_ids"] = [f.get("task_id", -1) for f in features]
        batch["assistant_texts"] = [f.get("assistant_text", "") for f in features]

        return batch


class CervicalTrainerStage1(Trainer):
    """Stage-1 Trainer：只用标准 LM loss，不带辅助损失。"""

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self._input_diag_printed = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 首个 step 打印输入键，确认数据管线
        if not self._input_diag_printed:
            try:
                print(
                    f"[Stage1 InputDiag] keys={sorted(list(inputs.keys()))}, "
                    f"has_task_ids={'task_ids' in inputs}, "
                    f"has_assistant_texts={'assistant_texts' in inputs}"
                )
            except Exception:
                pass
            self._input_diag_printed = True

        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        attention_mask = inputs.pop("attention_mask")
        images = inputs.pop("images")
        image_grid_thws = inputs.pop("image_grid_thws")
        task_ids = inputs.pop("task_ids", None)
        assistant_texts = inputs.pop("assistant_texts", None)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "image_grid_thws": image_grid_thws,
            "labels": labels,
        }

        if "images_aux" in inputs:
            model_inputs["images_aux"] = inputs.pop("images_aux")
        if "bbox_list" in inputs:
            model_inputs["bbox_list"] = inputs.pop("bbox_list")

        outputs = model(**model_inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # 可选：每 N step 打印一次 task 分布，方便确认三种任务是否都在训练
        if self.state.global_step % 100 == 0 and task_ids is not None:
            try:
                counter = Counter(int(t) for t in task_ids)
                print(f"[Stage1 TaskDist][step={self.state.global_step}] {dict(counter)}")
            except Exception:
                pass

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM-FO1 Cervical Stage-1 domain adaptation with LoRA (no path_features)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config_stage1_lora.yaml",
        help="Stage-1 训练配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})
    freeze_config = config.get("freeze", {})
    hardware_config = config.get("hardware", {})
    misc_config = config.get("misc", {})

    base_model = model_config["base_model"]
    train_json = data_config["train_json"]
    val_json = data_config["val_json"]
    image_size = data_config.get("image_size", 896)
    max_regions = data_config.get("max_regions", 100)

    output_dir = training_config["output_dir"]
    num_epochs = training_config.get("num_epochs", 5)
    batch_size = training_config.get("batch_size", 1)
    grad_accum = training_config.get("gradient_accumulation_steps", 8)
    learning_rate = training_config.get("learning_rate", 1e-4)
    warmup_steps = training_config.get("warmup_steps", 50)
    save_steps = training_config.get("save_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    logging_steps = training_config.get("logging_steps", 10)
    save_total_limit = training_config.get("save_total_limit", 5)

    optimizer_cfg = training_config.get("optimizer", {})
    weight_decay = optimizer_cfg.get("weight_decay", 0.05)
    lr_sched_cfg = training_config.get("lr_scheduler", {})
    scheduler_type = lr_sched_cfg.get("type", "cosine")
    warmup_ratio = lr_sched_cfg.get("warmup_ratio", None)

    device = hardware_config.get("device", "cuda")
    mixed_precision = hardware_config.get("mixed_precision", "bf16")
    num_workers = hardware_config.get("dataloader_num_workers", 4)
    pin_memory = hardware_config.get("dataloader_pin_memory", True)

    seed = misc_config.get("seed", 42)
    max_grad_norm = misc_config.get("max_grad_norm", 1.0)
    report_to = misc_config.get("report_to", "tensorboard")
    run_name = misc_config.get("run_name", "cervical_stage1_domain_adapt_lora")

    set_seed(seed)

    print("========== Stage-1 领域适配（LoRA，无 path_feature） ==========")
    print(f"Config:       {args.config}")
    print(f"Base model:   {base_model}")
    print(f"Train JSON:   {train_json}")
    print(f"Val JSON:     {val_json}")
    print(f"Output dir:   {output_dir}")
    print("============================================================")

    tokenizer, model, image_processors = load_pretrained_model(
        base_model, load_8bit=False, load_4bit=False, device=device
    )

    # 冻结视觉编码器
    if freeze_config.get("vision_tower", True):
        print("Freezing vision_tower parameters...")
        for name, param in model.named_parameters():
            if "vision_tower" in name or "get_vision_tower" in name:
                param.requires_grad = False

    # 应用 LoRA 到 LLM
    if lora_config.get("enabled", False):
        print("Applying LoRA to language model...")
        model = setup_lora(model, lora_config)

    # 构建数据集
    train_dataset = CervicalDataset(
        json_path=train_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=base_model,
        max_regions=max_regions,
        image_size=image_size,
        is_train=True,
        verbose=False,
        path_feature_root=None,  # 显式禁用 path_feature
    )
    val_dataset = CervicalDataset(
        json_path=val_json,
        tokenizer=tokenizer,
        image_processors=image_processors,
        model_name=base_model,
        max_regions=max_regions,
        image_size=image_size,
        is_train=False,
        verbose=False,
        path_feature_root=None,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    data_collator = CervicalDataCollatorStage1(tokenizer=tokenizer)

    # 启用 Stage-1 启动检查（与 Stage-2 统一风格）
    from scripts.train_cervical_stage2 import run_startup_sanity_check
    data_collator = CervicalDataCollatorStage1(tokenizer=tokenizer)
    run_startup_sanity_check(train_dataset, data_collator)

    # TrainingArguments
    mp = mixed_precision
    bf16 = mp == "bf16"
    fp16 = mp == "fp16"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=run_name,
        max_grad_norm=max_grad_norm,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        label_names=["labels", "task_ids", "assistant_texts"],
    )

    # 按配置限制可训练模块（LoRA + projector）
    trainable_modules = training_config.get(
        "trainable_modules", ["lora_layers", "mm_projector", "mm_projector_aux"]
    )
    enabled, _ = _enable_trainable_modules(model, trainable_modules)
    print(f"Enabled trainable parameters in Stage-1: {enabled:,}")

    trainer = CervicalTrainerStage1(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        tokenizer=tokenizer,
    )

    print("\n===== 开始 Stage-1 训练 =====")
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    print("===== Stage-1 训练完成，模型已保存 =====")


if __name__ == "__main__":
    main()