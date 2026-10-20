#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM-FO1 宫颈细胞 Stage-1 对齐训练脚本（不使用 path_feature）

数据集：
- 训练集: data/stage1/stage1_align_train.json
- 验证集: data/stage1/stage1_align_val.json

任务：
- task_id = 0：定位阴性宫颈细胞（带 <ground>/<objects>/<regionX>）
- task_id = 1：根据描述“找出阴性细胞”进行定位
- task_id = 2：对指定 <regionX> 区域做病理形态描述（caption）
"""

import os
import sys
import argparse

# 添加项目根目录到路径（需在 transformers 之前加载 env compat）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.flash_attn_compat  # noqa: F401

import torch
from transformers import TrainingArguments, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from scripts.train_cervical_stage2 import (
    setup_lora,
    load_config,
    _enable_trainable_modules,
    HIERARCHICAL_FUSION_PEFT_MODULE,
)
from vlm_fo1.constants import IGNORE_INDEX
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import HierarchicalFeatureFusion
from scripts.vt_fpn_utils import (
    apply_vt_fpn_ablation,
    peft_modules_to_save_for_fpn,
    resolve_fpn_num_levels,
    save_fpn_ablation_meta,
)


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
        if task_ids is not None:
            model_inputs["task_ids"] = task_ids

        outputs = model(**model_inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 output_dir 下存在 checkpoint-*，则从最新检查点继续训练（optimizer/步数等由 Trainer 恢复）",
    )
    parser.add_argument(
        "--fpn_num_levels",
        type=int,
        default=None,
        help="覆盖 yaml 中 model.fpn_num_levels（1/2/4/6）",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="训练期间不跑验证集（训练结束后由 eval_stage1_align.py 单独评估）",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="训练期间按 eval_steps 跑验证（覆盖 yaml 中 no_val: true）",
    )
    return parser.parse_args()


def _enable_hierarchical_fusion_if_needed(model, model_config: dict):
    enabled = bool(model_config.get("use_hierarchical_fusion", False))
    heads = int(model_config.get("hierarchical_num_heads", 8))
    fusion_mode = str(model_config.get("hierarchical_fusion_mode", "legacy")).strip().lower()
    if fusion_mode not in ("legacy", "task_adaptive", "softmax_gate", "gated", "mean", "concat", "weighted"):
        fusion_mode = "legacy"
    if not enabled:
        return

    core = model.get_model() if hasattr(model, "get_model") else model
    extractor = getattr(core, "object_vp_extractor", None)
    if extractor is None:
        print("Warning: object_vp_extractor not found, skip hierarchical fusion enable.")
        return
    if not getattr(extractor, "use_simpleFPN_for_vt", False):
        print("Warning: hierarchical fusion currently expects mm_use_simpleFPN_for_vt=true. Skip.")
        return

    num_levels = int(getattr(extractor, "fpn_num_levels", 4))
    extractor.use_hierarchical_fusion = True
    extractor.hierarchical_num_heads = heads
    extractor.hierarchical_fusion_mode = fusion_mode

    existing = getattr(extractor, "hierarchical_fusion", None)
    need_rebuild = (
        existing is None
        or getattr(existing, "num_levels", 4) != num_levels
        or getattr(existing, "fusion_mode", fusion_mode) != fusion_mode
    )
    if need_rebuild:
        extractor.hierarchical_fusion = HierarchicalFeatureFusion(
            feature_dim=512,
            out_dim=2048,
            num_heads=heads,
            fusion_mode=fusion_mode,
            num_levels=num_levels,
        )
    elif hasattr(extractor.hierarchical_fusion, "fusion_mode"):
        extractor.hierarchical_fusion.fusion_mode = fusion_mode
    ref_param = next(model.parameters())
    extractor.hierarchical_fusion.to(device=ref_param.device, dtype=ref_param.dtype)
    print(
        f"[Stage1] Hierarchical fusion enabled "
        f"(heads={heads}, mode={fusion_mode}, num_levels={num_levels})."
    )


def _print_hierarchical_gate_trainability(model):
    """Startup diagnostics: verify hierarchical gate parameters are trainable."""
    gate_named = []
    gate_trainable = 0
    gate_total = 0
    for name, param in model.named_parameters():
        if "hierarchical_fusion.gate_network" in name:
            gate_named.append((name, bool(param.requires_grad), param.numel()))
            gate_total += param.numel()
            if param.requires_grad:
                gate_trainable += param.numel()

    if gate_total == 0:
        print("[Stage1 GateDiag] No gate_network parameters found (check hierarchical fusion mount).")
        return

    print(
        f"[Stage1 GateDiag] gate_network trainable params: "
        f"{gate_trainable:,}/{gate_total:,} ({(100.0 * gate_trainable / gate_total):.2f}%)"
    )
    for name, req_grad, numel in gate_named:
        flag = "T" if req_grad else "F"
        print(f"[Stage1 GateDiag] [{flag}] {name} ({numel:,})")


def main():
    args = parse_args()
    config = load_config(args.config)

    model_config = config.get("model", {})
    if args.fpn_num_levels is not None:
        model_config["fpn_num_levels"] = int(args.fpn_num_levels)
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
    language = data_config.get("language", "zh")

    output_dir = training_config["output_dir"]
    num_epochs = training_config.get("num_epochs", 5)
    batch_size = training_config.get("batch_size", 1)
    grad_accum = training_config.get("gradient_accumulation_steps", 8)
    learning_rate = training_config.get("learning_rate", 1e-4)
    warmup_steps = training_config.get("warmup_steps", 50)
    save_steps = training_config.get("save_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    logging_steps = training_config.get("logging_steps", 10)
    load_best_at_end = training_config.get("load_best_model_at_end", True)
    no_val = bool(training_config.get("no_val", False)) or args.no_val
    if args.val:
        no_val = False
    if no_val:
        load_best_at_end = False
    if load_best_at_end and not no_val and eval_steps > 0 and save_steps % eval_steps != 0:
        raise ValueError(
            "load_best_model_at_end=True 时，HF Trainer 要求 save_steps 为 eval_steps 的整数倍；"
            f"当前 save_steps={save_steps}, eval_steps={eval_steps}。"
            "请修改 training.save_steps / eval_steps，或设 load_best_model_at_end: false。"
        )
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
    print(f"Val JSON:     {val_json if not no_val else '(训练期间跳过，结束后单独 eval)'}")
    print(f"Output dir:   {output_dir}")
    print(f"No val train: {no_val}")
    print("============================================================")

    tokenizer, model, image_processors = load_pretrained_model(
        base_model, load_8bit=False, load_4bit=False, device=device
    )

    fpn_num_levels = resolve_fpn_num_levels(model_config)
    use_hf = bool(model_config.get("use_hierarchical_fusion", False))
    # 注意：不要把 task_adaptive/gated 强改成 legacy。
    # FPN 层数消融请直接使用 configs/train_config_stage1_fpn_ablation_*.yaml（已设 legacy）。

    fpn_meta = apply_vt_fpn_ablation(
        model,
        fpn_num_levels=fpn_num_levels,
        use_hierarchical_fusion=use_hf,
    )
    save_fpn_ablation_meta(output_dir, fpn_meta)

    _enable_hierarchical_fusion_if_needed(model, model_config)

    # 冻结视觉编码器
    if freeze_config.get("vision_tower", True):
        print("Freezing vision_tower parameters...")
        for name, param in model.named_parameters():
            if "vision_tower" in name or "get_vision_tower" in name:
                param.requires_grad = False

    # 应用 LoRA 到 LLM（可选：将 hierarchical_fusion 纳入 PEFT modules_to_save，以便写入 adapter）
    modules_to_save = None
    save_list = []
    if lora_config.get("enabled", False) and bool(model_config.get("use_hierarchical_fusion")):
        core = model.get_model() if hasattr(model, "get_model") else model
        _ex = getattr(core, "object_vp_extractor", None)
        if _ex is not None and getattr(_ex, "hierarchical_fusion", None) is not None:
            save_list.append(HIERARCHICAL_FUSION_PEFT_MODULE)
    save_list.extend(peft_modules_to_save_for_fpn(model, model_config))
    if save_list:
        modules_to_save = save_list
        print(f"[Stage1] PEFT modules_to_save (full weights in adapter): {modules_to_save}")

    if lora_config.get("enabled", False):
        print("Applying LoRA to language model...")
        model = setup_lora(model, lora_config, modules_to_save=modules_to_save)

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
        language=language,
    )
    val_dataset = None
    if not no_val:
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
            language=language,
        )

    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val dataset size:   {len(val_dataset)}")
    else:
        print("Val dataset:        (disabled during training)")

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
        eval_steps=eval_steps if val_dataset is not None else None,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_at_end and val_dataset is not None,
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
        evaluation_strategy="no" if val_dataset is None else "steps",
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
    _print_hierarchical_gate_trainability(model)

    trainer = CervicalTrainerStage1(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        tokenizer=tokenizer,
    )

    resume_from = None
    if args.resume:
        resume_from = get_last_checkpoint(output_dir)
        if resume_from:
            print(f"[Stage1] Resume from checkpoint: {resume_from}")
        else:
            print(f"[Stage1] --resume 已指定但 {output_dir} 下无 checkpoint-*，将从头训练。")

    print("\n===== 开始 Stage-1 训练 =====")
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    trainer.save_state()
    print("===== Stage-1 训练完成，模型已保存 =====")


if __name__ == "__main__":
    main()