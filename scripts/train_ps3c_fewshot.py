#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PS3C few-shot 二分类微调：从 Stage2 checkpoint 加载，在 5/10/20-shot 数据上 LoRA 微调。

用法:
  python scripts/train_ps3c_fewshot.py --config configs/train_config_ps3c_fewshot.yaml \\
    --train_json .../ps3c_5shot_seed1.json --val_json .../ps3c_5shot_seed1_test.json \\
    --output_dir outputs/ps3c_5shot_seed1_finetune --run_name ps3c_5shot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 Stage2 训练基础设施
from scripts import train_stage2_cot as stage2_train
from dataset.ps3c_classification_dataset import PS3CClassificationDataset

DEFAULT_VLM_FO1_BASE = stage2_train.DEFAULT_VLM_FO1_BASE


def parse_args():
    parser = argparse.ArgumentParser(description="PS3C few-shot 二分类微调")
    parser.add_argument("--config", type=str, default="configs/train_config_ps3c_fewshot.yaml")
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 output_dir 存在 checkpoint-*，从最新检查点续训",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="强制从头训练（忽略已有 checkpoint）",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="指定续训 checkpoint 目录（如 outputs/.../checkpoint-100）",
    )
    parser.add_argument("--no_val", action="store_true", help="训练时不跑验证（测试集很大时推荐）")
    return parser.parse_args()


def _resolve_resume_checkpoint(output_dir: str, cli) -> str | None:
    """解析续训 checkpoint；默认 RESUME_TRAINING=1 时自动从最新 checkpoint 续训。"""
    from transformers.trainer_utils import get_last_checkpoint

    if cli.no_resume:
        print("[Resume] --no_resume：从头训练")
        return None

    if cli.resume_from_checkpoint:
        ckpt = os.path.abspath(cli.resume_from_checkpoint)
        if os.path.isdir(ckpt):
            print(f"[Resume] 使用指定检查点: {ckpt}")
            return ckpt
        raise FileNotFoundError(f"resume_from_checkpoint 不存在: {ckpt}")

    resume_env = str(os.environ.get("RESUME_TRAINING", "1")).strip()
    auto_resume = resume_env not in ("0", "false", "False", "no", "NO")
    last_ckpt = get_last_checkpoint(output_dir)

    if cli.resume or (auto_resume and last_ckpt):
        if last_ckpt:
            print(f"[Resume] 从最新检查点续训: {last_ckpt}")
            return last_ckpt
        if cli.resume:
            print(f"[Resume] --resume 已指定但 {output_dir} 下无 checkpoint-*，从头训练。")
        return None

    if last_ckpt:
        print(
            f"[Resume] 发现已有检查点 {last_ckpt}，但未续训。"
            " 设置 RESUME_TRAINING=1（默认）或加 --resume 可从断点继续。"
        )
    return None


def main():
    cli = parse_args()
    config = stage2_train.load_config(cli.config)

    data_config = config.setdefault("data", {})
    training_config = config.setdefault("training", {})
    misc_config = config.setdefault("misc", {})

    if cli.train_json:
        data_config["train_json"] = cli.train_json
    if cli.val_json:
        data_config["val_json"] = cli.val_json
    if cli.output_dir:
        training_config["output_dir"] = cli.output_dir
    if cli.run_name:
        misc_config["run_name"] = cli.run_name
    if cli.num_epochs is not None:
        training_config["num_epochs"] = cli.num_epochs
    if cli.learning_rate is not None:
        training_config["learning_rate"] = cli.learning_rate
    if cli.gradient_accumulation_steps is not None:
        training_config["gradient_accumulation_steps"] = cli.gradient_accumulation_steps
    if cli.save_steps is not None:
        training_config["save_steps"] = cli.save_steps
    if cli.eval_steps is not None:
        training_config["eval_steps"] = cli.eval_steps

    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    freeze_config = config.get("freeze", {})
    hardware_config = config.get("hardware", {})

    base_model = model_config.get("base_model", "")
    load_stageA_lora = model_config.get("load_stageA_lora", True)
    device = hardware_config.get("device", "cuda")

    train_json = data_config.get("train_json", "")
    val_json = data_config.get("val_json", "")
    enrich_cache_dir = data_config.get("enrich_cache_dir")
    image_size = data_config.get("image_size", 896)
    max_regions = data_config.get("max_regions", 1)
    path_feature_root = data_config.get("path_feature_root")
    language = data_config.get("language", "zh")

    output_dir = training_config.get("output_dir", "outputs/ps3c_fewshot")
    num_epochs = training_config.get("num_epochs", 30)
    batch_size = training_config.get("batch_size", 1)
    grad_accum = training_config.get("gradient_accumulation_steps", 2)
    learning_rate = float(training_config.get("learning_rate", 2e-4))
    gate_lr_ratio = float(training_config.get("gate_lr_ratio", 0.0))
    warmup_ratio = float(training_config.get("warmup_ratio", 0.05))
    save_steps = training_config.get("save_steps", 10)
    eval_steps = training_config.get("eval_steps", 10)
    logging_steps = training_config.get("logging_steps", 2)
    save_total_limit = training_config.get("save_total_limit", 3)
    weight_decay = float(training_config.get("weight_decay", 0.01))
    trainable_modules = training_config.get("trainable_modules", ["lora_layers"])
    early_stopping_patience = training_config.get("early_stopping_patience", 8)

    mixed_precision = hardware_config.get("mixed_precision", "bf16")
    gradient_checkpointing = hardware_config.get("gradient_checkpointing", True)
    num_workers = hardware_config.get("dataloader_num_workers", 2)
    pin_memory = hardware_config.get("dataloader_pin_memory", True)
    seed = misc_config.get("seed", 42)
    max_grad_norm = misc_config.get("max_grad_norm", 1.0)
    report_to = misc_config.get("report_to", "tensorboard")
    run_name = misc_config.get("run_name", "ps3c_fewshot")

    stage2_train.set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("========== PS3C Few-shot 微调 ==========")
    print(f"Config:     {cli.config}")
    print(f"Base ckpt:  {base_model}")
    print(f"Train JSON: {train_json}")
    print(f"Val JSON:   {val_json if not cli.no_val else '(disabled)'}")
    print(f"Output:     {output_dir}")
    print(f"Epochs:     {num_epochs}  LR: {learning_rate}  accum: {grad_accum}")
    print("========================================")

    tokenizer, model, image_processors = stage2_train._load_stageA_model(
        base_model, load_stageA_lora, device, model_config=model_config
    )
    stage2_train._enable_hierarchical_fusion_if_needed(model, model_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    stage2_train._freeze_stageB(model, freeze_config)

    modules_to_save = None
    if lora_config.get("enabled", True) and bool(model_config.get("use_hierarchical_fusion")):
        core = model.get_model() if hasattr(model, "get_model") else model
        extractor = getattr(core, "object_vp_extractor", None)
        if extractor is not None and getattr(extractor, "hierarchical_fusion", None) is not None:
            if "object_vp_extractor.hierarchical_fusion" in trainable_modules:
                modules_to_save = [stage2_train.HIERARCHICAL_FUSION_PEFT_MODULE]

    if lora_config.get("enabled", True):
        if not stage2_train.PEFT_AVAILABLE:
            raise ImportError("PEFT is required for PS3C few-shot LoRA training.")
        print("应用 PS3C few-shot LoRA...")
        model = stage2_train.setup_lora(model, lora_config, modules_to_save=modules_to_save)

    enabled, _ = stage2_train._enable_trainable_modules(model, trainable_modules)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量 (requires_grad): {total_trainable:,}  (trainable_modules 匹配: {enabled:,})")
    if total_trainable == 0:
        raise RuntimeError("没有可训练参数，请检查 LoRA 是否已正确挂载。")

    def _make_ds(json_path, is_train):
        return PS3CClassificationDataset(
            json_path=json_path,
            tokenizer=tokenizer,
            image_processors=image_processors,
            model_name=DEFAULT_VLM_FO1_BASE,
            max_regions=max_regions,
            image_size=image_size,
            is_train=is_train,
            verbose=False,
            path_feature_root=path_feature_root,
            language=language,
            enrich_cache_dir=enrich_cache_dir,
        )

    train_dataset = _make_ds(train_json, True)
    val_dataset = None
    if not cli.no_val and val_json and os.path.exists(val_json):
        val_dataset = _make_ds(val_json, False)

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples:   {len(val_dataset)} (全量测试集，训练中 eval 较慢时可 --no_val)")

    data_collator = stage2_train.CervicalDataCollatorStage2LeftPad(tokenizer=tokenizer)
    stage2_train.run_startup_sanity_check(train_dataset, data_collator)

    from transformers import TrainingArguments, EarlyStoppingCallback

    resume_from = _resolve_resume_checkpoint(output_dir, cli)

    bf16 = mixed_precision == "bf16"
    fp16 = mixed_precision == "fp16"
    total_steps_est = max(1, (len(train_dataset) // max(1, batch_size * grad_accum)) * num_epochs)
    warmup_steps = max(1, int(total_steps_est * warmup_ratio))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if val_dataset else None,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=bool(val_dataset) and training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_drop_last=False,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        run_name=run_name,
        max_grad_norm=max_grad_norm,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        label_names=["labels", "task_ids", "assistant_texts"],
        weight_decay=weight_decay,
    )

    callbacks = []
    if val_dataset is not None:
        try:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        except ImportError:
            pass

    trainer = stage2_train.StageBCoTTrainer(
        train_sampler=None,
        gate_lr_ratio=gate_lr_ratio,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        tokenizer=tokenizer,
    )
    if callbacks:
        trainer.add_callback(callbacks[0])

    print("\n===== 开始 PS3C few-shot 训练 =====")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    trainer.save_state()

    if lora_config.get("enabled", True) and hasattr(trainer.model, "save_pretrained"):
        lora_dir = os.path.join(output_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        trainer.model.save_pretrained(lora_dir)
        print(f"LoRA 已保存到: {lora_dir}")

    print("===== PS3C few-shot 训练完成 =====")


if __name__ == "__main__":
    main()
