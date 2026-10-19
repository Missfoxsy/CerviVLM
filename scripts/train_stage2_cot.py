#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StageB CoT 推理注入训练

- 从 StageA checkpoint 加载（完整模型或 LoRA），在其上应用更小的 LoRA（仅 Attention）。
- 冻结：vision_encoder、region_encoder、vision_projector、region_projector、LLM MLP。
- 训练 LoRA（q_proj, k_proj, v_proj, o_proj）并保留 hierarchical_fusion 为可训练/可保存，
  使门控在 StageB 的 6 任务上继续适配。
- 可选：混合 StageA 数据（stageA_data_ratio）缓解遗忘。
"""

import os
import sys
import json
import gc
import inspect
import argparse
from contextlib import contextmanager
from collections import Counter
from typing import Any, Dict, Optional

# Project root must be on path before transformers (flash_attn ABI shield).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.flash_attn_compat  # noqa: F401  — hide broken flash_attn before transformers

import torch
import torch.utils.data
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from dataset.cervical_dataset import CervicalDataset
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.constants import IGNORE_INDEX
from scripts.train_cervical_stage2 import (
    load_config,
    setup_lora,
    _enable_trainable_modules,
    HIERARCHICAL_FUSION_PEFT_MODULE,
    run_startup_sanity_check,
)
from scripts.train_stage1_align import CervicalTrainerStage1
from vlm_fo1.constants import IGNORE_INDEX
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import HierarchicalFeatureFusion


class CervicalDataCollatorStage2LeftPad:
    """Stage2 专用：左填充，满足 Qwen2.5-VL + FlashAttention 对 padding_side='left' 的要求。"""

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
            # 左填充：先 pad 再 真实序列，使 attention_mask 最后一列全为 1
            padded_ids = torch.cat([
                torch.full((pad_length,), self.tokenizer.pad_token_id),
                ids,
            ])
            padded_lbls = torch.cat([
                torch.full((pad_length,), IGNORE_INDEX),
                lbls,
            ])
            attn = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ])
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_masks.append(attn)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["labels"] = torch.stack(padded_labels)
        batch["attention_mask"] = torch.stack(attention_masks)
        batch["images"] = torch.stack([f["images"] for f in features])
        batch["image_grid_thws"] = [f["image_grid_thws"] for f in features]
        if "images_aux" in features[0] and features[0]["images_aux"] is not None:
            batch["images_aux"] = torch.stack([f["images_aux"] for f in features])
        else:
            batch["images_aux"] = None
        if "bbox_list" in features[0] and features[0]["bbox_list"] is not None:
            batch["bbox_list"] = [f["bbox_list"] if f["bbox_list"] is not None else torch.tensor([], dtype=torch.float32) for f in features]
        else:
            batch["bbox_list"] = None
        batch["task_ids"] = [f.get("task_id", -1) for f in features]
        batch["assistant_texts"] = [f.get("assistant_text", "") for f in features]
        return batch

try:
    from peft import LoraConfig, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    PeftModel = None
    PEFT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 默认 VLM-FO1 基座（用于从 StageA LoRA 目录加载时先加载基座）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VLM_FO1_BASE = os.path.join(_PROJECT_ROOT, "resources", "VLM-FO1_Qwen2.5-VL-3B-v01")


def _peft_model_from_pretrained_compat(model, adapter_path: str):
    """
    加载 LoRA 权重。若 adapter_config.json 含当前 peft 不支持的字段
    （如 alora_invocation_tokens），则按 LoraConfig 签名过滤后再加载。
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT 不可用")
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return PeftModel.from_pretrained(model, adapter_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if str(raw.get("peft_type", "LORA")).upper() != "LORA":
        return PeftModel.from_pretrained(model, adapter_path)
    sig = inspect.signature(LoraConfig.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    dropped = sorted(k for k in raw if k not in allowed)
    if dropped:
        print(f"[PEFT compat] 忽略 adapter_config 中当前 peft 不支持的键: {dropped}")
    try:
        lora_config = LoraConfig(**filtered)
    except TypeError as e:
        raise RuntimeError(
            f"无法根据 {cfg_path} 构造 LoraConfig: {e}。"
            "若 adapter 由较新版本训练产生，可尝试: pip install -U 'peft>=0.14'"
        ) from e
    return PeftModel.from_pretrained(model, adapter_path, config=lora_config)


@contextmanager
def _peft_adapter_weights_load_on_cpu():
    """
    PEFT 的 load_adapter 里 torch_device = infer_device()，只要 CUDA 可用就会用 cuda，
    与基座是否在 CPU 无关，导致 adapter safetensors 在 GPU 上解压而 OOM。
    加载 StageA LoRA 时强制走 CPU，再在 merge 后一次性 .to(cuda)。
    """
    import peft.peft_model as peft_pm
    import peft.utils.other as peft_other
    import peft.utils.save_and_load as peft_sl

    def _cpu():
        return "cpu"

    patched = []
    for mod in (peft_other, peft_sl, peft_pm):
        if hasattr(mod, "infer_device"):
            patched.append((mod, mod.infer_device))
            mod.infer_device = _cpu
    try:
        yield
    finally:
        for mod, fn in patched:
            mod.infer_device = fn


def _load_stageA_model(
    base_model_path: str,
    load_stageA_lora: bool,
    device: str,
    model_config: Optional[Dict[str, Any]] = None,
):
    """
    加载 StageA 模型。若 base_model_path 为 LoRA 目录则先加载基座再加载 LoRA 并 merge；
    否则尝试作为完整 checkpoint 加载（需路径含 vlm-fo1 或从基座 + state_dict 灌入）。

    若 StageA adapter 使用 PEFT ``modules_to_save`` 保存了 ``hierarchical_fusion``，必须在
    ``PeftModel.from_pretrained`` 之前挂载同名子模块，否则门控权重无法加载。
    """
    adapter_path = os.path.join(base_model_path, "adapter_config.json")
    has_adapter = os.path.exists(adapter_path)

    if has_adapter and load_stageA_lora and PEFT_AVAILABLE:
        # 基座 + LoRA 在 GPU 上先全量再 merge 会瞬时占双倍显存；先在 CPU 上 merge 再一次性迁入 GPU
        print(f"从 LoRA 目录加载 StageA: {base_model_path}（CPU 上合并 LoRA 以降低显存峰值）")
        tokenizer, model, image_processors = load_pretrained_model(
            DEFAULT_VLM_FO1_BASE, load_8bit=False, load_4bit=False, device="cpu"
        )
        if model_config and bool(model_config.get("use_hierarchical_fusion")):
            _enable_hierarchical_fusion_if_needed(model, model_config)
            print(
                "[StageB] 已在加载 StageA PEFT adapter 前挂载 hierarchical_fusion，"
                "以匹配 adapter 中的 modules_to_save 权重。"
            )
        with _peft_adapter_weights_load_on_cpu():
            model = _peft_model_from_pretrained_compat(model, base_model_path)
            model = model.merge_and_unload()
        print("StageA LoRA 已合并到基座")
        gc.collect()
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        tgt = device
        if isinstance(tgt, str) and tgt == "cuda":
            tgt = "cuda:0"
        model = model.to(device=tgt, dtype=torch.bfloat16)
        return tokenizer, model, image_processors

    _leaf = os.path.basename(os.path.normpath(base_model_path)).lower()
    if "vlm-fo1" in _leaf:
        print(f"加载 StageA 完整模型: {base_model_path}")
        tokenizer, model, image_processors = load_pretrained_model(
            base_model_path, load_8bit=False, load_4bit=False, device=device
        )
        return tokenizer, model, image_processors

    # 非 vlm-fo1 路径：先加载基座，再灌入 checkpoint 权重
    print(f"加载基座后灌入 StageA 权重: {base_model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        DEFAULT_VLM_FO1_BASE, load_8bit=False, load_4bit=False, device=device
    )
    bin_path = os.path.join(base_model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        ckpt = torch.load(bin_path, map_location="cpu")
        state = {k: v for k, v in ckpt.items() if not k.startswith("trainer_state") and not k.startswith("optimizer")}
        model.load_state_dict(state, strict=False)
        print("StageA 完整权重已加载")
    else:
        try:
            from safetensors.torch import load_file
            idx_path = os.path.join(base_model_path, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                state = {}
                for wf in set(index.get("weight_map", {}).values()):
                    state.update(load_file(os.path.join(base_model_path, wf)))
                model.load_state_dict(state, strict=False)
                print("StageA safetensors 权重已加载")
        except Exception as e:
            print(f"Warning: 无法加载 StageA 权重: {e}")
    return tokenizer, model, image_processors


def _freeze_stageB(model, freeze_config: dict):
    """冻结 vision、projector、LLM MLP，只保留 LoRA 可训。"""
    for name, param in model.named_parameters():
        if "vision_tower" in name or "get_vision_tower" in name:
            if freeze_config.get("vision_tower", True):
                param.requires_grad = False
        elif "mm_projector" in name and freeze_config.get("vision_projector", True):
            param.requires_grad = False
        elif "mm_projector_aux" in name and freeze_config.get("region_projector", True):
            param.requires_grad = False
        elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            if freeze_config.get("llm_mlp", True):
                param.requires_grad = False


def _enable_hierarchical_fusion_if_needed(model, model_config: dict):
    """
    Runtime switch for hierarchical fusion in HFRE.
    This allows enabling the module without editing pretrained config.json.
    """
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

    extractor.use_hierarchical_fusion = True
    extractor.hierarchical_num_heads = heads
    extractor.hierarchical_fusion_mode = fusion_mode
    if not hasattr(extractor, "hierarchical_fusion"):
        # Keep dimensionality compatible with existing VT branch output.
        extractor.hierarchical_fusion = HierarchicalFeatureFusion(
            feature_dim=512,
            out_dim=2048,
            num_heads=heads,
            fusion_mode=fusion_mode,
        )
    elif hasattr(extractor.hierarchical_fusion, "fusion_mode"):
        extractor.hierarchical_fusion.fusion_mode = fusion_mode
    ref_param = next(model.parameters())
    extractor.hierarchical_fusion.to(device=ref_param.device, dtype=ref_param.dtype)
    print(f"[StageB] Hierarchical fusion enabled (heads={heads}, mode={fusion_mode}).")


class StageBCoTTrainer(CervicalTrainerStage1):
    """StageB CoT 训练器：支持加权采样与 hierarchical_fusion 分组学习率。"""

    def __init__(self, train_sampler=None, gate_lr_ratio=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler
        self._gate_lr_ratio = float(gate_lr_ratio)
        self._gate_key = "object_vp_extractor.hierarchical_fusion"

    def get_train_dataloader(self):
        if self._train_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
        gate_lr = self.args.learning_rate * self._gate_lr_ratio

        decay_default, decay_gate = [], []
        no_decay_default, no_decay_gate = [], []
        decay_default_n, decay_gate_n = 0, 0
        no_decay_default_n, no_decay_gate_n = 0, 0

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_gate = self._gate_key in name
            use_no_decay = any(k in name for k in no_decay_keys)
            n = p.numel()
            if use_no_decay and is_gate:
                no_decay_gate.append(p)
                no_decay_gate_n += n
            elif use_no_decay:
                no_decay_default.append(p)
                no_decay_default_n += n
            elif is_gate:
                decay_gate.append(p)
                decay_gate_n += n
            else:
                decay_default.append(p)
                decay_default_n += n

        param_groups = []
        if decay_default:
            param_groups.append(
                {"params": decay_default, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate}
            )
        if no_decay_default:
            param_groups.append(
                {"params": no_decay_default, "weight_decay": 0.0, "lr": self.args.learning_rate}
            )
        if decay_gate:
            param_groups.append(
                {"params": decay_gate, "weight_decay": self.args.weight_decay, "lr": gate_lr}
            )
        if no_decay_gate:
            param_groups.append(
                {"params": no_decay_gate, "weight_decay": 0.0, "lr": gate_lr}
            )

        self.optimizer = AdamW(
            param_groups, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon
        )
        print(
            "[StageB] Optimizer param groups: "
            f"default_lr={self.args.learning_rate:.3e}, gate_lr={gate_lr:.3e} (ratio={self._gate_lr_ratio:.3f})"
        )
        print(
            "[StageB] Params by group: "
            f"decay_default={decay_default_n:,}, no_decay_default={no_decay_default_n:,}, "
            f"decay_gate={decay_gate_n:,}, no_decay_gate={no_decay_gate_n:,}"
        )
        return self.optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="StageB CoT 推理注入训练")
    parser.add_argument("--config", type=str, default="configs/train_config_stageB_lora.yaml", help="配置文件路径")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 output_dir 下存在 checkpoint-*，则从最新检查点继续训练",
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

    base_model = model_config.get("base_model", "")
    load_stageA_lora = model_config.get("load_stageA_lora", True)
    device = hardware_config.get("device", "cuda")

    train_json = data_config.get("train_json", "")
    val_json = data_config.get("val_json", "")
    mix_stageA = data_config.get("mix_stageA_data", False)
    stageA_ratio = float(data_config.get("stageA_data_ratio", 0.3))
    stageA_json = data_config.get("stageA_train_json", "")
    image_size = data_config.get("image_size", 896)
    max_regions = data_config.get("max_regions", 100)
    path_feature_root = data_config.get("path_feature_root")
    language = data_config.get("language", "zh")

    output_dir = training_config.get("output_dir", "outputs/stageB_cot_lora")
    num_epochs = training_config.get("num_epochs", 8)
    batch_size = training_config.get("batch_size", 2)
    grad_accum = training_config.get("gradient_accumulation_steps", 8)
    learning_rate = float(training_config.get("learning_rate", 5e-5))
    gate_lr_ratio = float(training_config.get("gate_lr_ratio", 0.2))
    warmup_ratio = float(training_config.get("warmup_ratio", 0.1))
    save_steps = training_config.get("save_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    eval_strategy = str(training_config.get("eval_strategy", "steps" if val_json else "no")).strip().lower()
    if eval_strategy not in ("no", "steps", "epoch"):
        eval_strategy = "steps" if val_json else "no"
    logging_steps = training_config.get("logging_steps", 10)
    save_total_limit = training_config.get("save_total_limit", 3)
    weight_decay = float(training_config.get("weight_decay", 0.1))
    trainable_modules = training_config.get("trainable_modules", ["lora_layers"])
    early_stopping_patience = training_config.get("early_stopping_patience", 3)

    mixed_precision = hardware_config.get("mixed_precision", "bf16")
    gradient_checkpointing = hardware_config.get("gradient_checkpointing", True)
    num_workers = hardware_config.get("dataloader_num_workers", 4)
    pin_memory = hardware_config.get("dataloader_pin_memory", True)
    seed = misc_config.get("seed", 42)
    max_grad_norm = misc_config.get("max_grad_norm", 1.0)
    report_to = misc_config.get("report_to", "tensorboard")
    run_name = misc_config.get("run_name", "stageB_cot_lora")

    # 配置 wandb（与 train_cervical_stage2 一致）：训练/验证 loss 会由 Trainer 自动上报
    # 超算无 tty 且未配置 WANDB_API_KEY 时，自动降级为 tensorboard，避免训练直接失败。
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    if report_to == "wandb" and WANDB_AVAILABLE:
        wandb_mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
        has_api_key = bool(str(os.environ.get("WANDB_API_KEY", "")).strip())
        non_interactive = not sys.stdin.isatty()
        if non_interactive and (not has_api_key) and wandb_mode not in ("offline", "disabled"):
            print(
                "Warning: report_to=wandb 但当前无 tty 且未配置 WANDB_API_KEY，"
                "自动回退到 tensorboard。若需 wandb，请设置 WANDB_API_KEY 或 WANDB_MODE=offline。"
            )
            report_to = "tensorboard"
        else:
            wandb_config = config.get("wandb", {})
            wandb_project = wandb_config.get("project", "vlm-fo1-cervical")
            wandb_entity = wandb_config.get("entity", None)
            wandb_name = wandb_config.get("name", None) or run_name
            wandb_tags = wandb_config.get("tags", [])
            wandb_notes = wandb_config.get("notes", "")
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    tags=wandb_tags,
                    notes=wandb_notes,
                    config={
                        "model": config.get("model", {}),
                        "data": config.get("data", {}),
                        "training": config.get("training", {}),
                        "lora": config.get("lora", {}),
                    },
                )
                print(f"Wandb initialized: project={wandb_project}, name={wandb_name}")
            except Exception as e:
                print(f"Warning: wandb 初始化失败，自动回退到 tensorboard。原因: {e}")
                report_to = "tensorboard"
    elif report_to == "wandb" and not WANDB_AVAILABLE:
        print("Warning: wandb is requested but not installed. Falling back to tensorboard.")
        report_to = "tensorboard"

    set_seed(seed)

    print("========== StageB CoT 推理注入（LoRA 仅 Attention） ==========")
    print(f"Config:       {args.config}")
    print(f"Base (StageA): {base_model}")
    print(f"Train JSON:   {train_json}")
    print(f"Val JSON:     {val_json}")
    print(f"Mix StageA:   {mix_stageA}, ratio={stageA_ratio}")
    print(f"Output dir:   {output_dir}")
    print("==============================================================")

    tokenizer, model, image_processors = _load_stageA_model(
        base_model, load_stageA_lora, device, model_config=model_config
    )
    _enable_hierarchical_fusion_if_needed(model, model_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen2.5-VL + FlashAttention: batched forward/eval requires left padding
    tokenizer.padding_side = "left"

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    _freeze_stageB(model, freeze_config)

    modules_to_save = None
    if lora_config.get("enabled", True) and bool(model_config.get("use_hierarchical_fusion")):
        core = model.get_model() if hasattr(model, "get_model") else model
        _extractor = getattr(core, "object_vp_extractor", None)
        if _extractor is not None and getattr(_extractor, "hierarchical_fusion", None) is not None:
            # Keep Stage2 gates trainable/savable with LoRA so task3/4/5 can continue adapting.
            modules_to_save = [HIERARCHICAL_FUSION_PEFT_MODULE]
            print(f"[StageB] PEFT modules_to_save (full weights in adapter): {modules_to_save}")

    if lora_config.get("enabled", True):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for StageB LoRA.")
        print("应用 StageB LoRA（仅 q/k/v/o_proj）...")
        model = setup_lora(model, lora_config, modules_to_save=modules_to_save)

    enabled, _ = _enable_trainable_modules(model, trainable_modules)
    print(f"StageB 可训练参数量: {enabled:,}")
    hf_key = "object_vp_extractor.hierarchical_fusion"
    hf_total = 0
    hf_trainable = 0
    for name, param in model.named_parameters():
        if hf_key in name:
            n = param.numel()
            hf_total += n
            if param.requires_grad:
                hf_trainable += n
    if hf_total > 0:
        hf_ratio = 100.0 * hf_trainable / hf_total
        print(
            f"[StageB] hierarchical_fusion 参数统计: "
            f"trainable={hf_trainable:,} / total={hf_total:,} ({hf_ratio:.2f}%)"
        )
    else:
        print(f"[StageB] Warning: 未找到包含 `{hf_key}` 的参数名，无法统计 hierarchical_fusion 参数量。")

    # 数据集
    def _make_ds(json_path, is_train):
        return CervicalDataset(
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
        )

    train_dataset = _make_ds(train_json, True)
    val_dataset = _make_ds(val_json, False) if os.path.exists(val_json) else None

    train_sampler = None
    if mix_stageA and stageA_json and os.path.exists(stageA_json) and 0 < stageA_ratio < 1:
        stageA_ds = _make_ds(stageA_json, True)
        n_a, n_b = len(stageA_ds), len(train_dataset)
        w_a = stageA_ratio / max(n_a, 1)
        w_b = (1 - stageA_ratio) / max(n_b, 1)
        weights = [w_a] * n_a + [w_b] * n_b
        combined = torch.utils.data.ConcatDataset([stageA_ds, train_dataset])
        num_samples = len(combined)
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples)
        train_dataset = combined
        print(f"混合数据集: StageA={n_a}, StageB={n_b}, ratio≈{stageA_ratio}")

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples:   {len(val_dataset)}")

    data_collator = CervicalDataCollatorStage2LeftPad(tokenizer=tokenizer)
    run_startup_sanity_check(train_dataset, data_collator)

    bf16 = mixed_precision == "bf16"
    fp16 = mixed_precision == "fp16"
    total_steps_est = (len(train_dataset) // (batch_size * grad_accum)) * num_epochs
    warmup_steps = max(1, int(total_steps_est * warmup_ratio))

    load_best_model_at_end = bool(training_config.get("load_best_model_at_end", True))
    if eval_strategy == "no":
        load_best_model_at_end = False

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
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_drop_last=True,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        run_name=run_name,
        max_grad_norm=max_grad_norm,
        eval_strategy=eval_strategy if val_dataset else "no",
        save_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        label_names=["labels", "task_ids", "assistant_texts"],
        weight_decay=weight_decay,
    )

    callbacks = []
    if val_dataset is not None and eval_strategy != "no":
        try:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        except ImportError:
            pass

    trainer = StageBCoTTrainer(
        train_sampler=train_sampler,
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

    resume_from = None
    if args.resume:
        resume_from = get_last_checkpoint(output_dir)
        if resume_from:
            print(f"[StageB] Resume from checkpoint: {resume_from}")
        else:
            print(f"[StageB] --resume 已指定但 {output_dir} 下无 checkpoint-*，将从头训练。")

    print("\n===== 开始 StageB 训练 =====")
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    trainer.save_state()

    if lora_config.get("enabled", True) and hasattr(trainer.model, "save_pretrained"):
        lora_dir = os.path.join(output_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        trainer.model.save_pretrained(lora_dir)
        print(f"LoRA 已保存到: {lora_dir}")

    print("===== StageB 训练完成 =====")

    if report_to == "wandb" and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished.")


if __name__ == "__main__":
    main()
