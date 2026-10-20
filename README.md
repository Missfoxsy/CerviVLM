# CerviVLM

Cervical cytology VQA with hierarchical region feature fusion on a VLM-FO1 (Qwen2.5-VL) backbone.

Two-stage LoRA training:

1. **Stage 1** — domain adaptation (region alignment)
2. **Stage 2** — CoT multi-task fine-tuning

## Setup

```bash
pip install -r requirements.txt
# Place the VLM-FO1 base checkpoint under:
#   resources/VLM-FO1_Qwen2.5-VL-3B-v01
```

Pin `transformers==4.50.1` (see `requirements.txt`).

Edit data paths in `configs/*.yaml` (placeholders like `/path/to/dataset/...`).

## Dataset format

Each split is a JSON list. Every sample needs `image`, `regions`, and `conversations`:

```json
[
  {
    "image": "data/images/xxx.jpg",
    "regions": [
      {"bbox": [x1, y1, x2, y2]}
    ],
    "conversations": [
      {"from": "human", "value": "<image>\n<region>\nYour question..."},
      {"from": "gpt", "value": "Answer..."}
    ]
  }
]
```

- `bbox`: absolute pixel coordinates `[x1, y1, x2, y2]` with `x1 < x2`, `y1 < y2`
- Stage 1 JSON: localization / region-alignment dialogues
- Stage 2 JSON: CoT multi-task dialogues (description, classification, diagnosis, etc.)
- Evaluation may also need an origin JSON with GT boxes for localization metrics; set `ORIGIN_JSON` when running eval

Point configs to your files, for example:

- `data.train_json` / `data.val_json` in Stage-1 / Stage-2 YAMLs
- `EVAL_JSON`, `ORIGIN_JSON` env vars for evaluation scripts

## Quick start (main pipeline)

```bash
# Full pipeline: Stage1 train → Stage1 eval → Stage2 train → Stage2 eval
START_FROM=stage1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_two_stage_and_eval.sh

# Resume from Stage2 only
START_FROM=stage2 CUDA_VISIBLE_DEVICES=0 bash scripts/run_two_stage_and_eval.sh

# Eval only
START_FROM=eval CUDA_VISIBLE_DEVICES=0 bash scripts/run_two_stage_and_eval.sh
```

Default configs:

- Stage 1: `configs/train_config_stage1_lora.yaml` (`task_adaptive` fusion)
- Stage 2: `configs/train_config_stageB_lora.yaml`

Or train / eval separately:

```bash
bash scripts/train_stage1_align.sh
bash scripts/eval_stage1_align.sh
bash scripts/train_stage2_cot.sh
bash scripts/eval_stage2_cot.sh
```

## Hyperparameters (paper defaults)

| Item | Stage 1 | Stage 2 |
|------|---------|---------|
| Epochs | 1 | 1 |
| Batch size | 1 | 1 |
| Grad accum | 8 | 8 |
| Learning rate | 1e-4 | 5e-5 |
| Image size | 896 | 896 |
| Precision | bf16 | bf16 |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA dropout | 0.1 | 0.05 |
| LoRA targets | q,k,v,o + MLP | q,k,v,o (attn only) |
| Weight decay | 0.05 | 0.1 |
| Fusion heads | 8 | 8 |
| Gate LR ratio | — | 0.2 |
| Freeze vision tower | yes | yes |

Main fusion mode: `hierarchical_fusion_mode: task_adaptive` (Sigmoid Gate).

## Ablations (optional)

**Matched Stage1+Stage2 fusion modes** (`legacy` / `concat` / `softmax_gate` / `task_adaptive`):

```bash
FUSION_MODE=task_adaptive bash scripts/run_matched_fusion_ablation.sh
# or: legacy | concat | softmax_gate
```

**FPN depth** (Stage 1, legacy attention):

```bash
bash scripts/run_stage1_fpn_ablation.sh
```

Configs live under `configs/train_config_stage1_fpn_ablation_*.yaml` and `configs/train_config_stage*_lora_*.yaml`.

**Stage2 without Stage1**:

```bash
bash scripts/run_stage2_ablation_no_stage1.sh
```

**Zero-shot external sets** (after training):

```bash
python scripts/eval_ps3c_classification.py ...
python scripts/eval_sipakmed_posneg.py ...
```

## Project layout

```
configs/     # training YAMLs
dataset/     # CervicalDataset loaders
scripts/     # train / eval / ablation entrypoints
vlm_fo1/     # model (HFRE fusion, FPN, encoders)
utils/       # helpers (e.g. flash-attn compatibility)
```

## Notes

- Set `CUDA_VISIBLE_DEVICES` explicitly on your machine.
- Checkpoints are written to `training.output_dir` in each YAML (under `outputs/` by default).
- For multi-GPU Stage2: `CUDA_VISIBLE_DEVICES=0,1 NPROC_STAGE2=2 bash scripts/run_two_stage_and_eval.sh`
