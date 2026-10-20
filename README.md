# CerviVLM

Cervical cytology VQA with hierarchical region feature fusion on a VLM-FO1 (Qwen2.5-VL) backbone.

Two-stage LoRA training:

1. **Stage 1** — domain adaptation (region alignment)
2. **Stage 2** — CoT multi-task fine-tuning

## Environment

Recommended: conda + Python 3.10, CUDA GPU.

```bash
conda create -n cervivlm python=3.10 -y
conda activate cervivlm

# Install PyTorch matching your CUDA (example: CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Project dependencies (pin transformers==4.50.1)
pip install -r requirements.txt

# Common extras used by training / LoRA
pip install peft==0.10.0 deepspeed qwen-vl-utils sentencepiece protobuf pillow opencv-python
```

Optional: `flash-attn` can speed up attention; if the wheel fails to build, the code falls back via `utils/flash_attn_compat.py`.

Place the VLM-FO1 base checkpoint at:

```text
resources/VLM-FO1_Qwen2.5-VL-3B-v01
```

Edit data paths in `configs/*.yaml` (use placeholders such as `/path/to/dataset/...`).

## Dataset format

Each split is a **JSON list**. Every sample includes `task_id`, `id`, `image`, `regions`, and `conversations`.

Roles must be `"user"` / `"assistant"`. Each region has `region_id`, `bbox` `[x1, y1, x2, y2]` (absolute pixels), and optional `confidence`.

**Localization example (`task_id=0`):**

```json
{
  "task_id": 0,
  "id": "identity_4374",
  "image": "/path/to/images/sample_a.jpg",
  "regions": [
    {
      "region_id": "region0",
      "bbox": [368.47, 776.83, 584.67, 1023.93],
      "confidence": 0.61
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nPlease mark the locations of positive cervical cells in the image."
    },
    {
      "from": "assistant",
      "value": "<ground>positive cervical cells</ground><objects><region0></objects>"
    }
  ]
}
```

**Region description example (`task_id=2`):**

```json
{
  "task_id": 2,
  "id": "identity_10623",
  "image": "/path/to/images/sample_b.jpg",
  "regions": [
    {
      "region_id": "region0",
      "bbox": [867.72, 325.73, 1002.81, 453.94],
      "confidence": 0.26
    },
    {
      "region_id": "region1",
      "bbox": [287.62, 465.11, 438.91, 657.80],
      "confidence": 0.25
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nDescribe the cell in <region0>."
    },
    {
      "from": "assistant",
      "value": "Positive cervical cell. Type: HSIL. Origin: cervical squamous basal layer. The cell shows enlarged, irregular and dark-stained nuclei."
    }
  ]
}
```

Notes:

- User text should start with `<image>`; region references use `<region0>`, `<region1>`, …
- Localization answers use `<ground>…</ground><objects><regionk></objects>`
- Point `data.train_json` / `data.val_json` in the YAMLs to your splits
- For localization metrics in eval, set `ORIGIN_JSON` (GT boxes) and `EVAL_JSON`

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
