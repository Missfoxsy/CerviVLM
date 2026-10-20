#!/bin/bash
# StageB CoT 推理注入训练（小 LoRA 仅 Attention，冻结 vision/projector/MLP）
# 用法: bash scripts/train_stage2_cot.sh
# 指定 GPU: CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage2_cot.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

ENTRY_FILE="scripts/train_stage2_cot.py"
CONFIG="${1:-configs/train_config_stageB_lora.yaml}"

echo "=========================================="
echo "VLM-FO1 StageB CoT 推理注入训练"
echo "=========================================="
echo "Entry file:       ${ENTRY_FILE}"
echo "Config:           ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR:      ${MASTER_ADDR}"
echo "MASTER_PORT:      ${MASTER_PORT}"
echo "NPROC_PER_NODE:   ${NPROC_PER_NODE}"
echo "=========================================="

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} \
         --config "${CONFIG}"

echo "StageB 训练完成！模型保存在 config 中 output_dir（默认 outputs/stageB_cot_lora）"
