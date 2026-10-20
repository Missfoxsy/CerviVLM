#!/bin/bash

# Stage-1 对齐训练脚本（不使用 path_feature）

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 默认使用 1–4 张 GPU，可根据需要调整
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

ENTRY_FILE="scripts/train_stage1_align.py"
CONFIG="${1:-configs/train_config_stage1_lora.yaml}"

echo "=========================================="
echo "VLM-FO1 宫颈细胞 Stage-1 对齐训练（无 path_feature）"
echo "=========================================="
echo "Entry file:      ${ENTRY_FILE}"
echo "Config:          ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR:     ${MASTER_ADDR}"
echo "MASTER_PORT:     ${MASTER_PORT}"
echo "NPROC_PER_NODE:  ${NPROC_PER_NODE}"
echo "=========================================="

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} \
         --config "${CONFIG}"

echo "Stage-1 训练完成！模型保存在 config 中 output_dir。"

