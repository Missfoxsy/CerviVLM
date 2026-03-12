#!/bin/bash

# GPU selection: 只使用空闲的GPU
# 根据 nvidia-smi，选择空闲的GPU
# 检查GPU使用情况：nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Distributed training configuration (可选，单GPU训练可以不用)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}  # 使用全部4张卡

# Model configuration
# Model path: relative to project root or absolute path
CONFIG_FILE="configs/train_config_stageA.yaml"

# Training entry point
ENTRY_FILE="scripts/train_cervical_stage2.py"

# Output configuration
OUTPUT_DIR="./outputs/cervical_PAfull_A"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印配置信息
echo "=========================================="
echo "VLM-FO1 宫颈细胞定位第二阶段 LoRA 微调"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Stage 1 model: outputs/cervical_PAfull_A"
echo "=========================================="

# 启动训练
# 单GPU训练
# python ${ENTRY_FILE} \
#     --config ${CONFIG_FILE}

# 如果需要多GPU训练，取消下面的注释并使用torchrun
# 建议开启 tee + log_dir 以保存每个 rank 的独立日志，便于定位某个 rank 的真实报错（如 OOM/数据异常）
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} \
         --config ${CONFIG_FILE}

echo "第二阶段训练完成！模型保存在: $OUTPUT_DIR"

