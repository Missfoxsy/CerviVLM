#!/bin/bash
set -euo pipefail

# GPU 选择
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 分布式训练配置
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}  # 使用全部4张卡

# 配置文件与入口
CONFIG_FILE="configs/train_config_stageB.yaml"
ENTRY_FILE="scripts/train_cervical_stage2.py"

# 阶段A与阶段B输出目录
STAGE_A_DIR="./outputs/cervical_PAfull_A"
OUTPUT_DIR="./outputs/cervical_PAfull_B"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 检查阶段A checkpoint
if [ ! -d "$STAGE_A_DIR" ]; then
    echo "Error: Stage A checkpoint directory not found: $STAGE_A_DIR"
    echo "请先完成阶段A训练（train_cervical_stage2.sh）后再运行本脚本。"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印配置信息
echo "=========================================="
echo "VLM-FO1 宫颈细胞定位第二阶段 LoRA 微调（阶段B）"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Stage A checkpoint: $STAGE_A_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 启动阶段B训练（从阶段A完整模型 checkpoint 继续，内部会先加载再套 LoRA）
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} \
         --config ${CONFIG_FILE} \
         --resume_from_checkpoint "${STAGE_A_DIR}"

echo "阶段B训练完成！模型保存在: $OUTPUT_DIR"

