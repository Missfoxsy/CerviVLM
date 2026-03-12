#!/bin/bash
set -euo pipefail

# GPU 选择
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,7

# 分布式训练配置
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}

# 基础配置与入口
BASE_CONFIG_FILE="configs/train_config_stageB.yaml"
ENTRY_FILE="scripts/train_cervical_stage2.py"

# 阶段A与本实验输出目录
STAGE_A_DIR="./outputs/cervical_PAfull_A"
OUTPUT_DIR="./outputs/cervical_PAfull_B_noaux"
GENERATED_CONFIG="${OUTPUT_DIR}/train_config_stageB_noaux.generated.yaml"

# 检查配置文件
if [ ! -f "$BASE_CONFIG_FILE" ]; then
    echo "Error: Config file not found: $BASE_CONFIG_FILE"
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

# 生成“无辅助损失”配置：
# - 保持其余配置不变
# - 将 training.aux_loss 的 region/cls 权重全部置零
# - 覆盖输出目录，避免与默认 stageB 实验互相覆盖
python - <<'PY'
import os
import yaml

base_cfg = "configs/train_config_stageB.yaml"
out_dir = "./outputs/cervical_PAfull_B_noaux"
gen_cfg = os.path.join(out_dir, "train_config_stageB_noaux.generated.yaml")

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

training = cfg.setdefault("training", {})
aux = training.setdefault("aux_loss", {})
aux["region_loss_weight"] = 0.0
aux["cls_loss_weight"] = 0.0
aux["region_loss_weight_by_task"] = {}
aux["cls_loss_weight_by_task"] = {}
training["output_dir"] = out_dir

with open(gen_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"Generated no-aux config: {gen_cfg}")
PY

# 打印配置信息
echo "=========================================="
echo "VLM-FO1 宫颈细胞定位第二阶段 LoRA 微调（无辅助损失）"
echo "=========================================="
echo "Base config: $BASE_CONFIG_FILE"
echo "Generated config: $GENERATED_CONFIG"
echo "Stage A checkpoint: $STAGE_A_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 启动阶段B训练（关闭 L_region / L_cls）
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} \
         --config "${GENERATED_CONFIG}" \
         --resume_from_checkpoint "${STAGE_A_DIR}"

echo "阶段B（无辅助损失）训练完成！模型保存在: $OUTPUT_DIR"
