#!/bin/bash
set -euo pipefail

# =========================
# VLM-FO1 Early Fusion 两阶段串行训练脚本
# Stage A: feature_fusion_warmup
# Stage B: full (LoRA + fusion modules)
# =========================

# GPU 选择（可通过环境变量覆盖）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# 分布式训练配置（可覆盖）
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# 配置与入口（可覆盖）
ENTRY_FILE=${ENTRY_FILE:-"scripts/train_cervical_stage2.py"}
CONFIG_A=${CONFIG_A:-"configs/train_config_stageA_fusion.yaml"}
CONFIG_B=${CONFIG_B:-"configs/train_config_stageB_fusion.yaml"}

if [ ! -f "$ENTRY_FILE" ]; then
  echo "Error: entry file not found: $ENTRY_FILE"
  exit 1
fi
if [ ! -f "$CONFIG_A" ]; then
  echo "Error: Stage A config not found: $CONFIG_A"
  exit 1
fi
if [ ! -f "$CONFIG_B" ]; then
  echo "Error: Stage B config not found: $CONFIG_B"
  exit 1
fi

# 从 yaml 中读取 output_dir（避免硬编码）
STAGE_A_DIR=$(CONFIG_PATH="$CONFIG_A" python - <<'PY'
import yaml
import os
with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["training"]["output_dir"])
PY
)

STAGE_B_DIR=$(CONFIG_PATH="$CONFIG_B" python - <<'PY'
import yaml
import os
with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["training"]["output_dir"])
PY
)

mkdir -p "$STAGE_A_DIR" "$STAGE_B_DIR"

echo "=========================================="
echo "VLM-FO1 Early Fusion 两阶段训练"
echo "=========================================="
echo "Entry file:         $ENTRY_FILE"
echo "Stage A config:     $CONFIG_A"
echo "Stage A output:     $STAGE_A_DIR"
echo "Stage B config:     $CONFIG_B"
echo "Stage B output:     $STAGE_B_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE:     $NPROC_PER_NODE"
echo "=========================================="

# ---- Stage A ----
MASTER_PORT_A=${MASTER_PORT_A:-$(shuf -i 20001-24999 -n 1)}
echo ""
echo "[Stage A] feature_fusion_warmup start (master_port=${MASTER_PORT_A}) ..."
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT_A}" \
         "${ENTRY_FILE}" \
         --config "${CONFIG_A}"
echo "[Stage A] done."

# ---- Stage B ----
if [ ! -d "$STAGE_A_DIR" ]; then
  echo "Error: Stage A output directory not found: $STAGE_A_DIR"
  exit 1
fi

MASTER_PORT_B=${MASTER_PORT_B:-$(shuf -i 25001-29999 -n 1)}
echo ""
echo "[Stage B] full LoRA + fusion start (master_port=${MASTER_PORT_B}) ..."
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT_B}" \
         "${ENTRY_FILE}" \
         --config "${CONFIG_B}" \
         --resume_from_checkpoint "${STAGE_A_DIR}"
echo "[Stage B] done."

echo ""
echo "两阶段训练完成！"
echo "Stage A output: ${STAGE_A_DIR}"
echo "Stage B output: ${STAGE_B_DIR}"

