#!/bin/bash
# 并行启动匹配两阶段融合对照（不含 task_adaptive 主模型）
#
# 用法：
#   bash scripts/run_stage2_fusion_ablation_all.sh
#
# 自定义 GPU：
#   GPU_MEAN=0 GPU_CONCAT=1 GPU_WEIGHTED=2 GPU_LEGACY=3 bash scripts/run_stage2_fusion_ablation_all.sh
#
# 仅 Stage2+评估（各模式 Stage1 已完成）：
#   START_FROM=stage2 bash scripts/run_stage2_fusion_ablation_all.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs/stage2_fusion_ablation

GPU_MEAN=${GPU_MEAN:-0}
GPU_CONCAT=${GPU_CONCAT:-1}
GPU_WEIGHTED=${GPU_WEIGHTED:-2}
GPU_LEGACY=${GPU_LEGACY:-3}

START_FROM=${START_FROM:-stage1}
RESUME_TRAINING=${RESUME_TRAINING:-1}
TS=$(date +%Y%m%d_%H%M%S)

launch() {
  local mode="$1"
  local gpu="$2"
  local log="logs/stage2_fusion_ablation/matched_${mode}_${TS}.log"
  echo "Launch matched ${mode} on GPU ${gpu} -> ${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export FUSION_MODE="${mode}"
    export START_FROM="${START_FROM}"
    export RESUME_TRAINING="${RESUME_TRAINING}"
    bash "${ROOT_DIR}/scripts/run_matched_fusion_ablation.sh"
  ) > "${log}" 2>&1 &
  echo "  pid=$!"
}

echo "======================================================"
echo "Matched fusion ablations (Stage1 mode == Stage2 mode)"
echo "  mean      -> GPU ${GPU_MEAN}"
echo "  concat    -> GPU ${GPU_CONCAT}"
echo "  weighted  -> GPU ${GPU_WEIGHTED}"
echo "  legacy    -> GPU ${GPU_LEGACY}"
echo "  (task_adaptive skipped)"
echo "  START_FROM=${START_FROM}  RESUME_TRAINING=${RESUME_TRAINING}"
echo "======================================================"

launch mean     "${GPU_MEAN}"
launch concat   "${GPU_CONCAT}"
launch weighted "${GPU_WEIGHTED}"
launch legacy   "${GPU_LEGACY}"

wait
echo "All matched fusion ablation jobs finished. See logs/stage2_fusion_ablation/"
