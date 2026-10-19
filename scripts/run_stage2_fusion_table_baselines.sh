#!/bin/bash
# 论文表格对照（匹配两阶段，不从 HAGE Stage1 起步）：
#   Average (mean) + Concat + Linear (concat)
#
# 用法：
#   bash scripts/run_stage2_fusion_table_baselines.sh
#
# 自定义 GPU：
#   GPU_MEAN=0 GPU_CONCAT=1 bash scripts/run_stage2_fusion_table_baselines.sh
#
# Stage1 均已完成、只跑 Stage2+评估：
#   START_FROM=stage2 bash scripts/run_stage2_fusion_table_baselines.sh
#
# 仅评估：
#   START_FROM=eval bash scripts/run_stage2_fusion_table_baselines.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs/stage2_fusion_ablation

GPU_MEAN=${GPU_MEAN:-0}
GPU_CONCAT=${GPU_CONCAT:-1}
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
echo "Matched table baselines: Average(mean) + Concat+Linear(concat)"
echo "  mean   -> GPU ${GPU_MEAN}   (Stage1+Stage2 both mean)"
echo "  concat -> GPU ${GPU_CONCAT} (Stage1+Stage2 both concat)"
echo "  START_FROM=${START_FROM}  RESUME_TRAINING=${RESUME_TRAINING}"
echo "======================================================"

launch mean   "${GPU_MEAN}"
launch concat "${GPU_CONCAT}"

wait
echo "Matched table baseline jobs finished."
echo "  Average Stage1/2:       outputs/stage1_fusion_mean , outputs/stage2_fusion_mean"
echo "  Concat  Stage1/2:       outputs/stage1_fusion_concat , outputs/stage2_fusion_concat"
echo "  Eval:                   eval_result/stage2_fusion_mean , eval_result/stage2_fusion_concat"
