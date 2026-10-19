#!/bin/bash
# Average 对照：匹配 Stage1(mean) → Stage2(mean) → Eval
# 用法：bash scripts/run_stage2_fusion_mean.sh
# 换卡：CUDA_VISIBLE_DEVICES=4 bash scripts/run_stage2_fusion_mean.sh
#
# Stage1 已完成时：
#   START_FROM=stage2 bash scripts/run_stage2_fusion_mean.sh
# 仅评估：
#   START_FROM=eval bash scripts/run_stage2_fusion_mean.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FUSION_MODE=mean
export START_FROM=${START_FROM:-stage1}
exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
