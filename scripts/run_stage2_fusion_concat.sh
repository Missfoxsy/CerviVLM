#!/bin/bash
# Concat + Linear 对照：匹配 Stage1(concat) → Stage2(concat) → Eval
# 用法：bash scripts/run_stage2_fusion_concat.sh
# 换卡：CUDA_VISIBLE_DEVICES=7 bash scripts/run_stage2_fusion_concat.sh
#
# Stage1 已完成时：
#   START_FROM=stage2 bash scripts/run_stage2_fusion_concat.sh
# 仅评估：
#   START_FROM=eval bash scripts/run_stage2_fusion_concat.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export FUSION_MODE=concat
export START_FROM=${START_FROM:-stage1}
exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
