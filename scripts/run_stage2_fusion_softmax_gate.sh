#!/bin/bash
# Softmax Gate 对照：匹配 Stage1(softmax_gate) → Stage2(softmax_gate) → Eval
# 与 task_adaptive 同结构（跨层 MHA + level_selector + task bias），仅门控改为 Softmax(z)
#
# 用法：bash scripts/run_stage2_fusion_softmax_gate.sh
# 换卡：CUDA_VISIBLE_DEVICES=5 bash scripts/run_stage2_fusion_softmax_gate.sh
#
# Stage1 已完成：
#   START_FROM=stage2 bash scripts/run_stage2_fusion_softmax_gate.sh
# 仅评估：
#   START_FROM=eval bash scripts/run_stage2_fusion_softmax_gate.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export FUSION_MODE=softmax_gate
export START_FROM=${START_FROM:-stage1}
exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
