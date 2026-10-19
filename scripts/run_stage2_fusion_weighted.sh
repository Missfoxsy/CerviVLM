#!/bin/bash
# weighted 对照：匹配 Stage1(weighted) → Stage2(weighted) → Eval
# 用法：bash scripts/run_stage2_fusion_weighted.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export FUSION_MODE=weighted
export START_FROM=${START_FROM:-stage1}
exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
