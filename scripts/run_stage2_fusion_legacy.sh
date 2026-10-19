#!/bin/bash
# legacy 对照：匹配 Stage1(legacy) → Stage2(legacy) → Eval
# 用法：bash scripts/run_stage2_fusion_legacy.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export FUSION_MODE=legacy
export START_FROM=${START_FROM:-stage1}
exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
