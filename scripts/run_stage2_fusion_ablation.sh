#!/bin/bash
# 兼容入口：转发到「匹配 Stage1+Stage2」融合消融脚本。
#
# 旧逻辑仅从 HAGE Stage1（task_adaptive）初始化再换 fusion_mode，对照不公平。
# 现默认走 scripts/run_matched_fusion_ablation.sh：
#   Stage1(fusion_mode=X) → Stage2(fusion_mode=X) → Eval
#
# 用法同前：
#   FUSION_MODE=concat CUDA_VISIBLE_DEVICES=3 bash scripts/run_stage2_fusion_ablation.sh
#   START_FROM=stage2 FUSION_MODE=mean bash scripts/run_stage2_fusion_ablation.sh
#
# 若仍需旧行为（不推荐，仅复现污染实验）：
#   ALLOW_HAGE_STAGE1_INIT=1 FUSION_MODE=concat bash scripts/run_stage2_fusion_ablation_legacy_init.sh

set -euo pipefail

if [[ "${ALLOW_HAGE_STAGE1_INIT:-0}" == "1" ]]; then
  echo "Error: contaminated HAGE-Stage1 init path removed from this entry."
  echo "       Use scripts/run_matched_fusion_ablation.sh instead."
  exit 1
fi

# 旧脚本只有 SKIP_TRAIN；映射到新脚本 START_FROM
if [[ -z "${START_FROM:-}" ]]; then
  if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    export START_FROM=eval
  else
    export START_FROM=stage1
  fi
fi

exec bash "$(dirname "$0")/run_matched_fusion_ablation.sh"
