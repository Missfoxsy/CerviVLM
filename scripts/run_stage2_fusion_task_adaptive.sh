#!/bin/bash
# B2d: task_adaptive 主模型 HAGE（跨层 MHA + 任务自适应门控）— 默认 GPU 3
# 用法：bash scripts/run_stage2_fusion_task_adaptive.sh
# 换卡：CUDA_VISIBLE_DEVICES=7 bash scripts/run_stage2_fusion_task_adaptive.sh
#
# 若 outputs/stage2_layerGated_taskAdaptive 已训练完，可仅评估：
#   SKIP_TRAIN=1 bash scripts/run_stage2_fusion_task_adaptive.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export FUSION_MODE=task_adaptive
exec bash "$(dirname "$0")/run_stage2_fusion_ablation.sh"
