#!/bin/bash

# Stage2 CoT 多任务评估（任务 0,1,2,4,5），用法对齐 eval_stage1_align.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 使用指定卡示例: CUDA_VISIBLE_DEVICES=5 bash scripts/eval_stage2_cot.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6"}

ENTRY_FILE="scripts/eval_stage2_cot.py"
# 参数: $1 基座/合并模型路径  $2 Stage2 checkpoint  $3 测试 JSON  $4 输出目录
BASE_MODEL="${1:-outputs/stage1_layerGated_taskAdaptive}"
CHECKPOINT="${2:-outputs/stage2_layerGated_taskAdaptive}"
TEST_JSON="${3:-/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json}"
OUTPUT_DIR="${4:-eval_result/gate_01delete}"
# 可选第 5 个参数：约束语言 zh | en（与训练 CervicalDataset.language 一致）
LANGUAGE="${5:-${LANGUAGE:-zh}}"
# 原始数据集（用于 task0/1 的 GT bbox）；可用环境变量覆盖
ORIGIN_JSON="${ORIGIN_JSON:-/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json}"
# 中间结果保存间隔
SAVE_EVERY_N_SAMPLES="${SAVE_EVERY_N_SAMPLES:-1000}"
# 仅跑前 N 条样本（0 或不设=全量）；与 TASK_IDS 叠加
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
# 要评的 task_id，逗号分隔；默认与脚本内一致
TASK_IDS="${TASK_IDS:-0,1}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "VLM-FO1 Stage2 CoT 多任务评估"
echo "=========================================="
echo "Entry file:   ${ENTRY_FILE}"
echo "Base model:   ${BASE_MODEL}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Test JSON:    ${TEST_JSON}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Language:     ${LANGUAGE}"
echo "Origin JSON:  ${ORIGIN_JSON}"
echo "Save every N: ${SAVE_EVERY_N_SAMPLES}"
echo "TASK_IDS:       ${TASK_IDS}"
echo "EVAL_MAX_SAMPLES: ${EVAL_MAX_SAMPLES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

EXTRA_ARGS=(
  --base_model "${BASE_MODEL}"
  --checkpoint "${CHECKPOINT}"
  --test_json "${TEST_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --language "${LANGUAGE}"
  --task_ids "${TASK_IDS}"
  --origin_json "${ORIGIN_JSON}"
  --save_every_n_samples "${SAVE_EVERY_N_SAMPLES}"
  --use_hierarchical_fusion
  --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE:-task_adaptive}"
)
if [[ "${EVAL_MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${EVAL_MAX_SAMPLES}")
fi

# 可选：环境变量 PATH_FEATURE_ROOT 传入病理特征目录
if [ -n "${PATH_FEATURE_ROOT:-}" ]; then
  EXTRA_ARGS+=(--path_feature_root "${PATH_FEATURE_ROOT}")
fi
# 可选：VISUALIZE=1 保存定位可视化
if [ "${VISUALIZE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--visualize)
fi

python "${ENTRY_FILE}" "${EXTRA_ARGS[@]}"

echo "Stage2 CoT 评估完成，结果保存在: ${OUTPUT_DIR}"
