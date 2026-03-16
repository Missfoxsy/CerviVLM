#!/bin/bash

# Stage-1 对齐模型评估脚本

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

ENTRY_FILE="scripts/eval_stage1_align.py"
BASE_MODEL="${1:-outputs/cervical_stage1_align}"
TEST_JSON="${2:-/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json}"
OUTPUT_DIR="${3:-eval_result/stage1_align}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "VLM-FO1 宫颈 Stage-1 对齐模型评估"
echo "=========================================="
echo "Entry file:   ${ENTRY_FILE}"
echo "Base model:   ${BASE_MODEL}"
echo "Test JSON:    ${TEST_JSON}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

python ${ENTRY_FILE} \
  --base_model "${BASE_MODEL}" \
  --test_json "${TEST_JSON}" \
  --output_dir "${OUTPUT_DIR}"

echo "Stage-1 评估完成，结果保存在: ${OUTPUT_DIR}"

