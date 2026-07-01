#!/bin/bash
# Task1：FO1 checkpoint 上跑 mllms_know 式 rel_att（LLM 层相对注意力 + 自适应 bbox）
# 配置与 infer_attMap_task1.sh / visualize_attMap_task1.sh 对齐。
#
# 输出: attention_infer/task1_mllms_know_fo1/{sample_id}/
#   *_rel_att_heatmap.png
#   *_rel_att_bbox.png
#   *_rel_att_meta.json
#
# 列出 task1 id:
#   bash scripts/visualize_mllms_know_fo1.sh --list_task1_ids

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"5"}

export VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION:-"eager"}

EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${ROOT}/outputs/stage1_layerGated_taskAdaptive"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${ROOT}/outputs/stage2_layerGated_taskAdaptive"}
EVAL_JSON=${EVAL_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json"}
SAMPLE_ID=${SAMPLE_ID:-"identity_2919"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/attention_infer/task1_mllms_know_fo1"}
ATT_LAYER=${ATT_LAYER:-22}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
LANGUAGE=${LANGUAGE:-"zh"}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}

HF_ARGS=()
if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  HF_ARGS+=(--use_hierarchical_fusion --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}" --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}")
fi

EXTRA=()
if [[ "${1:-}" == "--list_task1_ids" ]]; then
  EXTRA+=(--list_task1_ids)
  shift || true
fi

echo "[visualize_mllms_know_fo1] VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION}"
echo "[visualize_mllms_know_fo1] sample_id=${SAMPLE_ID} att_layer=${ATT_LAYER} -> ${OUTPUT_DIR}"

python "${ROOT}/scripts/visualize_task1_mllms_know_fo1.py" \
  --test_json "${EVAL_JSON}" \
  --sample_id "${SAMPLE_ID}" \
  --base_model "${EVAL_BASE_MODEL}" \
  --checkpoint "${EVAL_CHECKPOINT}" \
  --device "${EVAL_DEVICE}" \
  --language "${LANGUAGE}" \
  --output_dir "${OUTPUT_DIR}" \
  --att_layer "${ATT_LAYER}" \
  "${HF_ARGS[@]}" \
  "${EXTRA[@]}" \
  "$@"

echo "完成。热图: ${OUTPUT_DIR}/<sample_id>/*_rel_att_heatmap.png"
