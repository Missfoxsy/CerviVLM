#!/bin/bash
# Task1 单样本：与 infer_attMap_task1.sh 相同的 checkpoint / JSON / 层级融合配置；
# 可视化方法对齐 scripts/qwen-vl-visualize.py（Qwen2.5-VL 视觉塔逐层自注意力热图）。
#
# 输出目录默认 attention_infer/task1_qwen_vit_layers/{sample_id}/
#   *_qwen_vit_attention_all_layers_grid.png  — 全部视觉层热图在一张 figure 中对比（默认）
#   *_qwen_vit_attn_meta.json
# 可选：SAVE_INDIVIDUAL_LAYERS=1 另存每层单独 PNG
#
# 列出可用 task1 id：
#   bash scripts/visualize_attMap_task1.sh --list_task1_ids
#
# 优先看全图注意力层（配置默认 7,15,23,31，图上标 †）：
#   LAYERS=7,15,23,31 bash scripts/visualize_attMap_task1.sh
# 全层对比（默认）：
#   LAYERS=all bash scripts/visualize_attMap_task1.sh
# 更稀疏的热图（自注意力对角）：
#   SALIENCY_MODE=self bash scripts/visualize_attMap_task1.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 必须为 eager，否则 FlashAttention 路径无法导出注意力矩阵（与 qwen-vl-visualize 一致）
export VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION:-"eager"}

EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${ROOT}/outputs/stage1_layerGated_taskAdaptive"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${ROOT}/outputs/stage2_layerGated_taskAdaptive"}
EVAL_JSON=${EVAL_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json"}
SAMPLE_ID=${SAMPLE_ID:-"identity_2919"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/attention_infer/task1_qwen_vit_layers"}
LAYERS=${LAYERS:-"all"}
SAVE_INDIVIDUAL_LAYERS=${SAVE_INDIVIDUAL_LAYERS:-0}
GRID_COLS=${GRID_COLS:-4}
SALIENCY_MODE=${SALIENCY_MODE:-received}
SMOOTH_KSIZE=${SMOOTH_KSIZE:-0}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}

HF_ARGS=()
if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  HF_ARGS+=(--use_hierarchical_fusion --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}" --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}")
fi

EXTRA=()
if [[ "${1:-}" == "--list_task1_ids" ]]; then
  EXTRA+=(--list_task1_ids)
  shift || true
fi

echo "[visualize_attMap_task1] VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION}"
echo "[visualize_attMap_task1] sample_id=${SAMPLE_ID} layers=${LAYERS} grid_cols=${GRID_COLS} -> ${OUTPUT_DIR}"

PY_EXTRA=()
if [[ "${SAVE_INDIVIDUAL_LAYERS}" == "1" ]]; then
  PY_EXTRA+=(--save_individual_layers)
fi

python "${ROOT}/scripts/visualize_task1_qwen_vit_attention.py" \
  --test_json "${EVAL_JSON}" \
  --sample_id "${SAMPLE_ID}" \
  --base_model "${EVAL_BASE_MODEL}" \
  --checkpoint "${EVAL_CHECKPOINT}" \
  --device "${EVAL_DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --layers "${LAYERS}" \
  --grid_cols "${GRID_COLS}" \
  --saliency_mode "${SALIENCY_MODE}" \
  --smooth_ksize "${SMOOTH_KSIZE}" \
  "${PY_EXTRA[@]}" \
  "${EXTRA[@]}" \
  "${HF_ARGS[@]}" \
  "$@"

echo "完成。全层对比总图: ${OUTPUT_DIR}/<sample_id>/*_qwen_vit_attention_all_layers_grid.png"
