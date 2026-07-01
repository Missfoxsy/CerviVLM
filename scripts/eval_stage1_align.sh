#!/bin/bash
set -euo pipefail

# Stage1 checkpoint 评估脚本（默认评估 checkpoint-1000）
# 用法：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage1_align.sh
#   bash scripts/eval_stage1_align.sh <CKPT_DIR> <TEST_JSON> <OUTPUT_DIR> <TASK_IDS>

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
# 无 flash-attn 时默认走 PyTorch SDPA，避免 FlashAttention2 报错
export VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION:-"sdpa"}

# 某些平台会注入非法 OMP_NUM_THREADS（例如空/非数字），libgomp 会直接报错
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi

ENTRY_FILE="scripts/eval_stage1_align.py"

# 1) Stage1 checkpoint 目录（需含 adapter_config.json + adapter_model.safetensors）
CKPT_DIR="${1:-outputs/stage1_layerGated_sigmoid/checkpoint-1000}"
# 2) Stage1 验证集
TEST_JSON="${2:-/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json}"
# 3) 输出目录
OUTPUT_DIR="${3:-eval_result/stage1_layerGated_sigmoid_ckpt1000}"
# 4) 评估任务（仅 0,1,2）
TASK_IDS="${4:-0,1,2}"

# 与 run_two_stage_and_eval.sh 保持一致
USE_HIERARCHICAL_FUSION="${USE_HIERARCHICAL_FUSION:-1}"
HIERARCHICAL_NUM_HEADS="${HIERARCHICAL_NUM_HEADS:-8}"
HIERARCHICAL_FUSION_MODE="${HIERARCHICAL_FUSION_MODE:-task_adaptive}"
SAVE_HIER_ATTN_PER_TASK_MAX="${SAVE_HIER_ATTN_PER_TASK_MAX:-10}"
SAVE_HIER_ATTN_SEED="${SAVE_HIER_ATTN_SEED:-42}"
# 为空则不导出注意力图
STAGE1_SAVE_HIER_ATTN_DIR="${STAGE1_SAVE_HIER_ATTN_DIR:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${TEST_JSON}" ]]; then
  echo "[ERROR] TEST_JSON not found: ${TEST_JSON}"
  exit 1
fi

# checkpoint 目录基本检查
if [[ ! -f "${CKPT_DIR}/adapter_config.json" ]]; then
  echo "[ERROR] adapter_config.json not found under: ${CKPT_DIR}"
  exit 1
fi
if [[ ! -f "${CKPT_DIR}/adapter_model.safetensors" ]]; then
  echo "[ERROR] adapter_model.safetensors not found under: ${CKPT_DIR}"
  exit 1
fi

# PEFT 可用性检查：无 peft 时不会加载 LoRA/checkpoint（会退化为仅基座）
if ! python -c "import peft" >/dev/null 2>&1; then
  echo "[ERROR] Python env missing 'peft'. Please activate env or install it first."
  echo "        Current python: $(command -v python)"
  exit 1
fi

echo "=========================================="
echo "VLM-FO1 Stage1 评估（checkpoint）"
echo "=========================================="
echo "Entry file:   ${ENTRY_FILE}"
echo "Checkpoint:   ${CKPT_DIR}"
echo "Test JSON:    ${TEST_JSON}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Task IDs:     ${TASK_IDS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

CMD=(
  python "${ENTRY_FILE}"
  --base_model "${CKPT_DIR}"
  --checkpoint "${CKPT_DIR}"
  --test_json "${TEST_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --task_ids "${TASK_IDS}"
)

if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  CMD+=(
    --use_hierarchical_fusion
    --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}"
    --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}"
  )
fi

if [[ -n "${STAGE1_SAVE_HIER_ATTN_DIR}" ]]; then
  CMD+=(
    --save_hierarchical_attn_dir "${STAGE1_SAVE_HIER_ATTN_DIR}"
    --save_hierarchical_attn_per_task_max "${SAVE_HIER_ATTN_PER_TASK_MAX}"
    --save_hierarchical_attn_seed "${SAVE_HIER_ATTN_SEED}"
  )
fi

"${CMD[@]}"

echo "Stage1 评估完成，结果保存在: ${OUTPUT_DIR}"
