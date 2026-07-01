#!/bin/bash
# 修复 task_ids 透传后，重跑 Stage2 全任务评估（不重新训练）
#
# 修复内容：vlm_fo1/model/language_model/omchat_qwen2_5_vl.py
#   prepare_inputs_for_generation 显式转发 task_ids -> task_adaptive 门控在 eval 时生效
#
# 用法：
#   bash scripts/rerun_stage2_eval_task_gate_fix.sh
#
# 指定 GPU：
#   CUDA_VISIBLE_DEVICES=6 bash scripts/rerun_stage2_eval_task_gate_fix.sh
#
# 覆盖 checkpoint / 输出目录：
#   EVAL_OUTPUT_DIR=eval_result/stage2_taskAdaptive_rerun \
#   bash scripts/rerun_stage2_eval_task_gate_fix.sh
#
# 仅快速冒烟（前 50 条）：
#   EVAL_MAX_SAMPLES=50 bash scripts/rerun_stage2_eval_task_gate_fix.sh
#
# 后台运行并写日志：
#   nohup bash scripts/rerun_stage2_eval_task_gate_fix.sh > logs/rerun_stage2_eval_gate_fix.log 2>&1 &

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"7"}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi

# 与此前 stage2_taskAdaptive_0512 评估对齐
EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"outputs/stage1_layerGated_taskAdaptive"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"outputs/stage2_layerGated_taskAdaptive"}
EVAL_JSON=${EVAL_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_taskAdaptive_gate_fix_rerun"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}
TASK_IDS=${TASK_IDS:-"0,1,2,3,4,5"}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}

SAVE_EVERY_N_SAMPLES=${SAVE_EVERY_N_SAMPLES:-1000}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}

# 可选：导出层级门控统计与注意力图（gate_stats.json 在 eval 过程中写入 output_dir）
SAVE_HIER_ATTN_DIR=${SAVE_HIER_ATTN_DIR:-""}
SAVE_HIER_ATTN_PER_TASK_MAX=${SAVE_HIER_ATTN_PER_TASK_MAX:-10}
SAVE_HIER_ATTN_SEED=${SAVE_HIER_ATTN_SEED:-42}

mkdir -p "${EVAL_OUTPUT_DIR}"
mkdir -p logs

echo "======================================================"
echo "Stage2 eval rerun (task_ids fix applied)"
echo "======================================================"
echo "[Eval] base_model   = ${EVAL_BASE_MODEL}"
echo "[Eval] checkpoint   = ${EVAL_CHECKPOINT}"
echo "[Eval] test_json    = ${EVAL_JSON}"
echo "[Eval] origin_json  = ${ORIGIN_JSON}"
echo "[Eval] output_dir   = ${EVAL_OUTPUT_DIR}"
echo "[Eval] task_ids     = ${TASK_IDS}"
echo "[Eval] language     = ${EVAL_LANGUAGE}"
echo "[Eval] hierarchical = ${USE_HIERARCHICAL_FUSION} (${HIERARCHICAL_FUSION_MODE})"
echo "[Eval] max_samples  = ${EVAL_MAX_SAMPLES} (0=全量)"
echo "[Env]  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "======================================================"

EVAL_CMD=(
  python scripts/eval_stage2_cot.py
  --base_model "${EVAL_BASE_MODEL}"
  --checkpoint "${EVAL_CHECKPOINT}"
  --origin_json "${ORIGIN_JSON}"
  --test_json "${EVAL_JSON}"
  --output_dir "${EVAL_OUTPUT_DIR}"
  --device "${EVAL_DEVICE}"
  --language "${EVAL_LANGUAGE}"
  --task_ids "${TASK_IDS}"
  --save_every_n_samples "${SAVE_EVERY_N_SAMPLES}"
)

if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  EVAL_CMD+=(
    --use_hierarchical_fusion
    --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}"
    --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}"
  )
fi

if [[ "${EVAL_MAX_SAMPLES}" != "0" ]]; then
  EVAL_CMD+=("--max_samples" "${EVAL_MAX_SAMPLES}")
fi

if [[ -n "${SAVE_HIER_ATTN_DIR}" ]]; then
  mkdir -p "${SAVE_HIER_ATTN_DIR}"
  EVAL_CMD+=(
    --save_hierarchical_attn_dir "${SAVE_HIER_ATTN_DIR}"
    --save_hierarchical_attn_per_task_max "${SAVE_HIER_ATTN_PER_TASK_MAX}"
    --save_hierarchical_attn_seed "${SAVE_HIER_ATTN_SEED}"
  )
fi

if [[ "${VISUALIZE:-0}" == "1" ]]; then
  EVAL_CMD+=(--visualize)
fi

if [[ -n "${PATH_FEATURE_ROOT:-}" ]]; then
  EVAL_CMD+=(--path_feature_root "${PATH_FEATURE_ROOT}")
fi

"${EVAL_CMD[@]}"

echo "======================================================"
echo "Stage2 eval rerun finished."
echo "  metrics.json    -> ${EVAL_OUTPUT_DIR}/metrics.json"
echo "  gate_stats.json -> ${EVAL_OUTPUT_DIR}/gate_stats.json"
echo "  results.json    -> ${EVAL_OUTPUT_DIR}/results.json"
echo "======================================================"
