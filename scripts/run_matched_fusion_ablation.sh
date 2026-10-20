#!/bin/bash
# 匹配融合消融：同一 fusion_mode 跑 Stage1 → Stage2 → Eval
#
# 关键：避免对照实验从 HAGE（task_adaptive）Stage1 初始化，
#       否则 Average / Concat+Linear 会继承门控相关权重，对比不公平。
#
# 论文 Table II 中的融合方式：
#   concat         — (b) 逐层 Linear 再拼接
#   legacy         — (c) 跨层注意力，无门控
#   softmax_gate   — (d) 跨层注意力 + Softmax 门控
#   task_adaptive  — (e) 跨层注意力 + Sigmoid 门控（主模型）
#
# 用法：
#   FUSION_MODE=concat CUDA_VISIBLE_DEVICES=0 bash scripts/run_matched_fusion_ablation.sh
#   FUSION_MODE=softmax_gate CUDA_VISIBLE_DEVICES=0 bash scripts/run_matched_fusion_ablation.sh
#   START_FROM=stage2 FUSION_MODE=mean bash scripts/run_matched_fusion_ablation.sh
#   START_FROM=eval   FUSION_MODE=concat bash scripts/run_matched_fusion_ablation.sh
#
# 从头训练（不续训）：
#   RESUME_TRAINING=0 FUSION_MODE=softmax_gate bash scripts/run_matched_fusion_ablation.sh
#
# 表格对照并行：
#   bash scripts/run_stage2_fusion_table_baselines.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

FUSION_MODE=${FUSION_MODE:-"concat"}
# stage1 | stage2 | eval
START_FROM=${START_FROM:-"stage1"}
RESUME_TRAINING=${RESUME_TRAINING:-1}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_STAGE1=${MASTER_PORT_STAGE1:-$(shuf -i 20001-24999 -n 1)}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-$(shuf -i 25000-29999 -n 1)}
NPROC_STAGE1=${NPROC_STAGE1:-1}
NPROC_STAGE2=${NPROC_STAGE2:-1}

EVAL_JSON=${EVAL_JSON:-"data/stage2/stage2_cot_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/path/to/dataset/origin_val.json"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
SAVE_HIER_ATTN_PER_TASK_MAX=${SAVE_HIER_ATTN_PER_TASK_MAX:-10}
SAVE_HIER_ATTN_SEED=${SAVE_HIER_ATTN_SEED:-42}

case "${START_FROM}" in
  stage1|stage2|eval) ;;
  *)
    echo "Error: START_FROM must be one of: stage1, stage2, eval (got: ${START_FROM})"
    exit 1
    ;;
esac

case "${FUSION_MODE}" in
  concat|legacy|softmax_gate)
    STAGE1_CONFIG=${STAGE1_CONFIG:-"configs/train_config_stage1_lora_${FUSION_MODE}.yaml"}
    STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"outputs/stage1_fusion_${FUSION_MODE}"}
    STAGE2_CONFIG=${STAGE2_CONFIG:-"configs/train_config_stageB_lora_${FUSION_MODE}.yaml"}
    STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-"outputs/stage2_fusion_${FUSION_MODE}"}
    EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_fusion_${FUSION_MODE}"}
    SAVE_HIER_ATTN_DIR=${SAVE_HIER_ATTN_DIR:-"attentionMap/stage2_fusion_${FUSION_MODE}"}
    ;;
  task_adaptive)
    # 主模型：沿用既有 Stage1 目录命名
    STAGE1_CONFIG=${STAGE1_CONFIG:-"configs/train_config_stage1_lora.yaml"}
    STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"outputs/stage1_layerGated_taskAdaptive"}
    STAGE2_CONFIG=${STAGE2_CONFIG:-"configs/train_config_stageB_lora.yaml"}
    STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-"outputs/stage2_layerGated_taskAdaptive"}
    EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_layerGated_taskAdaptive"}
    SAVE_HIER_ATTN_DIR=${SAVE_HIER_ATTN_DIR:-"attentionMap/stage2_layerGated_taskAdaptive"}
    ;;
  *)
    echo "Error: FUSION_MODE must be one of: concat, legacy, softmax_gate, task_adaptive (got: ${FUSION_MODE})"
    exit 1
    ;;
esac

HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"${FUSION_MODE}"}
EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${STAGE1_OUTPUT_DIR}"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${STAGE2_OUTPUT_DIR}"}

RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

echo "======================================================"
echo "Matched Fusion Ablation (mode=${FUSION_MODE}, START_FROM=${START_FROM})"
echo "======================================================"
echo "[Stage1] config=${STAGE1_CONFIG}"
echo "[Stage1] out=${STAGE1_OUTPUT_DIR}  nproc=${NPROC_STAGE1}  port=${MASTER_PORT_STAGE1}"
echo "[Stage2] config=${STAGE2_CONFIG}"
echo "[Stage2] out=${STAGE2_OUTPUT_DIR}  nproc=${NPROC_STAGE2}  port=${MASTER_PORT_STAGE2}"
echo "[Eval]   base=${EVAL_BASE_MODEL}  ckpt=${EVAL_CHECKPOINT}"
echo "[Eval]   fusion_mode=${HIERARCHICAL_FUSION_MODE}  out=${EVAL_OUTPUT_DIR}"
echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  RESUME_TRAINING=${RESUME_TRAINING}"
echo "======================================================"

if [[ "${START_FROM}" == "stage1" ]]; then
  echo "[1/3] Stage1 training (fusion_mode=${FUSION_MODE})..."
  torchrun --nproc_per_node="${NPROC_STAGE1}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE1}" \
    scripts/train_stage1_align.py \
    --config "${STAGE1_CONFIG}" \
    "${RESUME_ARGS[@]}"
else
  echo "[1/3] Skip Stage1 (START_FROM=${START_FROM})"
fi

if [[ "${START_FROM}" == "stage1" || "${START_FROM}" == "stage2" ]]; then
  if [[ ! -d "${STAGE1_OUTPUT_DIR}" ]]; then
    echo "Error: Stage1 output missing: ${STAGE1_OUTPUT_DIR}"
    echo "       Please run with START_FROM=stage1 first."
    exit 1
  fi
  echo "[2/3] Stage2 training (fusion_mode=${FUSION_MODE}, init from matched Stage1)..."
  torchrun --nproc_per_node="${NPROC_STAGE2}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    scripts/train_stage2_cot.py \
    --config "${STAGE2_CONFIG}" \
    "${RESUME_ARGS[@]}"
else
  echo "[2/3] Skip Stage2 (START_FROM=${START_FROM})"
fi

echo "[3/3] Stage2 evaluation..."
EVAL_CMD=(
  python scripts/eval_stage2_cot.py
  --base_model "${EVAL_BASE_MODEL}"
  --checkpoint "${EVAL_CHECKPOINT}"
  --origin_json "${ORIGIN_JSON}"
  --test_json "${EVAL_JSON}"
  --output_dir "${EVAL_OUTPUT_DIR}"
  --device "${EVAL_DEVICE}"
  --language "${EVAL_LANGUAGE}"
  --task_ids "0,1,2,3,4,5"
)

if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  EVAL_CMD+=(
    --use_hierarchical_fusion
    --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}"
    --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}"
  )
fi

if [[ -n "${SAVE_HIER_ATTN_DIR}" ]]; then
  EVAL_CMD+=(
    --save_hierarchical_attn_dir "${SAVE_HIER_ATTN_DIR}"
    --save_hierarchical_attn_per_task_max "${SAVE_HIER_ATTN_PER_TASK_MAX}"
    --save_hierarchical_attn_seed "${SAVE_HIER_ATTN_SEED}"
  )
fi

"${EVAL_CMD[@]}"

echo "Matched fusion ablation finished (mode=${FUSION_MODE})."
echo "  Stage1: ${STAGE1_OUTPUT_DIR}"
echo "  Stage2: ${STAGE2_OUTPUT_DIR}"
echo "  Eval:   ${EVAL_OUTPUT_DIR}"
