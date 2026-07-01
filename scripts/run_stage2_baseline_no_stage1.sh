#!/bin/bash
# Baseline：跳过 Stage1，直接从官方 VLM-FO1 预训练基座微调 Stage2 LoRA，并评估全部 task
#
# 与 scripts/run_stage2_ablation_no_stage1.sh 的区别：
#   - 不启用层级注意力 / 门控（use_hierarchical_fusion=false，仅训 LoRA）
#   - 评估时不传 --use_hierarchical_fusion，不保存注意力图
#
# 用法：
#   bash scripts/run_stage2_baseline_no_stage1.sh
#
# 多卡 Stage2：
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_STAGE2=2 bash scripts/run_stage2_baseline_no_stage1.sh
#
# 从头训练（不续训）：
#   RESUME_TRAINING=0 bash scripts/run_stage2_baseline_no_stage1.sh
#
# 仅评估（训练已完成）：
#   SKIP_TRAIN=1 bash scripts/run_stage2_baseline_no_stage1.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,3"}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

RESUME_TRAINING=${RESUME_TRAINING:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-$(shuf -i 25000-29999 -n 1)}
NPROC_STAGE2=${NPROC_STAGE2:-1}

STAGE2_CONFIG=${STAGE2_CONFIG:-"configs/train_config_stageB_baseline_no_stage1.yaml"}

VLM_FO1_BASE=${VLM_FO1_BASE:-"resources/VLM-FO1_Qwen2.5-VL-3B-v01"}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-"outputs/stage2_baseline_no_stage1"}

EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${VLM_FO1_BASE}"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${STAGE2_OUTPUT_DIR}"}
EVAL_JSON=${EVAL_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_baseline_no_stage1"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}

echo "======================================================"
echo "Baseline: Stage2 only (no Stage1, no hierarchical fusion)"
echo "======================================================"
echo "[Stage2 Train] config=${STAGE2_CONFIG}"
echo "[Stage2 Train] base=${VLM_FO1_BASE}  load_stageA_lora=false  use_hierarchical_fusion=false"
echo "[Stage2 Train] out=${STAGE2_OUTPUT_DIR}  nproc=${NPROC_STAGE2}  port=${MASTER_PORT_STAGE2}"
echo "[Stage2 Eval]  base=${EVAL_BASE_MODEL}"
echo "[Stage2 Eval]  ckpt=${EVAL_CHECKPOINT}"
echo "[Stage2 Eval]  json=${EVAL_JSON}"
echo "[Stage2 Eval]  out=${EVAL_OUTPUT_DIR}"
echo "[Stage2 Eval]  task_ids=0,1,2,3,4,5  hierarchical_fusion=off"
echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Env] RESUME_TRAINING=${RESUME_TRAINING}  SKIP_TRAIN=${SKIP_TRAIN}"
echo "======================================================"

RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[1/2] Running Stage2 baseline training (VLM-FO1 + LoRA only)..."
  torchrun --nproc_per_node="${NPROC_STAGE2}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    scripts/train_stage2_cot.py \
    --config "${STAGE2_CONFIG}" \
    "${RESUME_ARGS[@]}"
else
  echo "[1/2] Skip Stage2 training (SKIP_TRAIN=1)"
fi

echo "[2/2] Running Stage2 evaluation (all tasks, no hierarchical fusion)..."
python scripts/eval_stage2_cot.py \
  --base_model "${EVAL_BASE_MODEL}" \
  --checkpoint "${EVAL_CHECKPOINT}" \
  --origin_json "${ORIGIN_JSON}" \
  --test_json "${EVAL_JSON}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --device "${EVAL_DEVICE}" \
  --language "${EVAL_LANGUAGE}" \
  --task_ids "0,1,2,3,4,5"

echo "Baseline pipeline finished."
echo "Stage2 checkpoint: ${STAGE2_OUTPUT_DIR}"
echo "Stage2 evaluation: ${EVAL_OUTPUT_DIR}"
