#!/bin/bash
# PS3C few-shot 实验：5/10/20-shot 依次微调 + 测试集分类评估
#
# 用法:
#   bash scripts/run_ps3c_fewshot.sh
#   CUDA_VISIBLE_DEVICES=0 SKIP_TRAIN=1 bash scripts/run_ps3c_fewshot.sh   # 仅评估
#   CUDA_VISIBLE_DEVICES=0 SHOTS="5" bash scripts/run_ps3c_fewshot.sh      # 只跑 5-shot

#   RESUME_TRAINING=0 bash scripts/run_ps3c_fewshot.sh           # 强制从头训练
#   RESUME_TRAINING=1 bash scripts/run_ps3c_fewshot.sh           # 自动从 checkpoint 续训（默认）

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4"}
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

# 起始 checkpoint（已训练好的 Stage2）
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"outputs/stage2_layerGated_taskAdaptive"}
PS3C_ROOT=${PS3C_ROOT:-"/home/nas3/biod/shiyu/data/data_vl_20260115/PS3C"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/ps3c_fewshot"}
EVAL_ROOT=${EVAL_ROOT:-"eval_result/ps3c_fewshot"}
CACHE_DIR=${CACHE_DIR:-"outputs/ps3c_fewshot_cache"}

TRAIN_CONFIG=${TRAIN_CONFIG:-"configs/train_config_ps3c_fewshot.yaml"}
SKIP_TRAIN=${SKIP_TRAIN:-0}
RESUME_TRAINING=${RESUME_TRAINING:-1}
SHOTS=${SHOTS:-"5 10 20"}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}
EVAL_IMAGE_SIZE=${EVAL_IMAGE_SIZE:-896}
EVAL_START_IDX=${EVAL_START_IDX:-0}

# few-shot 超参（按 shot 数区分；可通过环境变量覆盖）
declare -A SHOT_EPOCHS=( [5]=15 [10]=10 [20]=5 )
declare -A SHOT_LR=( [5]=2e-4 [10]=1e-4 [20]=8e-5 )
declare -A SHOT_ACCUM=( [5]=2 [10]=2 [20]=4 )

echo "======================================================"
echo "PS3C Few-shot Pipeline"
echo "======================================================"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "PS3C root:       ${PS3C_ROOT}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "Eval root:       ${EVAL_ROOT}"
echo "Shots:           ${SHOTS}"
echo "SKIP_TRAIN:      ${SKIP_TRAIN}"
echo "RESUME_TRAIN:    ${RESUME_TRAINING}"
echo "CUDA:            ${CUDA_VISIBLE_DEVICES}"
echo "======================================================"

for SHOT in ${SHOTS}; do
  TRAIN_JSON="${PS3C_ROOT}/ps3c_${SHOT}shot_seed1.json"
  TEST_JSON="${PS3C_ROOT}/ps3c_${SHOT}shot_seed1_test.json"
  OUT_DIR="${OUTPUT_ROOT}/ps3c_${SHOT}shot_seed1"
  EVAL_DIR="${EVAL_ROOT}/ps3c_${SHOT}shot_seed1"

  EPOCHS=${SHOT_EPOCHS[$SHOT]:-20}
  LR=${SHOT_LR[$SHOT]:-1e-4}
  ACCUM=${SHOT_ACCUM[$SHOT]:-2}

  if [[ ! -f "${TRAIN_JSON}" ]]; then
    echo "[WARN] 训练集不存在，跳过 ${SHOT}-shot: ${TRAIN_JSON}"
    continue
  fi
  if [[ ! -f "${TEST_JSON}" ]]; then
    echo "[WARN] 测试集不存在，跳过 ${SHOT}-shot: ${TEST_JSON}"
    continue
  fi

  echo ""
  echo "---------- PS3C ${SHOT}-shot ----------"
  echo "Train: ${TRAIN_JSON}"
  echo "Test:  ${TEST_JSON}"
  echo "Out:   ${OUT_DIR}"
  echo "Epochs=${EPOCHS} LR=${LR} accum=${ACCUM}"

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    TRAIN_ARGS=(
      python scripts/train_ps3c_fewshot.py
      --config "${TRAIN_CONFIG}"
      --train_json "${TRAIN_JSON}"
      --val_json "${TEST_JSON}"
      --output_dir "${OUT_DIR}"
      --run_name "ps3c_${SHOT}shot_seed1"
      --num_epochs "${EPOCHS}"
      --learning_rate "${LR}"
      --gradient_accumulation_steps "${ACCUM}"
      --no_val
    )
    if [[ "${RESUME_TRAINING}" == "1" ]]; then
      TRAIN_ARGS+=(--resume)
    else
      TRAIN_ARGS+=(--no_resume)
    fi
    "${TRAIN_ARGS[@]}"
  else
    echo "[Skip] 训练 (SKIP_TRAIN=1)"
  fi

  FINETUNE_CKPT="${OUT_DIR}"
  if [[ ! -d "${FINETUNE_CKPT}" ]]; then
    echo "[ERROR] checkpoint 不存在: ${FINETUNE_CKPT}"
    exit 1
  fi

  EVAL_CMD=(
    python scripts/eval_ps3c_classification.py
    --base_model "${INIT_CHECKPOINT}"
    --checkpoint "${FINETUNE_CKPT}"
    --test_json "${TEST_JSON}"
    --output_dir "${EVAL_DIR}"
    --enrich_cache_dir "${CACHE_DIR}"
    --image_size "${EVAL_IMAGE_SIZE}"
    --start_idx "${EVAL_START_IDX}"
  )
  if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
    EVAL_CMD+=(--use_hierarchical_fusion --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}")
  fi

  "${EVAL_CMD[@]}"
  echo "[Done] ${SHOT}-shot -> ${EVAL_DIR}/metrics.json"
done

echo ""
echo "全部 PS3C few-shot 实验完成。汇总目录: ${EVAL_ROOT}"
