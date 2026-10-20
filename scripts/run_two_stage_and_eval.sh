#!/bin/bash
# 一键执行：Stage1 训练 -> Stage1 评估 -> Stage2 训练 -> Stage2 评估
#
# 用法：
#   bash scripts/run_two_stage_and_eval.sh
#
# 从第二阶段开始（Stage1 已跑完时，避免重复跑 Stage1）：
#   START_FROM=stage2 bash scripts/run_two_stage_and_eval.sh
#
# 只跑 Stage2 评估（训练均已完成）：
#   START_FROM=eval bash scripts/run_two_stage_and_eval.sh
#
# 断点续训（默认开启）：若 yaml 里 output_dir 下已有 checkpoint-*，训练会从最新检查点继续。
# 从头训练请设：RESUME_TRAINING=0
#
# 多卡 Stage2：显式指定进程数与可见 GPU，例如 2 卡 DDP：
#   CUDA_VISIBLE_DEVICES=0 NPROC_STAGE2=2 START_FROM=stage2 bash scripts/run_two_stage_and_eval.sh
#
# 注意：NPROC_STAGE2 必须 <= CUDA_VISIBLE_DEVICES 中可见 GPU 数量；默认 NPROC_STAGE2=1
#       避免 torchrun 多进程挤在同一张卡上导致 OOM。

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# OMP 在部分超算环境会被注入非法值（如空串/非数字），会触发 libgomp 报错
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
# 超算共享目录常见 git ownership 检查告警；禁用 wandb git 扫描以避免噪音/失败
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

# stage1 | stage2 | eval — 从哪一步开始执行（之前步骤跳过）
START_FROM=${START_FROM:-"stage2"}

# 1=传递 --resume 给 Stage1/Stage2 训练脚本（依赖 HF Trainer 从 output_dir 恢复）
RESUME_TRAINING=${RESUME_TRAINING:-1}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_STAGE1=${MASTER_PORT_STAGE1:-$(shuf -i 20001-24999 -n 1)}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-$(shuf -i 25000-29999 -n 1)}
NPROC_STAGE1=${NPROC_STAGE1:-1}
# 默认单进程：每张卡只跑一个模型副本；多卡时再设 NPROC_STAGE2=2
NPROC_STAGE2=${NPROC_STAGE2:-1}

STAGE1_CONFIG=${STAGE1_CONFIG:-"configs/train_config_stage1_lora.yaml"}
STAGE2_CONFIG=${STAGE2_CONFIG:-"configs/train_config_stageB_lora.yaml"}

# Stage1 输出目录（需与 STAGE1_CONFIG 内 training.output_dir 一致，供 Stage1 评估与 Stage2 的 EVAL_BASE_MODEL 使用）
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"outputs/stage1_layerGated_taskAdaptive"}
# Stage1 验证集与评估输出（默认与 configs/train_config_stage1_lora.yaml 中 val_json 对齐）
STAGE1_EVAL_TEST_JSON=${STAGE1_EVAL_TEST_JSON:-"/path/to/dataset/stage1_align_val.json"}
STAGE1_EVAL_OUTPUT_DIR=${STAGE1_EVAL_OUTPUT_DIR:-"eval_result/stage1_layerGated_taskAdaptive"}
# 可选：Stage1 评估时导出层级注意力图目录（空则关闭）
STAGE1_SAVE_HIER_ATTN_DIR=${STAGE1_SAVE_HIER_ATTN_DIR:-"attentionMap/stage1_layerGated_taskAdaptive"}

# Stage2 评估时的 base（Stage1 输出 LoRA）；默认与 STAGE1_OUTPUT_DIR 一致
EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${STAGE1_OUTPUT_DIR}"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"outputs/stage2_layerGated_taskAdaptive"}
EVAL_JSON=${EVAL_JSON:-"/path/to/dataset/stage2_cot_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/path/to/dataset/origin_val.json"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_layerGated_taskAdaptive"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}

# 方案一参数
USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}

# 注意力可视化（可选）
SAVE_HIER_ATTN_DIR=${SAVE_HIER_ATTN_DIR:-"attentionMap/stage2_layerGated_taskAdaptive"}
SAVE_HIER_ATTN_PER_TASK_MAX=${SAVE_HIER_ATTN_PER_TASK_MAX:-10}
SAVE_HIER_ATTN_SEED=${SAVE_HIER_ATTN_SEED:-42}

case "${START_FROM}" in
  stage1|stage2|eval) ;;
  *)
    echo "Error: START_FROM must be one of: stage1, stage2, eval (got: ${START_FROM})"
    exit 1
    ;;
esac

echo "======================================================"
echo "Pipeline: Stage1 train -> Stage1 eval -> Stage2 train -> Stage2 eval   (START_FROM=${START_FROM})"
echo "======================================================"
echo "[Stage1] config=${STAGE1_CONFIG}  out=${STAGE1_OUTPUT_DIR}  nproc=${NPROC_STAGE1}  port=${MASTER_PORT_STAGE1}"
echo "[Stage1 Eval] json=${STAGE1_EVAL_TEST_JSON}  out=${STAGE1_EVAL_OUTPUT_DIR}"
echo "[Stage2] config=${STAGE2_CONFIG}  nproc=${NPROC_STAGE2}  port=${MASTER_PORT_STAGE2}"
echo "[Stage2 Eval] base=${EVAL_BASE_MODEL}"
echo "[Stage2 Eval] ckpt=${EVAL_CHECKPOINT}"
echo "[Stage2 Eval] json=${EVAL_JSON}"
echo "[Stage2 Eval] out=${EVAL_OUTPUT_DIR}"
echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Env] RESUME_TRAINING=${RESUME_TRAINING}"
echo "======================================================"

RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

if [[ "${START_FROM}" == "stage1" ]]; then
  echo "[1/4] Running Stage1 training..."
  torchrun --nproc_per_node="${NPROC_STAGE1}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE1}" \
    scripts/train_stage1_align.py \
    --config "${STAGE1_CONFIG}" \
    "${RESUME_ARGS[@]}"

  echo "[2/4] Running Stage1 evaluation (task 0,1,2)..."
  STAGE1_EVAL_CMD=(
    python scripts/eval_stage1_align.py
    --base_model "${STAGE1_OUTPUT_DIR}"
    --checkpoint "${STAGE1_OUTPUT_DIR}"
    --test_json "${STAGE1_EVAL_TEST_JSON}"
    --output_dir "${STAGE1_EVAL_OUTPUT_DIR}"
    --origin_json "${ORIGIN_JSON}"
    --device "${EVAL_DEVICE}"
    --language "${EVAL_LANGUAGE}"
    --task_ids "0,1,2"
  )
  if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
    STAGE1_EVAL_CMD+=(
      --use_hierarchical_fusion
      --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}"
      --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}"
    )
  fi
  if [[ -n "${STAGE1_SAVE_HIER_ATTN_DIR}" ]]; then
    STAGE1_EVAL_CMD+=(
      --save_hierarchical_attn_dir "${STAGE1_SAVE_HIER_ATTN_DIR}"
      --save_hierarchical_attn_per_task_max "${SAVE_HIER_ATTN_PER_TASK_MAX}"
      --save_hierarchical_attn_seed "${SAVE_HIER_ATTN_SEED}"
    )
  fi
  "${STAGE1_EVAL_CMD[@]}"
else
  echo "[1/4] Skip Stage1 training (START_FROM=${START_FROM})"
  echo "[2/4] Skip Stage1 evaluation (START_FROM=${START_FROM})"
fi

if [[ "${START_FROM}" == "stage1" || "${START_FROM}" == "stage2" ]]; then
  echo "[3/4] Running Stage2 training..."
  torchrun --nproc_per_node="${NPROC_STAGE2}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    scripts/train_stage2_cot.py \
    --config "${STAGE2_CONFIG}" \
    "${RESUME_ARGS[@]}"
else
  echo "[3/4] Skip Stage2 (START_FROM=${START_FROM})"
fi

echo "[4/4] Running Stage2 evaluation..."
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

echo "Pipeline finished."
echo "Stage1 evaluation outputs: ${STAGE1_EVAL_OUTPUT_DIR}"
echo "Stage2 evaluation outputs: ${EVAL_OUTPUT_DIR}"
