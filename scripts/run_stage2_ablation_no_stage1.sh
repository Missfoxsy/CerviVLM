#!/bin/bash
# 消融实验：跳过 Stage1，直接从 VLM-FO1 预训练基座训练 Stage2，并评估
# 注意：本脚本仍启用层级注意力 + 门控（task_adaptive），不是 baseline。
# 若需无层级注意力/门控的 baseline，请用 scripts/run_stage2_baseline_no_stage1.sh
#
# 与 scripts/run_two_stage_and_eval.sh 的区别：
#   - 不跑 Stage1 训练/评估
#   - model.base_model = resources/VLM-FO1_Qwen2.5-VL-3B-v01，load_stageA_lora = false
#   - 仍训练 hierarchical_fusion（与完整两阶段 Stage2 结构一致，仅起点不同）
#   - 输出目录独立：outputs/stage2_no_stage1_ablation
#
# 用法：
#   bash scripts/run_stage2_ablation_no_stage1.sh
#
# 多卡 Stage2：
#   CUDA_VISIBLE_DEVICES=0 NPROC_STAGE2=2 bash scripts/run_stage2_ablation_no_stage1.sh
#
# 从头训练（不续训）：
#   RESUME_TRAINING=0 bash scripts/run_stage2_ablation_no_stage1.sh
#
# 仅评估（训练已完成）：
#   SKIP_TRAIN=1 bash scripts/run_stage2_ablation_no_stage1.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

RESUME_TRAINING=${RESUME_TRAINING:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-$(shuf -i 25000-29999 -n 1)}
NPROC_STAGE2=${NPROC_STAGE2:-1}

STAGE2_CONFIG=${STAGE2_CONFIG:-"configs/train_config_stageB_lora_no_stage1.yaml"}

# 与 yaml 中 model.base_model 一致（评估时作为 base，不加载 Stage1 LoRA）
VLM_FO1_BASE=${VLM_FO1_BASE:-"resources/VLM-FO1_Qwen2.5-VL-3B-v01"}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-"outputs/stage2_no_stage1_ablation"}

EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${VLM_FO1_BASE}"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${STAGE2_OUTPUT_DIR}"}
EVAL_JSON=${EVAL_JSON:-"/path/to/dataset/stage2_cot_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/path/to/dataset/origin_val.json"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"eval_result/stage2_no_stage1_ablation"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}

SAVE_HIER_ATTN_DIR=${SAVE_HIER_ATTN_DIR:-"attentionMap/stage2_no_stage1_ablation"}
SAVE_HIER_ATTN_PER_TASK_MAX=${SAVE_HIER_ATTN_PER_TASK_MAX:-10}
SAVE_HIER_ATTN_SEED=${SAVE_HIER_ATTN_SEED:-42}

echo "======================================================"
echo "Ablation: Stage2 only (no Stage1)"
echo "======================================================"
echo "[Stage2 Train] config=${STAGE2_CONFIG}"
echo "[Stage2 Train] base=${VLM_FO1_BASE}  load_stageA_lora=false (in yaml)"
echo "[Stage2 Train] out=${STAGE2_OUTPUT_DIR}  nproc=${NPROC_STAGE2}  port=${MASTER_PORT_STAGE2}"
echo "[Stage2 Eval]  base=${EVAL_BASE_MODEL}"
echo "[Stage2 Eval]  ckpt=${EVAL_CHECKPOINT}"
echo "[Stage2 Eval]  json=${EVAL_JSON}"
echo "[Stage2 Eval]  out=${EVAL_OUTPUT_DIR}"
echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Env] RESUME_TRAINING=${RESUME_TRAINING}  SKIP_TRAIN=${SKIP_TRAIN}"
echo "======================================================"

RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[1/2] Running Stage2 training (from pretrained VLM-FO1, no Stage1)..."
  torchrun --nproc_per_node="${NPROC_STAGE2}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    scripts/train_stage2_cot.py \
    --config "${STAGE2_CONFIG}" \
    "${RESUME_ARGS[@]}"
else
  echo "[1/2] Skip Stage2 training (SKIP_TRAIN=1)"
fi

echo "[2/2] Running Stage2 evaluation..."
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

echo "Ablation pipeline finished."
echo "Stage2 checkpoint: ${STAGE2_OUTPUT_DIR}"
echo "Stage2 evaluation: ${EVAL_OUTPUT_DIR}"
