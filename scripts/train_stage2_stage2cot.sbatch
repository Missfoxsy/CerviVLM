#!/bin/bash
#SBATCH --job-name=stage2-cot-medplib-s2
#SBATCH --account=fengjing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# 1) env
# conda activate cervivlm
set -u

# 2) repo root (MedPLIB)
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/logs"

# 3) config (override via sbatch --export)
VERSION="${VERSION:-/path/to/"
VISION_TOWER="${VISION_TOWER:-/path/to/"
VISION_PRETRAINED="${VISION_PRETRAINED:-/path/to/"
PRETRAIN_MM_MLP_ADAPTER="${PRETRAIN_MM_MLP_ADAPTER:-}"

TRAIN_JSON="${TRAIN_JSON:-/path/to/"
VAL_JSON="${VAL_JSON:-/path/to/"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/images}"

EXP_NAME="${EXP_NAME:-medplib-stage2cot}"
RUN_DIR="${RUN_DIR:-/path/to/"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_DIR}/eval_stage2_cot}"
ANSWERS_FILE="${ANSWERS_FILE:-${EVAL_OUTPUT_DIR}/predictions.jsonl}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-2}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-400}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
LORA_R="${LORA_R:-16}"
MASTER_PORT="${MASTER_PORT:-65021}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for var_name in VERSION VISION_TOWER VISION_PRETRAINED TRAIN_JSON VAL_JSON IMAGE_FOLDER RUN_DIR EVAL_OUTPUT_DIR ANSWERS_FILE; do
  var_val="${!var_name}"
  if [[ "${var_val}" != /* ]]; then
    eval "${var_name}=\"${ROOT_DIR}/${var_val#./}\""
  fi
done

echo "==> VERSION=${VERSION}"
echo "==> VISION_TOWER=${VISION_TOWER}"
echo "==> VISION_PRETRAINED=${VISION_PRETRAINED}"
echo "==> TRAIN_JSON=${TRAIN_JSON}"
echo "==> VAL_JSON=${VAL_JSON}"
echo "==> IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "==> RUN_DIR=${RUN_DIR}"
echo "==> EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "==> ANSWERS_FILE=${ANSWERS_FILE}"
echo "==> EPOCHS/BATCH/GRAD_ACC=${EPOCHS}/${BATCH_SIZE}/${GRAD_ACC}"
echo "==> MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH}"
echo "==> MASTER_PORT=${MASTER_PORT}"

if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "[ERROR] TRAIN_JSON not found: ${TRAIN_JSON}"
  exit 1
fi
if [[ ! -f "${VAL_JSON}" ]]; then
  echo "[ERROR] VAL_JSON not found: ${VAL_JSON}"
  exit 1
fi
if [[ ! -f "${VISION_PRETRAINED}" ]]; then
  echo "[ERROR] VISION_PRETRAINED not found: ${VISION_PRETRAINED}"
  exit 1
fi
if [[ ! -f "${VISION_TOWER}/preprocessor_config.json" ]]; then
  echo "[ERROR] VISION_TOWER invalid: ${VISION_TOWER}"
  exit 1
fi

mkdir -p "${RUN_DIR}" "${EVAL_OUTPUT_DIR}"

PRETRAIN_ARGS=()
if [[ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]]; then
  PRETRAIN_ARGS+=(--pretrain_mm_mlp_adapter "${PRETRAIN_MM_MLP_ADAPTER}")
fi

# 4) train (Stage II style)
deepspeed --include=localhost:0 --master_port="${MASTER_PORT}" train_ds_medplib.py \
  --version "${VERSION}" \
  --vision_tower "${VISION_TOWER}" \
  "${PRETRAIN_ARGS[@]}" \
  --data_path "${TRAIN_JSON}" \
  --val_data_path "${VAL_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --exp_name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --image_aspect_ratio "pad" \
  --is_multimodal True \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --grad_accumulation_steps "${GRAD_ACC}" \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 0.0 \
  --bce_loss_weight 0.0 \
  --lora_r "${LORA_R}" \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --sft_modules "lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector" \
  --lr "${LR}" \
  --no_eval \
  --save_steps "${SAVE_STEPS}" \
  2>&1 | tee -a "${RUN_DIR}/train.log"

# 5) inference on val (generate jsonl)
rm -f "${ANSWERS_FILE}"
deepspeed --include=localhost:0 --master_port="$((MASTER_PORT+1))" model/eval/vqa_infer.py \
  --version "${RUN_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --answer_type "open" \
  --val_data_path "${VAL_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --eval_vqa \
  --answers-file "${ANSWERS_FILE}" \
  --temperature 0.0 \
  --num_beams 1 \
  2>&1 | tee -a "${EVAL_OUTPUT_DIR}/infer.log"

# 6) metrics
python scripts/eval_stage2_cot_medplib.py \
  --pred-jsonl "${ANSWERS_FILE}" \
  --val-json "${VAL_JSON}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  2>&1 | tee -a "${EVAL_OUTPUT_DIR}/metrics.log"

echo "==> Done."
echo "==> Checkpoint dir: ${RUN_DIR}"
echo "==> Prediction jsonl: ${ANSWERS_FILE}"
echo "==> Metrics dir: ${EVAL_OUTPUT_DIR}"

