#!/bin/bash
#SBATCH --job-name=fo1-eval-stage1
#SBATCH --account=fengjing
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# conda activate cervivlm
set -u

# 2) 项目根目录（sbatch 在仓库根提交时 SLURM_SUBMIT_DIR 即为根目录）
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/logs"

# 3) 运行参数（可通过 sbatch --export 覆盖）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_FO1_ATTN_IMPLEMENTATION="${VLM_FO1_ATTN_IMPLEMENTATION:-sdpa}"

CKPT_DIR="${CKPT_DIR:-outputs/stage1_layerGated_sigmoid/checkpoint-1000}"
TEST_JSON="${TEST_JSON:-data/stage1_align_val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_result/stage1_layerGated_sigmoid_ckpt1000}"
TASK_IDS="${TASK_IDS:-0,1,2}"

export USE_HIERARCHICAL_FUSION="${USE_HIERARCHICAL_FUSION:-1}"
export HIERARCHICAL_NUM_HEADS="${HIERARCHICAL_NUM_HEADS:-8}"
export HIERARCHICAL_FUSION_MODE="${HIERARCHICAL_FUSION_MODE:-gated}"
export SAVE_HIER_ATTN_PER_TASK_MAX="${SAVE_HIER_ATTN_PER_TASK_MAX:-10}"
export SAVE_HIER_ATTN_SEED="${SAVE_HIER_ATTN_SEED:-42}"
export STAGE1_SAVE_HIER_ATTN_DIR="${STAGE1_SAVE_HIER_ATTN_DIR:-}"

echo "==> CKPT_DIR=${CKPT_DIR}"
echo "==> TEST_JSON=${TEST_JSON}"
echo "==> OUTPUT_DIR=${OUTPUT_DIR}"
echo "==> TASK_IDS=${TASK_IDS}"
echo "==> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==> VLM_FO1_ATTN_IMPLEMENTATION=${VLM_FO1_ATTN_IMPLEMENTATION}"

bash "${ROOT_DIR}/scripts/eval_stage1_align.sh" \
  "${CKPT_DIR}" \
  "${TEST_JSON}" \
  "${OUTPUT_DIR}" \
  "${TASK_IDS}"

echo "==> Done (Stage1 eval)."
