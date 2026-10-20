#!/bin/bash
#SBATCH --job-name=fo1-two-stage
#SBATCH --account=fengjing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
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

# 3) 单机单卡时常用：Slurm 已分配 GPU 0；若多卡请改 #SBATCH --gres 并设置 CUDA_VISIBLE_DEVICES / NPROC_*
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 未安装 flash-attn 或平台不支持时：使用 PyTorch SDPA（默认），勿设 flash_attention_2
# 本地有 FA2 且需最快推理时可：export VLM_FO1_ATTN_IMPLEMENTATION=flash_attention_2
export VLM_FO1_ATTN_IMPLEMENTATION="${VLM_FO1_ATTN_IMPLEMENTATION:-flash_attention_2}"

# 4) 跑两阶段流水线（逻辑仍在 scripts/run_two_stage_and_eval.sh，便于本地与集群共用）
# 示例：
#   sbatch scripts/run_two_stage_and_eval.sbatch
#   sbatch --export=ALL,START_FROM=stage2 scripts/run_two_stage_and_eval.sbatch
#   sbatch --export=ALL,RESUME_TRAINING=0,STAGE1_OUTPUT_DIR=outputs/my_s1 scripts/run_two_stage_and_eval.sbatch
bash "${ROOT_DIR}/scripts/run_two_stage_and_eval.sh"

echo "==> Done (FO1LayerAtt two-stage pipeline)."
