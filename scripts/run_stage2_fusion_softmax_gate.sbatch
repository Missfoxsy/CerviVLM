#!/bin/bash
#SBATCH --job-name=fo1-fusion-softmax-gate
#SBATCH --account=fengjing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/stage2_fusion_ablation/matched_softmax_gate_%j.out
#SBATCH --error=logs/stage2_fusion_ablation/matched_softmax_gate_%j.err

set -eo pipefail
source /home/shiyu/project/miniconda3/bin/activate covt
set -u

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p logs/stage2_fusion_ablation

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FUSION_MODE=softmax_gate
export START_FROM="${START_FROM:-stage1}"
bash "${ROOT_DIR}/scripts/run_matched_fusion_ablation.sh"
