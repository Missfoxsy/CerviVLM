#!/bin/bash
# Stage1 VT-FPN 层数消融：legacy 层级注意力（保留）+ 无门控，依次训练 + 评估 1/2/4/6 层
#
# 用法:
#   bash scripts/run_stage1_fpn_ablation.sh
#
# 环境变量:
#   FPN_LEVELS="1 2 4 6"        要跑的层数（默认 1 2 4 6）
#   CUDA_VISIBLE_DEVICES=0
#   NPROC_STAGE1=1
#   RESUME_TRAINING=1           断点续训（默认）
#   SKIP_TRAIN=1                仅评估
#   训练期间不跑 val（--no_val）；指标在训练结束后由 eval_stage1_align.py 计算
#   USE_HIERARCHICAL_FUSION=1   启用 legacy 层级注意力（默认 1）
#   HIERARCHICAL_FUSION_MODE=legacy
#
# 示例:
#   FPN_LEVELS="4 6" bash scripts/run_stage1_fpn_ablation.sh
#   SKIP_TRAIN=1 bash scripts/run_stage1_fpn_ablation.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4"}
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export WANDB_DISABLE_GIT=${WANDB_DISABLE_GIT:-"true"}

FPN_LEVELS=${FPN_LEVELS:-"1 2 4 6"}
RESUME_TRAINING=${RESUME_TRAINING:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
NPROC_STAGE1=${NPROC_STAGE1:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_STAGE1=${MASTER_PORT_STAGE1:-$(shuf -i 20000-24999 -n 1)}

STAGE1_EVAL_TEST_JSON=${STAGE1_EVAL_TEST_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/cot/stage1_align_val.json"}
ORIGIN_JSON=${ORIGIN_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json"}
EVAL_ROOT=${EVAL_ROOT:-"eval_result/stage1_fpn_ablation"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}
USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"legacy"}

SUMMARY_JSON="${EVAL_ROOT}/fpn_ablation_summary.json"

echo "======================================================"
echo "Stage1 FPN Ablation (legacy attention, no gate)"
echo "======================================================"
echo "FPN levels:      ${FPN_LEVELS}"
echo "Eval root:       ${EVAL_ROOT}"
echo "CUDA:            ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_STAGE1:    ${NPROC_STAGE1}"
echo "RESUME_TRAIN:    ${RESUME_TRAINING}"
echo "SKIP_TRAIN:      ${SKIP_TRAIN}"
echo "USE_HF:          ${USE_HIERARCHICAL_FUSION}"
echo "HF mode:         ${HIERARCHICAL_FUSION_MODE}"
echo "======================================================"

RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

declare -A CONFIG_FOR_LEVEL=(
  [1]="configs/train_config_stage1_fpn_ablation_1.yaml"
  [2]="configs/train_config_stage1_fpn_ablation_2.yaml"
  [4]="configs/train_config_stage1_fpn_ablation_4.yaml"
  [6]="configs/train_config_stage1_fpn_ablation_6.yaml"
)

declare -A OUT_FOR_LEVEL=(
  [1]="outputs/stage1_fpn_ablation_1layer_legacyAttn"
  [2]="outputs/stage1_fpn_ablation_2layer_legacyAttn"
  [4]="outputs/stage1_fpn_ablation_4layer_legacyAttn"
  [6]="outputs/stage1_fpn_ablation_6layer_legacyAttn"
)

mkdir -p "${EVAL_ROOT}"

for N in ${FPN_LEVELS}; do
  CONFIG="${CONFIG_FOR_LEVEL[$N]:-}"
  OUT_DIR="${OUT_FOR_LEVEL[$N]:-}"
  EVAL_DIR="${EVAL_ROOT}/fpn_${N}layer_legacyAttn"

  if [[ -z "${CONFIG}" || -z "${OUT_DIR}" ]]; then
    echo "[WARN] 不支持的 fpn_num_levels=${N}，跳过（当前支持 1/2/4/6）"
    continue
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG}"
    exit 1
  fi

  echo ""
  echo "---------- FPN ${N} layer(s) ----------"
  echo "Config:  ${CONFIG}"
  echo "Train:   ${OUT_DIR}"
  echo "Eval:    ${EVAL_DIR}"

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    echo "[Train] Stage1 fpn_num_levels=${N} ..."
    torchrun --nproc_per_node="${NPROC_STAGE1}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT_STAGE1}" \
      scripts/train_stage1_align.py \
      --config "${CONFIG}" \
      --fpn_num_levels "${N}" \
      --no_val \
      "${RESUME_ARGS[@]}"
  else
    echo "[Skip] 训练 (SKIP_TRAIN=1)"
  fi

  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "[ERROR] checkpoint 目录不存在: ${OUT_DIR}"
    exit 1
  fi

  echo "[Eval] Stage1 task 0,1,2 ..."
  EVAL_CMD=(
    python scripts/eval_stage1_align.py
    --base_model "${OUT_DIR}"
    --checkpoint "${OUT_DIR}"
    --test_json "${STAGE1_EVAL_TEST_JSON}"
    --output_dir "${EVAL_DIR}"
    --origin_json "${ORIGIN_JSON}"
    --device "${EVAL_DEVICE}"
    --language "${EVAL_LANGUAGE}"
    --task_ids "0,1,2"
    --fpn_num_levels "${N}"
  )
  if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
    EVAL_CMD+=(
      --use_hierarchical_fusion
      --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}"
    )
  fi
  "${EVAL_CMD[@]}"
  echo "[Done] fpn=${N} -> ${EVAL_DIR}/metrics.json"
done

echo ""
echo "[Summary] 汇总各层 metrics -> ${SUMMARY_JSON}"
python - <<'PY' "${EVAL_ROOT}" "${SUMMARY_JSON}" ${FPN_LEVELS}
import json
import os
import sys

eval_root = sys.argv[1]
summary_path = sys.argv[2]
levels = [int(x) for x in sys.argv[3:]]

summary = {"fpn_ablation": [], "eval_root": eval_root}
for n in levels:
    tag = f"fpn_{n}layer_legacyAttn"
    metrics_path = os.path.join(eval_root, tag, "metrics.json")
    meta_path = os.path.join(f"outputs/stage1_fpn_ablation_{n}layer_legacyAttn", "fpn_ablation.json")
    entry = {"fpn_num_levels": n, "eval_dir": os.path.join(eval_root, tag)}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            entry["fpn_meta"] = json.load(f)
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            entry["metrics"] = json.load(f)
    else:
        entry["metrics"] = None
        entry["error"] = f"missing {metrics_path}"
    summary["fpn_ablation"].append(entry)

os.makedirs(eval_root, exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Wrote {summary_path}")
for e in summary["fpn_ablation"]:
    m = e.get("metrics") or {}
    overall = m.get("overall") or m.get("summary") or {}
    print(f"  fpn={e['fpn_num_levels']}: metrics keys={list(m.keys())[:8]} overall={overall}")
PY

echo ""
echo "Stage1 FPN 消融全部完成。"
echo "  训练输出: outputs/stage1_fpn_ablation_*layer_legacyAttn"
echo "  评估输出: ${EVAL_ROOT}/fpn_*layer_legacyAttn"
echo "  汇总:     ${SUMMARY_JSON}"
