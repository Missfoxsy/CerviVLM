#!/bin/bash
# 使用 EAGLE 子模集方法对 Stage2 训练模型做「关键词级」视觉归因，并导出热力图。
# 与 scripts/run_two_stage_and_eval.sh 中的 EVAL_* 变量对齐即可复用同一套 checkpoint 与数据。
#
# 依赖：项目根目录下可 import vlm_fo1、scripts；EAGLE 在 PYTHONPATH 中（本脚本已设置）。
#
# 步骤：
#   1) 先跑 eval_stage2_cot 得到 results.json
#   2) 用 EAGLE/resources/build_eval_list_from_stage2.py 生成 eval-list（含 generated_ids / 关键词位置）
#   3) 本脚本调用 faithfulness_stage2_qwen.py 写 npy+json
#   4) EAGLE/visualize_ours.py 生成 visualization/*.png
#
# 用法示例：
#   bash scripts/run_eagle_stage2_visual.sh
#
# 仅跑可视化（归因已完成，且勿覆盖手改 eval_list）：
#   SKIP_BUILD_EVAL_LIST=1 SKIP_ATTRIBUTION=1 bash scripts/run_eagle_stage2_visual.sh
#
# 手改 eval_list 里 selected_interpretation_* / words 后，必须重跑第 2 步归因才会更新 json/npy；
# 仅可视化读的是 slico-.../json 里上次归因的结果，不会读 eval_list。
# 第 3 步默认在 json 或 npy 比已有 png 新时重画；若仍看到旧图可设 EAGLE_FORCE_VISUALIZATION=1。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/EAGLE:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# 与 run_two_stage_and_eval.sh 对齐的默认项
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

EVAL_BASE_MODEL=${EVAL_BASE_MODEL:-"${ROOT}/outputs/stage1_layerGated_taskAdaptive"}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-"${ROOT}/outputs/stage2_layerGated_taskAdaptive"}
# Stage2 标注 JSON：与 eval_stage2_cot 的 --test_json 同结构（含 id / image / regions / conversations）
EVAL_JSON=${EVAL_JSON:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/eagle_sample.json"}
# EVAL_OUTPUT_DIR：跑 eval_stage2_cot.py 时传入的 --output_dir，其下需有 results.json（模型预测列表）。
# build_eval_list 会用「本 JSON 里的样本」去 results.json 里按 id / image 对齐；子集评测时可另设目录。
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${ROOT}/eval_result/eagle_task2only"}
# 若 results.json 不在上述目录，可单独指定，例如只对 eagle_sample 跑过 eval 的输出目录：
#   export RESULTS_JSON=/path/to/eval_eagle10/results.json
RESULTS_JSON=${RESULTS_JSON:-"${EVAL_OUTPUT_DIR}/results.json"}
EVAL_DEVICE=${EVAL_DEVICE:-"cuda"}
EVAL_LANGUAGE=${EVAL_LANGUAGE:-"zh"}
# 仅跑 eval_list 中前 N 条归因（省时间）；不设则跑全部匹配到的条目
FAITHFULNESS_BEGIN=${FAITHFULNESS_BEGIN:-}
FAITHFULNESS_END=${FAITHFULNESS_END:-}

USE_HIERARCHICAL_FUSION=${USE_HIERARCHICAL_FUSION:-1}
HIERARCHICAL_NUM_HEADS=${HIERARCHICAL_NUM_HEADS:-8}
HIERARCHICAL_FUSION_MODE=${HIERARCHICAL_FUSION_MODE:-"task_adaptive"}

# 图片根：eval-list 里 image_path 为相对路径时，拼在此目录下
DATASETS_ROOT=${DATASETS_ROOT:-"/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/img"}

# tokenizer / AutoProcessor：与 eval_stage2_cot 的 DEFAULT_VLM_FO1_BASE 一致（仓库内 FO1 发布目录，默认已写入 build 脚本）
FO1_PROCESSOR_ROOT=${FO1_PROCESSOR_ROOT:-"${ROOT}/resources/VLM-FO1_Qwen2.5-VL-3B-v01"}

EAGLE_SAVE_ROOT=${EAGLE_SAVE_ROOT:-"${ROOT}/eagle_interpretation/stage2_layerGated_taskAdaptive"}
EVAL_LIST_OUT=${EVAL_LIST_OUT:-"${EAGLE_SAVE_ROOT}/eval_list.json"}
# EAGLE 归因：每步 2 次完整 VLM 前向，显存峰值高。division_number 越小 → region_size 越大 → 超像素越少、越省显存/时间。
DIVISION_NUMBER=${DIVISION_NUMBER:-32}
# 短边分辨率（仅 faithfulness 的 prepare_inputs）；显存不足可设 512
EAGLE_IMAGE_SIZE=${EAGLE_IMAGE_SIZE:-672}

# 仅把 results 里「能对应到当前 EVAL_JSON(stage2) 里 id/image」的条目写入 eval-list。
# 小样本（如 10 条 eagle_sample）时避免对 6000 条 results 逐条跑 processor。全量跑设 RESTRICT_TO_STAGE2=0。
RESTRICT_TO_STAGE2=${RESTRICT_TO_STAGE2:-1}

# 设为 1 时跳过第 1 步，不调用 build_eval_list_from_stage2.py，保留现有 eval_list.json（手改关键词/位置时必须开）
SKIP_BUILD_EVAL_LIST=${SKIP_BUILD_EVAL_LIST:-1}

HF_ARGS=()
if [[ "${USE_HIERARCHICAL_FUSION}" == "1" ]]; then
  HF_ARGS+=(--use_hierarchical_fusion --hierarchical_num_heads "${HIERARCHICAL_NUM_HEADS}" --hierarchical_fusion_mode "${HIERARCHICAL_FUSION_MODE}")
fi

if [[ "${SKIP_BUILD_EVAL_LIST}" == "1" ]]; then
  echo "[1/3] 跳过生成 eval-list（SKIP_BUILD_EVAL_LIST=1），使用已有: ${EVAL_LIST_OUT}"
  if [[ ! -f "${EVAL_LIST_OUT}" ]]; then
    echo "错误: eval_list 不存在，请先跑一次不带 SKIP_BUILD_EVAL_LIST 的脚本生成 ${EVAL_LIST_OUT}" >&2
    exit 1
  fi
else
  echo "[1/3] 从 results.json 生成 EAGLE eval-list -> ${EVAL_LIST_OUT}"
  mkdir -p "$(dirname "${EVAL_LIST_OUT}")"
  BUILD_EXTRA=()
  if [[ "${RESTRICT_TO_STAGE2}" == "1" ]]; then
    BUILD_EXTRA+=(--restrict-to-stage2)
  fi

  python "${ROOT}/EAGLE/resources/build_eval_list_from_stage2.py" \
    --stage2-json "${EVAL_JSON}" \
    --results-json "${RESULTS_JSON}" \
    --output "${EVAL_LIST_OUT}" \
    --datasets-root "${DATASETS_ROOT}" \
    --model_name_or_path "${FO1_PROCESSOR_ROOT}" \
    --max_keywords_per_sample 10 \
    "${BUILD_EXTRA[@]}"
fi

EXPL_DIR="${EAGLE_SAVE_ROOT}/slico-1.0-1.0-division-number-${DIVISION_NUMBER}"

if [[ "${SKIP_ATTRIBUTION:-0}" != "1" ]]; then
  echo "[2/3] EAGLE 归因（较慢，与评估同构的 VLM-FO1 前向）-> ${EXPL_DIR}"
  mkdir -p "${EAGLE_SAVE_ROOT}"
  FF_EXTRA=()
  if [[ -n "${FAITHFULNESS_BEGIN}" ]]; then
    FF_EXTRA+=(--begin "${FAITHFULNESS_BEGIN}")
  fi
  if [[ -n "${FAITHFULNESS_END}" ]]; then
    FF_EXTRA+=(--end "${FAITHFULNESS_END}")
  fi
  python "${ROOT}/EAGLE/resources/faithfulness_stage2_qwen.py" \
    --Datasets "${DATASETS_ROOT}" \
    --eval-list "${EVAL_LIST_OUT}" \
    --save-dir "${EAGLE_SAVE_ROOT}" \
    --base-model "${EVAL_BASE_MODEL}" \
    --checkpoint "${EVAL_CHECKPOINT}" \
    --device "${EVAL_DEVICE}" \
    --stage2-json "${EVAL_JSON}" \
    --apply-task-constraints \
    --language "${EVAL_LANGUAGE}" \
    --division-number "${DIVISION_NUMBER}" \
    --image-size "${EAGLE_IMAGE_SIZE}" \
    "${FF_EXTRA[@]}" \
    "${HF_ARGS[@]}"
else
  echo "[2/3] 跳过归因（SKIP_ATTRIBUTION=1）"
fi

echo "[3/3] 可视化 -> ${EXPL_DIR}/visualization"
VIS_ARGS=(--explanation-dir "${EXPL_DIR}" --Datasets "${DATASETS_ROOT}")
if [[ "${EAGLE_FORCE_VISUALIZATION:-0}" == "1" ]]; then
  VIS_ARGS+=(--force)
fi
python "${ROOT}/EAGLE/visualize_ours.py" "${VIS_ARGS[@]}"

echo "完成。说明：EAGLE 输出为子区域重要性累积图（论文中的 sentence/word-level faithfulness 可视化），"
echo "与仓库内 save_hierarchical_attn 的「层门控内部注意力」是不同视角，可互为补充。"
