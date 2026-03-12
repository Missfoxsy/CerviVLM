#!/bin/bash
set -euo pipefail

HOVER_PRED_ROOT="/home/nas3/biod/shiyu/data/data_vl_20260115/pred"
OUTPUT_ROOT="/home/nas3/biod/shiyu/data/data_vl_20260115/path_features"
DATASET_DIR="/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/final"
TRAIN_JSON="${DATASET_DIR}/stageA_train.json"
VAL_JSON="${DATASET_DIR}/stageA_val.json"

if [ ! -f "${TRAIN_JSON}" ]; then
    echo "Error: train json not found: ${TRAIN_JSON}"
    exit 1
fi
if [ ! -f "${VAL_JSON}" ]; then
    echo "Error: val json not found: ${VAL_JSON}"
    exit 1
fi
if [ ! -d "${HOVER_PRED_ROOT}" ]; then
    echo "Error: hover pred root not found: ${HOVER_PRED_ROOT}"
    exit 1
fi

echo "[build_path_features] Step 1/2: stageA_train.json -> ${OUTPUT_ROOT}"
python scripts/build_path_features.py \
    --dataset_json "${TRAIN_JSON}" \
    --hover_pred_root "${HOVER_PRED_ROOT}" \
    --output_root "${OUTPUT_ROOT}"

echo "[build_path_features] Step 2/2: stageA_val.json -> ${OUTPUT_ROOT}"
python scripts/build_path_features.py \
    --dataset_json "${VAL_JSON}" \
    --hover_pred_root "${HOVER_PRED_ROOT}" \
    --output_root "${OUTPUT_ROOT}"

echo "[build_path_features] Done. stageA_train + stageA_val written to ${OUTPUT_ROOT}"