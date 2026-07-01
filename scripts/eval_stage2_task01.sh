CUDA_VISIBLE_DEVICES=7 python scripts/eval_stage2_cot_task01_bbox.py \
  --base_model resources/VLM-FO1_Qwen2.5-VL-3B-v01 \
  --checkpoint outputs/stage2_no_stage1_ablation \
  --eval_json /home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json \
  --origin_json /home/nas3/biod/shiyu/data/data_vl_20260115/origin_data/data_vl_20260115_val.json \
  --output_dir eval_result/stage2_no_stage1_ablation \
  --language zh \
  --use_hierarchical_fusion \
  --hierarchical_fusion_mode task_adaptive