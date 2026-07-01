python scripts/infer_stage2_task1_attention.py \
  --test_json /home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json \
  --sample_id identity_2919 \
  --base_model outputs/stage1_layerGated_taskAdaptive \
  --checkpoint outputs/stage2_layerGated_taskAdaptive \
  --use_hierarchical_fusion \
  --hierarchical_num_heads 8 \
  --hierarchical_fusion_mode task_adaptive \
  --save_spatial_attn \
  --output_dir attention_infer/task1_single