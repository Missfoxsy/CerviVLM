#!/bin/bash

# GPU selection: 只使用空闲的 GPU
# 根据 nvidia-smi，GPU 6,7 最空闲（只有约 10 MB 被占用）
# 如果只有 2 张 GPU 可用，使用 6,7；如果需要 4 张，可以加上 0（但需要确保有足够显存）
# 检查 GPU 使用情况：nvidia-smi
export CUDA_VISIBLE_DEVICES=6,7

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# 使用 2 张 GPU（对应 CUDA_VISIBLE_DEVICES 中的 2 张）
# 如果显存充足，可以改为 4
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# Model path: relative to project root or absolute path
llm=/home/nas3/biod/shiyu/project/QwenProject/models/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=1e-4
batch_size=1
grad_accum_steps=8  # 增加梯度累积步数以补偿更少的 GPU

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
train_datasets=cervical_train
eval_datasets=cervical_test

# Output configuration
run_name="qwen2vl-cervical-finetune"
output_dir=./output/cervical_3B_lora8

# 下面data_flatten为True时，会报错，因为数据集中的图片大小不一，导致无法进行batching
# LoRA 配置
use_lora=True
lora_r=8
lora_alpha=16
lora_dropout=0.05

# 自定义损失配置
# use_custom_loss=True
# iou_loss_weight=1.0
# bbox_loss_weight=1.0

# 混合bbox回归头配置（可选）
# use_hybrid_bbox_head=True  # 设置为True启用混合bbox回归头
# hybrid_bbox_loss_weight=1 # 混合bbox回归损失的权重
# 注意：始终使用ground truth labels进行训练，混合回归头通过hybrid_bbox_loss提供额外监督

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${train_datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --use_lora ${use_lora} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --use_custom_loss False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"
    #--resume_from_checkpoint ./output/cervical/checkpoint-500" 

# Launch training (print dataset selections for sanity check)
echo "Using train_datasets=${train_datasets}"
echo "Using eval_datasets=${eval_datasets}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

