# CerviVLM

在 [VLM-FO1](https://arxiv.org/abs/2509.25916)（Qwen2.5-VL-3B）上做宫颈细胞两阶段 LoRA，并比较多层 ROI 特征的融合方式。本仓库只含训练、评估和模型代码。基座权重、图像和标注 JSON 需要单独准备，不包含在压缩包里。

上游 VLM-FO1 说明见 `docs/VLM-FO1_README.md`。

## 环境

已在 Linux + NVIDIA GPU 上用下面的版本跑通。请固定 `transformers==4.50.1`。升到 5.x 后，本仓库里的 Qwen2.5-VL 配置无法加载。

```bash
conda create -n cervivlm python=3.10 -y
conda activate cervivlm
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install peft safetensors sentencepiece protobuf tensorboard
pip install nltk rouge-score
```

`requirements.txt` 里的核心约束：

| 包 | 版本 |
|---|---|
| python | 3.10 |
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| transformers | 4.50.1 |
| timm | 1.0.9 |
| accelerate | 1.4.0 |

可选：`bitsandbytes`、`wandb`。不用 wandb 时，把配置里的 `misc.report_to` 改成 `tensorboard`。

当前环境里 `flash_attn` 和 `torchaudio` 的 CUDA 扩展与 torch 不匹配。训练和评估入口会先导入 `utils/flash_attn_compat.py`，把这两个包藏起来，注意力走 SDPA。不需要为了跑通而重装 flash-attn。

下载基座并放到：

```text
resources/VLM-FO1_Qwen2.5-VL-3B-v01
```

Hugging Face：`omlab/VLM-FO1_Qwen2.5-VL-3B-v01`。

## 数据

配置里的 JSON 路径是本机绝对路径，换机器时改 `configs/*.yaml` 里的 `data.train_json` / `data.val_json`。

| 阶段 | 内容 | 默认文件 |
|---|---|---|
| Stage1 | 领域对齐：定位、带描述定位、区域描述 | `stage1_align_train.json` / `stage1_align_val.json` |
| Stage2 | CoT 多任务（task 0–5） | `stage2_cot_train.json` / `stage2_cot_val.json` |
| 评估 | task0/1 的原始框 | `origin_json`（验证集原始标注） |

每条样本是一条对话。图像路径写在 JSON 里。`language` 为 `zh` 或 `en`，必须和训练时一致。

## 训练

主模型是 `hierarchical_fusion_mode: task_adaptive`（跨层注意力 + Sigmoid 门控）。

从头跑两阶段并评估：

```bash
CUDA_VISIBLE_DEVICES=0 START_FROM=stage1 RESUME_TRAINING=0 \
  bash scripts/run_two_stage_and_eval.sh
```

| 变量 | 作用 |
|---|---|
| `START_FROM` | `stage1` / `stage2` / `eval` |
| `RESUME_TRAINING` | `1` 从 `output_dir` 里最新的 `checkpoint-*` 续训；`0` 从头训 |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU |
| `NPROC_STAGE1` / `NPROC_STAGE2` | `torchrun` 进程数，不要超过可见 GPU 数 |
| `STAGE1_CONFIG` / `STAGE2_CONFIG` | 配置文件 |

只训练、不走总脚本：

```bash
torchrun --nproc_per_node=1 scripts/train_stage1_align.py \
  --config configs/train_config_stage1_lora.yaml

torchrun --nproc_per_node=1 scripts/train_stage2_cot.py \
  --config configs/train_config_stageB_lora.yaml
```

Stage2 的 `model.base_model` 必须指向对应融合方式的 Stage1 输出。不要用 task_adaptive 的 Stage1 去初始化 concat 或 softmax_gate。

### 论文实验对应

本仓库只保留论文里的实验，不包含未写入论文的融合方式或外部数据集。

| 论文 | 设置 | 怎么跑 |
|---|---|---|
| 主结果 Table I，CerviVLM | 两阶段 + 跨层注意力 + Sigmoid 门控 | `scripts/run_two_stage_and_eval.sh` |
| Table II (a) | 四层直接拼接，无注意力、无门控 | `use_hierarchical_fusion: false` |
| Table II (b) | 逐层 Linear 再拼接 | `FUSION_MODE=concat bash scripts/run_matched_fusion_ablation.sh` |
| Table II (c) | 只有跨层注意力 | `FUSION_MODE=legacy bash scripts/run_matched_fusion_ablation.sh` |
| Table II (d) | 注意力 + Softmax 门控 | `FUSION_MODE=softmax_gate bash scripts/run_stage2_fusion_softmax_gate.sh` |
| Table II (e) | 注意力 + Sigmoid 门控（主模型） | 与 Table I 相同 |
| Table II (f) | 不做 Stage1 | `scripts/run_stage2_ablation_no_stage1.sh` |
| Table II (g) | 只做 Stage1 | `scripts/eval_stage1_align.py` |
| Fig. 4 | FPN 1/2/4/6 层 | `scripts/run_stage1_fpn_ablation.sh` |
| PS3C、SIPaKMeD | 零样本分类 | `scripts/eval_ps3c_classification.py`、`scripts/eval_sipakmed_posneg.py` |

Table I 里的 Qwen2.5-VL、LLaVA、InternVL、LLaVA-Med、MedPLIB、VLM-FO1，以及 Table III 的检测器，是外部基线，不在本仓库训练。

### 融合消融

同一融合方式依次跑 Stage1、Stage2、评估。Stage2 必须从同一融合方式的 Stage1 初始化。

```bash
FUSION_MODE=concat CUDA_VISIBLE_DEVICES=0 START_FROM=stage1 \
  bash scripts/run_matched_fusion_ablation.sh

FUSION_MODE=legacy CUDA_VISIBLE_DEVICES=0 START_FROM=stage1 \
  bash scripts/run_matched_fusion_ablation.sh

FUSION_MODE=softmax_gate CUDA_VISIBLE_DEVICES=0 START_FROM=stage1 \
  bash scripts/run_stage2_fusion_softmax_gate.sh
```

`FUSION_MODE` 只用论文中的四种：`concat`、`legacy`、`softmax_gate`、`task_adaptive`。

| 模式 | 论文 | Stage1 | Stage2 |
|---|---|---|---|
| 关闭层级融合 | (a) 直接拼接 | 在 Stage1 配置里设 `use_hierarchical_fusion: false` | 同左 |
| concat | (b) Lin+Cat | `configs/train_config_stage1_lora_concat.yaml` | `configs/train_config_stageB_lora_concat.yaml` |
| legacy | (c) HA only | `configs/train_config_stage1_lora_legacy.yaml` | `configs/train_config_stageB_lora_legacy.yaml` |
| softmax_gate | (d) | `configs/train_config_stage1_lora_softmax_gate.yaml` | `configs/train_config_stageB_lora_softmax_gate.yaml` |
| task_adaptive | (e) | `configs/train_config_stage1_lora.yaml` | `configs/train_config_stageB_lora.yaml` |

不做 Stage1、只做 Stage2（Table II (f)）：`configs/train_config_stageB_lora_no_stage1.yaml`，脚本 `scripts/run_stage2_ablation_no_stage1.sh`。

FPN 深度（Fig. 4）使用 legacy 注意力、无门控，层数 1/2/4/6：`bash scripts/run_stage1_fpn_ablation.sh`。作图：`scripts/plot_stage1_fpn_ablation_loc.py`。病变分级的逐类 F1（Fig. 5）用 `scripts/plot_per_class_bar_type34.py`。

## 评估

```bash
python scripts/eval_stage2_cot.py \
  --base_model outputs/stage1_layerGated_taskAdaptive \
  --checkpoint outputs/stage2_layerGated_taskAdaptive \
  --test_json /path/to/stage2_cot_val.json \
  --origin_json /path/to/origin_val.json \
  --output_dir eval_result/stage2_layerGated_taskAdaptive \
  --language zh \
  --task_ids 0,1,2,3,4,5 \
  --use_hierarchical_fusion \
  --hierarchical_num_heads 8 \
  --hierarchical_fusion_mode task_adaptive
```

`--base_model` 是 Stage1 LoRA 目录，`--checkpoint` 是 Stage2 LoRA 目录。脚本先加载官方基座，合并 Stage1，再加载 Stage2。

Stage1 评估：`scripts/eval_stage1_align.py`。

结果写到 `--output_dir`：

| 文件 | 内容 |
|---|---|
| `results.json` | 每条样本的预测和分数 |
| `metrics.json` | 汇总指标 |
| `gate_stats.json` | 层级门控统计（有门控时） |

### 指标

| 任务 | 含义 | 指标 |
|---|---|---|
| type0 | 定位 | Precision / Recall / F1 / mean IoU / mAP。表中 Loc 用 type0 mAP×100 |
| type1 | 带描述的定位 | 同上。表中 REC 用 type1 mAP×100 |
| type2 | 区域描述 | BLEU、ROUGE-1/2/L，以及英文管线 SBERT |
| type3 | 细胞类型 | Accuracy 与每类 P/R/F1。表中 Grade 用 Accuracy×100 |
| type4 | 来源 / 成熟度 | 同上。表中 Maturation 用 Accuracy×100 |
| type5 | 诊断依据 | 从诊断结论抽取类别后算分类指标。表中 Reason 用 Accuracy×100 |

Type2 SBERT 默认开启：中文预测和标注用规则译成英文，再与对齐的英文 GT 算余弦相似度。模型是 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`。关闭用 `--no-compute_task2_sbert`。英文 GT 默认 `eval_results/en_gt_ref_stage2_cot_val.json`（需自备，按样本 `id` 对齐）。已有 `results.json` 时也可以后补：

```bash
HF_HUB_OFFLINE=1 python scripts/compute_task2_sbert_en.py \
  --input eval_result/<run>/results.json \
  --name <run> \
  --en-gt-ref eval_results/en_gt_ref_stage2_cot_val.json \
  --device cuda
```

脚本会把 SBERT 写回该目录的 `metrics.json`。

## 超参数

未写出的字段沿用对应 yaml。两阶段都是 `num_epochs: 1`、`batch_size: 1`、`gradient_accumulation_steps: 8`（有效 batch 8）、`image_size: 896`、`max_regions: 100`、`seed: 42`、`max_grad_norm: 1.0`、`mixed_precision: bf16`。

### Stage1 领域适配

配置：`configs/train_config_stage1_lora.yaml`。

| 参数 | 值 | 说明 |
|---|---|---|
| `learning_rate` | 1e-4 | LoRA 与 projector |
| `warmup_steps` | 300 | |
| `lr_scheduler.type` | cosine | `warmup_ratio: 0.05` |
| `optimizer.weight_decay` | 0.05 | AdamW |
| `save_steps` / `eval_steps` | 1000 | `save_steps` 必须是 `eval_steps` 的整数倍 |
| `logging_steps` | 10 | |
| `save_total_limit` | 5 | |
| `lora.r` / `alpha` / `dropout` | 32 / 64 / 0.1 | |
| `lora.target_modules` | q,k,v,o,gate,up,down | Attention 和 MLP |
| 可训练 | LoRA、`mm_projector`、`mm_projector_aux`、`hierarchical_fusion` | |
| 冻结 | vision tower | |
| `dataloader_num_workers` | 4 | |

### Stage2 CoT

配置：`configs/train_config_stageB_lora.yaml`。

| 参数 | 值 | 说明 |
|---|---|---|
| `learning_rate` | 5e-5 | Attention LoRA |
| `gate_lr_ratio` | 0.2 | `hierarchical_fusion` 的学习率 = `learning_rate * 0.2` |
| `warmup_ratio` | 0.1 | |
| `weight_decay` | 0.1 | |
| `label_smoothing` | 0.05 | |
| `lora.r` / `alpha` / `dropout` | 32 / 64 / 0.05 | |
| `lora.target_modules` | q,k,v,o | 不训练 MLP，保留 Stage1 的领域适配 |
| 可训练 | LoRA、`model.object_vp_extractor.hierarchical_fusion` | 融合模块用 PEFT `modules_to_save` 存全量权重 |
| 冻结 | vision tower、vision projector、region projector、LLM MLP | |
| `gradient_checkpointing` | true | |
| `load_best_model_at_end` | true | 按 `eval_loss` |
| `early_stopping_patience` | 3 | |

`mix_stageA_data` 默认关闭。打开后按 `stageA_data_ratio`（配置里注释为 0.3）混入 Stage1 文本，用来减轻遗忘。

### 融合模块

实现：`vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py` 的 `HierarchicalFeatureFusion`。

四层 FPN ROI 特征，每层先 ROIAlign 再空间平均，得到 `[N, 512]`。`hierarchical_num_heads: 8`。

| 模式 | 做法 |
|---|---|
| 关闭融合，Table II (a) | `use_hierarchical_fusion: false`。四层 512 维直接拼接成 2048。维度已经对齐时不再接 Linear |
| `concat`，Table II (b) | 每层各自 `Linear(512→512)`，拼接后再 `Linear(2048→2048)`。无注意力、无门控 |
| `legacy`，Table II (c) | 逐层编码后，相邻层做 Multi-Head Attention（当前层 query，上一层 key/value），拼接再 Linear。无门控 |
| `softmax_gate`，Table II (d) | 与 task_adaptive 同一结构，门控改为 Softmax |
| `task_adaptive`，Table II (e) | 与 legacy 相同的注意力，再乘层级门控。Sigmoid、下限、行归一化 |

`task_adaptive` / `softmax_gate` 的门控常数写在代码里，不在 yaml 中：

| 常数 | 值 | 作用 |
|---|---|---|
| `task_adaptive_gate_temp` | 0.8 | 温度。logits 先除以温度 |
| `task_adaptive_gate_floor` | 0.02 | 仅 Sigmoid 路径。门控下限，避免某一层变成 0 |
| `task_adaptive_task_bias_scale` | 2.0 | 任务偏置的乘数 |
| `task_level_bias` | 6×4 | task0/4 偏粗层，task2/3/5 偏细层，task1 均衡。可学习 |
| `level_selector` | Linear(2048→128)→ReLU→Linear(128→4) | 由四层拼接特征预测门控 logits |

Sigmoid 路径：`sigmoid(z/0.8)`，再线性抬到 `[0.02, 1]`，最后按层归一化使和为 1。Softmax 路径：`softmax(z/0.8)`，四层互相竞争。

### 评估参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | cuda | |
| `--language` | zh | 与训练一致 |
| `--task_ids` | 0,1,2,3,4,5 | |
| `--hierarchical_fusion_mode` | 与训练相同 | 评估时必须和 checkpoint 一致 |
| `--save_every_n_samples` | 1000 | 中途写 `results.json` |
| `--max_samples` | 0 | 0 表示全量 |
| `--compute_task2_sbert` | true | `--no-compute_task2_sbert` 关闭 |
| `--task2_sbert_model` | paraphrase-multilingual-mpnet-base-v2 | |
| `--task2_en_gt_ref` | `eval_results/en_gt_ref_stage2_cot_val.json` | |
| `--save_hierarchical_attn_per_task_max` | 10 | 每个任务随机导出的注意力图数量 |
| `--save_hierarchical_attn_seed` | 42 | |

## 目录

```text
configs/          训练 yaml
dataset/          CervicalDataset 与任务约束文本
scripts/          训练、评估、消融启动脚本
utils/            flash_attn / torchaudio 兼容垫片等
vlm_fo1/          模型、融合模块、数据整理
resources/        基座权重（自行下载，不进压缩包）
outputs/          LoRA 与 checkpoint（不进压缩包）
eval_result/      评估输出（不进压缩包）
```
