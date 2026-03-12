# VLM-FO1 宫颈细胞定位微调 - 快速开始

## 快速开始

### 1. 准备数据

确保数据集文件存在：
```bash
dataset/stage1_train_region_text.json
```

### 2. 准备模型

下载或确认预训练模型存在：
```bash
resources/VLM-FO1_Qwen2.5-VL-3B-v01/
```

或者使用HuggingFace模型ID（在配置文件中设置）。

### 3. 配置训练参数

编辑 `configs/train_config.yaml`，根据需要调整参数。

### 4. 开始训练

**方式1：使用Python脚本（推荐）**
```bash
python scripts/train_cervical.py --config configs/train_config.yaml
```

**方式2：使用Shell脚本**
```bash
bash scripts/train_cervical.sh
```

**注意：** 默认配置已启用LoRA微调，可以大幅减少显存占用。如果需要全参数微调，请在配置文件中设置 `lora.enabled: false`。

### 5. 评估模型

```bash
python scripts/eval_cervical.py \
    --model_path outputs/cervical_finetune \
    --data_json dataset/stage1_train_region_text.json \
    --output_dir eval_results
```

## 文件结构

```
VLM-FO1/
├── dataset/
│   ├── stage1_train_region_text.json    # 训练数据
│   └── cervical_dataset.py              # 数据集类
├── scripts/
│   ├── train_cervical.py                 # 训练脚本
│   └── eval_cervical.py                  # 评估脚本
├── utils/
│   └── data_utils.py                     # 数据处理工具
├── configs/
│   └── train_config.yaml                 # 训练配置
└── docs/
    └── train_cervical.md                  # 详细文档
```

## LoRA微调说明

默认配置已启用LoRA微调，主要优势：
- **显存占用更少**：相比全参数微调，显存占用大幅减少
- **训练速度更快**：只训练少量参数，训练速度更快
- **模型文件更小**：只保存LoRA权重，文件大小显著减小

### LoRA配置参数

在 `configs/train_config.yaml` 中可以调整LoRA参数：

```yaml
lora:
  enabled: true      # 是否启用LoRA
  r: 8              # LoRA rank（建议8-16）
  alpha: 16         # LoRA alpha（通常为r的2倍）
  dropout: 0.05     # LoRA dropout
  target_modules: null  # 目标模块，null使用默认值
```

### 切换到全参数微调

如果需要全参数微调，设置：
```yaml
lora:
  enabled: false
```

## Wandb可视化

默认配置已启用Wandb可视化，可以实时查看训练损失曲线和指标。

### 使用Wandb

1. **安装和登录**
   ```bash
   pip install wandb
   wandb login
   ```

2. **配置Wandb项目**
   在 `configs/train_config.yaml` 中可以配置：
   ```yaml
   misc:
     report_to: "wandb"  # 使用wandb可视化
   
   wandb:
     project: "vlm-fo1-cervical"  # 项目名称
     entity: null  # 团队名称（可选）
     tags: ["cervical", "vlm-fo1", "lora"]  # 标签
   ```

3. **查看训练过程**
   - 训练开始后，会自动在wandb网站上创建运行
   - 访问 https://wandb.ai 查看实时训练曲线
   - 可以查看loss、learning rate、评估指标等

4. **切换到TensorBoard**
   如果需要使用TensorBoard，设置：
   ```yaml
   misc:
     report_to: "tensorboard"
   ```
   然后使用：
   ```bash
   tensorboard --logdir outputs/cervical_finetune/logs
   ```

## 详细文档

更多详细信息请参考：[docs/train_cervical.md](docs/train_cervical.md)

