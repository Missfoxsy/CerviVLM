# VLM-FO1 宫颈细胞定位微调指南

本文档介绍如何使用VLM-FO1框架对Qwen2.5-VL模型进行微调，以实现阳性宫颈细胞定位任务。

## 目录

- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型准备](#模型准备)
- [训练步骤](#训练步骤)
- [评估模型](#评估模型)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

## 环境准备

### 1. 安装依赖

确保已安装项目所需的所有依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch >= 2.0.0
- Transformers >= 4.50.0
- Pillow
- PyYAML
- tqdm

### 2. 检查CUDA

确保CUDA可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 数据准备

### 数据格式

数据集应为JSON格式，每个样本包含以下字段：

```json
{
  "id": "cervical_00000000",
  "image": "/path/to/image.jpg",
  "regions": [
    {
      "region_id": "region0",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.95
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请在图像中定位阳性宫颈细胞区域。"
    },
    {
      "from": "assistant",
      "value": "<ground>阳性宫颈细胞</ground><object><region0></object>"
    }
  ]
}
```

### 数据验证

在训练前，建议验证数据集的完整性：

```python
import json

with open('dataset/stage1_train_region_text.json', 'r') as f:
    data = json.load(f)

print(f"总样本数: {len(data)}")
print(f"样本示例: {data[0]}")
```

## 模型准备

### 下载预训练模型

从HuggingFace下载VLM-FO1预训练模型：

```bash
# 方式1: 使用HuggingFace模型ID（推荐）
# 在configs/train_config.yaml中设置:
# base_model: "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"

# 方式2: 下载到本地
# 将模型放置在 resources/VLM-FO1_Qwen2.5-VL-3B-v01/ 目录下
```

模型结构应包含：
- `config.json`
- `tokenizer.json`
- `model.safetensors` 或 `pytorch_model.bin`
- 其他必要的模型文件

## 训练步骤

### 1. 配置训练参数

编辑 `configs/train_config.yaml` 文件，根据您的需求调整参数：

```yaml
model:
  base_model: "resources/VLM-FO1_Qwen2.5-VL-3B-v01"

data:
  train_json: "dataset/stage1_train_region_text.json"
  val_split: 0.1
  image_size: 1024
  max_regions: 100

training:
  output_dir: "outputs/cervical_finetune"
  num_epochs: 5
  batch_size: 2
  learning_rate: 2e-5
  # ... 其他参数
```

### 2. 开始训练

运行训练脚本：

```bash
python scripts/train_cervical.py --config configs/train_config.yaml
```

### 3. 恢复训练

如果训练中断，可以从checkpoint恢复：

```bash
python scripts/train_cervical.py \
    --config configs/train_config.yaml \
    --resume_from_checkpoint outputs/cervical_finetune/checkpoint-1000
```

### 4. 监控训练

训练过程中会生成日志，可以使用TensorBoard查看：

```bash
tensorboard --logdir outputs/cervical_finetune/logs
```

## 评估模型

### 基本评估

使用评估脚本评估模型性能：

```bash
python scripts/eval_cervical.py \
    --model_path outputs/cervical_finetune \
    --data_json dataset/stage1_train_region_text.json \
    --output_dir eval_results \
    --iou_threshold 0.5
```

### 评估参数说明

- `--model_path`: 微调后的模型路径
- `--data_json`: 评估数据JSON文件路径
- `--output_dir`: 评估结果输出目录
- `--iou_threshold`: IoU阈值，用于判断预测框和真实框是否匹配
- `--val_split`: 如果使用训练数据评估，指定验证集比例
- `--no_visualize`: 禁用可视化（加快评估速度）

### 评估指标

评估脚本会计算以下指标：

- **平均IoU**: 预测边界框与真实边界框的平均IoU
- **精确率 (Precision)**: TP / (TP + FP)
- **召回率 (Recall)**: TP / (TP + FN)
- **F1分数**: 2 * Precision * Recall / (Precision + Recall)

评估结果会保存在 `eval_results/evaluation_results.json`。

## 配置说明

### 训练配置详解

#### 模型配置

```yaml
model:
  base_model: "resources/VLM-FO1_Qwen2.5-VL-3B-v01"
```

- `base_model`: 预训练模型路径或HuggingFace模型ID

#### 数据配置

```yaml
data:
  train_json: "dataset/stage1_train_region_text.json"
  val_split: 0.1          # 验证集比例
  image_size: 1024        # 图像尺寸（None表示使用原始尺寸）
  max_regions: 100        # 最大区域数量
```

#### 训练配置

```yaml
training:
  output_dir: "outputs/cervical_finetune"
  num_epochs: 5
  batch_size: 2                    # 根据GPU内存调整
  gradient_accumulation_steps: 4    # 有效batch_size = batch_size * gradient_accumulation_steps
  learning_rate: 2e-5               # 学习率，建议范围：1e-5 到 5e-5
  warmup_steps: 100
  save_steps: 500
  eval_steps: 500
  logging_steps: 50
```

#### 参数冻结配置

```yaml
freeze:
  vision_tower: true                # 冻结视觉编码器
  language_model_layers: [0,1,2,3,4,5]  # 冻结的语言模型层（可选）
```

**建议的冻结策略：**

1. **全参数微调**（需要大量GPU内存）：
   ```yaml
   freeze:
     vision_tower: false
     language_model_layers: []
   ```

2. **冻结视觉编码器**（推荐，平衡性能和资源）：
   ```yaml
   freeze:
     vision_tower: true
     language_model_layers: []
   ```

3. **冻结视觉编码器和部分语言模型层**（节省资源）：
   ```yaml
   freeze:
     vision_tower: true
     language_model_layers: [0, 1, 2, 3, 4, 5]
   ```

## 常见问题

### 1. 内存不足 (OOM)

**问题**: 训练时出现CUDA out of memory错误

**解决方案**:
- 减小 `batch_size`（例如从2改为1）
- 增加 `gradient_accumulation_steps` 以保持有效batch size
- 减小 `image_size`（例如从1024改为768）
- 启用梯度检查点（如果模型支持）
- 使用更激进的参数冻结策略

### 2. 训练速度慢

**问题**: 训练速度很慢

**解决方案**:
- 增加 `dataloader_num_workers`
- 使用混合精度训练（`mixed_precision: "bf16"`）
- 减小 `image_size`
- 减少 `max_regions`

### 3. 损失不下降

**问题**: 训练损失不下降或下降很慢

**解决方案**:
- 检查学习率是否合适（尝试调整 `learning_rate`）
- 检查数据是否正确加载
- 检查标签格式是否正确
- 尝试不冻结某些层
- 增加 `warmup_steps`

### 4. 评估时预测为空

**问题**: 模型预测结果为空

**解决方案**:
- 检查模型是否正确加载
- 检查输入格式是否正确
- 尝试降低生成时的 `temperature`
- 检查边界框proposals是否正确传递

### 5. 边界框格式错误

**问题**: 边界框坐标不正确

**解决方案**:
- 确保边界框格式为 `[x_min, y_min, x_max, y_max]`
- 检查坐标是否在图像范围内
- 使用 `utils/data_utils.py` 中的验证函数

## 使用示例

### 完整训练流程

```bash
# 1. 准备数据
# 确保 dataset/stage1_train_region_text.json 存在且格式正确

# 2. 配置训练参数
vim configs/train_config.yaml

# 3. 开始训练
python scripts/train_cervical.py --config configs/train_config.yaml

# 4. 评估模型
python scripts/eval_cervical.py \
    --model_path outputs/cervical_finetune \
    --data_json dataset/stage1_train_region_text.json \
    --output_dir eval_results

# 5. 查看结果
cat eval_results/evaluation_results.json
```

### 推理示例

训练完成后，可以使用微调后的模型进行推理：

```python
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes

# 加载模型
model_path = "outputs/cervical_finetune"
tokenizer, model, image_processors = load_pretrained_model(model_path)

# 准备输入
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
        {"type": "text", "text": "请在图像中定位阳性宫颈细胞区域。"}
    ],
    "bbox_list": [[x1, y1, x2, y2], ...]  # 边界框proposals
}]

# 生成预测
generation_kwargs = prepare_inputs(
    model_path=model_path,
    model=model,
    image_processors=image_processors,
    tokenizer=tokenizer,
    messages=messages,
    max_tokens=512
)

output_ids = model.generate(**generation_kwargs)
output_text = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:])

# 解析结果
pred_bboxes = extract_predictions_to_bboxes(output_text, messages[0]['bbox_list'])
print(f"预测结果: {pred_bboxes}")
```

## 性能优化建议

1. **数据增强**: 可以考虑添加图像增强（翻转、旋转等），但需要同步调整边界框
2. **学习率调度**: 使用cosine学习率调度器通常效果更好
3. **早停**: 监控验证集损失，如果不再下降可以提前停止
4. **模型集成**: 训练多个模型并集成可以提高性能
5. **后处理**: 对预测结果进行非极大值抑制（NMS）可以去除重复检测

## 参考资源

- [VLM-FO1 论文](https://arxiv.org/abs/2509.25916)
- [Qwen2.5-VL 文档](https://github.com/QwenLM/Qwen3-VL)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 许可证

请遵循VLM-FO1和Qwen2.5-VL的许可证要求。






















