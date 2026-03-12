# 第二阶段多任务推理脚本使用指南

## 脚本位置
`inference_cervical_stage2.py`

## 功能特点

- ✅ 支持四种任务类型的推理
- ✅ 自动识别任务类型
- ✅ 支持按任务类型筛选样本
- ✅ 自动可视化定位任务结果
- ✅ 支持LoRA权重合并（可选）

## 使用方法

### 1. 测试所有任务类型（每种类型一个样本）

```bash
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type all \
    --output_dir inference_results_stage2
```

### 2. 测试特定任务类型

```bash
# 测试类型1（基础定位）
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type type1 \
    --num_samples 3 \
    --output_dir inference_results_stage2

# 测试类型2（带描述的特征定位）
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type type2 \
    --num_samples 3 \
    --output_dir inference_results_stage2

# 测试类型3（区域描述）
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type type3 \
    --num_samples 3 \
    --output_dir inference_results_stage2

# 测试类型4（层级分类）
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type type4 \
    --num_samples 3 \
    --output_dir inference_results_stage2
```

### 3. 测试指定索引的样本

```bash
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --sample_idx 100 \
    --output_dir inference_results_stage2
```

### 4. 合并LoRA权重（加速推理）

```bash
python inference_cervical_stage2.py \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --task_type all \
    --merge_lora \
    --output_dir inference_results_stage2
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | `resources/VLM-FO1_Qwen2.5-VL-3B-v01` | 基础模型路径 |
| `--checkpoint` | `outputs/cervical_finetune_stage2` | Checkpoint路径 |
| `--test_json` | `dataset/stage2_test.json` | 测试集JSON文件 |
| `--sample_idx` | `None` | 指定样本索引（优先级高于task_type） |
| `--task_type` | `all` | 任务类型：`type1`, `type2`, `type3`, `type4`, `all` |
| `--num_samples` | `1` | 每个任务类型的样本数量 |
| `--output_dir` | `inference_results_stage2` | 输出目录 |
| `--device` | `cuda` | 设备（cuda/cpu） |
| `--merge_lora` | `False` | 是否合并LoRA权重 |

## 输出文件

### 定位任务（类型1和类型2）

- `pred_type1_<sample_id>.jpg` - 预测结果可视化（红色框）
- `gt_type1_<sample_id>.jpg` - 真实标签可视化（绿色框）
- `result_type1_<sample_id>.json` - 详细结果JSON

### 文本生成任务（类型3和类型4）

- `result_type3_<sample_id>.json` - 详细结果JSON（包含预测文本和真实文本）

### 汇总文件

- `summary.json` - 所有样本的结果汇总

## 任务类型说明

- **type1（基础定位）**：找出图片中的阳性宫颈细胞
- **type2（带描述的特征定位）**：找出具有特定特征的阳性细胞（如核深染、核增大等）
- **type3（区域描述）**：描述指定区域的特征
- **type4（层级分类）**：判断区域属于哪个细胞层级

## 示例输出

```
Loading test data from: dataset/stage2_test.json
Total test samples: 1240

Selected 4 sample(s) for inference

Loading checkpoint model from: outputs/cervical_finetune_stage2
Loading base model from: resources/VLM-FO1_Qwen2.5-VL-3B-v01
Loading LoRA weights from: outputs/cervical_finetune_stage2
LoRA weights loaded successfully

============================================================
Sample 123: cervical_00001234
Task Type: type1
Image path: /path/to/image.jpg
User text: 请指出图像中阳性宫颈细胞的具体位置
============================================================
Number of regions: 5
Running inference...

Model output: <ground>阳性宫颈细胞</ground><objects><region0><region2></objects>
Ground truth: <ground>阳性宫颈细胞</ground><objects><region0><region2><region4></objects>

Extracting bounding boxes...
Predicted bounding boxes (dict with 1 labels):
  Label '阳性宫颈细胞': 2 bboxes
Prediction visualization saved to: inference_results_stage2/pred_type1_cervical_00001234.jpg
Ground truth visualization saved to: inference_results_stage2/gt_type1_cervical_00001234.jpg
Result saved to: inference_results_stage2/result_type1_cervical_00001234.json
...
```

