# 第二阶段训练和评估指南

## 概述

第二阶段训练数据集包含四种任务类型：
- **类型1（10%）**：基础定位(OD) - 找出图片中的阳性宫颈细胞
- **类型2（30%）**：带描述的特征定位（REC）- 找出具有特定特征的阳性细胞
- **类型3（30%）**：区域描述(Region Caption) - 描述指定区域的特征
- **类型4（30%）**：层级分类(Classification) - 判断区域属于哪个细胞层级

## 1. 是否需要构建多任务测试集？

**答案：是的，强烈建议构建包含四种任务类型的测试集。**

### 原因：
1. **全面评估模型性能**：只有包含所有任务类型的测试集才能全面评估模型在第二阶段的多任务学习效果
2. **发现任务间的相互影响**：多任务学习可能带来任务间的相互促进或干扰，需要分别评估
3. **指导模型优化**：通过各任务的独立指标，可以识别哪些任务需要更多训练数据或调整

### 测试集构建建议：
- 从训练数据集中按比例随机采样10-20%作为测试集
- 保持四种任务类型的比例与训练集一致（或接近）
- 确保测试集与训练集不重叠

## 2. 训练策略是否需要调整？

### 当前策略分析：

**当前训练策略（与第一阶段相同）是合理的**，原因如下：

#### ✅ 优点：
1. **统一损失函数**：所有任务都使用语言模型的交叉熵损失，模型会自动学习不同任务的输出格式
2. **简单高效**：不需要复杂的多任务损失权重调整
3. **端到端训练**：模型可以学习任务间的共享表示

#### ⚠️ 可能的改进：

如果需要进一步优化，可以考虑：

1. **任务权重平衡**（可选）：
   - 如果某些任务表现较差，可以增加该任务的采样频率
   - 或在损失函数中为不同任务设置权重

2. **评估策略调整**：
   - 在训练过程中分别监控四种任务的指标
   - 使用多任务综合指标选择最佳checkpoint

3. **学习率调整**（当前已采用）：
   - 第二阶段使用更小的学习率（0.00005 vs 0.0001），这是合理的
   - 因为模型已经学习了基础定位能力，需要更精细的调整

### 建议：
**保持当前训练策略不变**，因为：
- 多任务学习通常不需要复杂的损失权重调整
- 模型会自动学习任务间的平衡
- 当前策略已经足够简单有效

## 3. 评估脚本使用

### 多任务评估脚本：`scripts/eval_cervical_stage2.py`

该脚本支持四种任务类型的评估：

#### 使用方法：

```bash
python scripts/eval_cervical_stage2.py \
    --base_model resources/VLM-FO1_Qwen2.5-VL-3B-v01 \
    --checkpoint outputs/cervical_finetune_stage2 \
    --test_json dataset/stage2_test.json \
    --output_dir eval_results_stage2 \
    --device cuda
```

#### 评估指标：

1. **类型1和类型2（定位任务）**：
   - Precision, Recall, F1
   - Mean IoU
   - TP, FP, FN

2. **类型3（区域描述）**：
   - Word Overlap（词汇重叠率）
   - Phrase Recall（关键短语召回率）

3. **类型4（层级分类）**：
   - Accuracy（准确率）

#### 输出文件：
- `results.json`：每个样本的详细结果
- `metrics.json`：各任务的汇总指标
- `visualizations/`：定位任务的可视化结果

## 4. 测试集构建脚本示例

如果需要从训练集中构建测试集，可以使用以下脚本：

```python
import json
import random
from collections import defaultdict

def classify_task_type(user_text, assistant_text):
    """分类任务类型"""
    if '<ground>' in assistant_text and '<objects>' in assistant_text:
        if '找出' in user_text and ('核' in user_text or '浆' in user_text or '挖空' in user_text or '异常' in user_text):
            return 'type2'
        else:
            return 'type1'
    elif '描述图中这个区域' in user_text:
        return 'type3'
    elif '属于哪个细胞层级' in user_text:
        return 'type4'
    return 'unknown'

# 加载训练集
with open('dataset/stage2_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 按任务类型分组
task_groups = defaultdict(list)
for item in train_data:
    user_text = ''
    assistant_text = ''
    for conv in item['conversations']:
        if conv['from'] == 'user':
            user_text = conv['value']
        elif conv['from'] == 'assistant':
            assistant_text = conv['value']
    
    task_type = classify_task_type(user_text, assistant_text)
    task_groups[task_type].append(item)

# 从每个任务类型中采样20%作为测试集
test_data = []
train_data_filtered = []

random.seed(42)
for task_type, items in task_groups.items():
    random.shuffle(items)
    test_size = int(len(items) * 0.2)
    test_data.extend(items[:test_size])
    train_data_filtered.extend(items[test_size:])

# 保存测试集和新的训练集
with open('dataset/stage2_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

with open('dataset/stage2_train_filtered.json', 'w', encoding='utf-8') as f:
    json.dump(train_data_filtered, f, ensure_ascii=False, indent=2)

print(f"测试集大小: {len(test_data)}")
print(f"新训练集大小: {len(train_data_filtered)}")
```

## 5. 训练配置建议

当前配置（`configs/train_config_stage2.yaml`）已经合理：

- ✅ 学习率：0.00005（第一阶段的一半）
- ✅ 训练轮数：3（比第一阶段少）
- ✅ 评估频率：每100步（更频繁的评估）
- ✅ LoRA配置：保持与第一阶段相同

## 总结

1. **必须构建多任务测试集**：包含四种任务类型，比例与训练集一致
2. **训练策略保持不变**：当前策略已经合理，无需复杂调整
3. **使用多任务评估脚本**：`eval_cervical_stage2.py` 可以全面评估所有任务
4. **监控各任务指标**：分别关注四种任务的表现，识别需要改进的任务

