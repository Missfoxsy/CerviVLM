# 评估脚本计算逻辑对比

## 概述

两个评估脚本都使用**基于region索引集合匹配**的方法来计算TP、FP、FN和定位指标。核心思想是：模型输出的是对proposal region的引用（索引），而不是直接生成坐标，因此可以直接通过索引集合的集合运算来计算指标。

## 核心逻辑

### 1. Region索引提取

两个脚本都从模型输出和GT中提取region索引：

```python
# 从模型输出解析
pred_dict = parse_grounding_output(output_text)
# 例如: {'阳性宫颈细胞': [0, 1, 9, 10, 11]}

# 从GT解析
gt_dict = parse_grounding_output(assistant_text)
# 例如: {'阳性宫颈细胞': [0, 1, 4, 9, 17]}

# 提取索引集合（忽略标签，只关注索引）
pred_indices_set = set()
for label, indices in pred_dict.items():
    pred_indices_set.update(indices)
# 例如: {0, 1, 9, 10, 11}

gt_indices_set = set()
for label, indices in gt_dict.items():
    gt_indices_set.update(indices)
# 例如: {0, 1, 4, 9, 17}
```

### 2. TP、FP、FN计算

两个脚本使用**完全相同的集合运算逻辑**：

```python
# TP (True Positives): 预测正确，即在预测集合和GT集合的交集中
tp = len(pred_indices_set & gt_indices_set)
# 例如: len({0, 1, 9, 10, 11} & {0, 1, 4, 9, 17}) = len({0, 1, 9}) = 3

# FP (False Positives): 预测错误，即在预测集合中但不在GT集合中
fp = len(pred_indices_set - gt_indices_set)
# 例如: len({0, 1, 9, 10, 11} - {0, 1, 4, 9, 17}) = len({10, 11}) = 2

# FN (False Negatives): 漏检，即在GT集合中但不在预测集合中
fn = len(gt_indices_set - pred_indices_set)
# 例如: len({0, 1, 4, 9, 17} - {0, 1, 9, 10, 11}) = len({4, 17}) = 2
```

**关键点**：
- 使用集合的交集(`&`)和差集(`-`)运算
- 直接比较region索引，不需要坐标匹配
- 因为模型引用的是相同的proposal region，所以索引匹配即表示坐标匹配

### 3. IoU计算

两个脚本的IoU计算逻辑**完全相同**：

```python
all_ious = []
for pred_idx in pred_indices_set:
    if pred_idx in gt_indices_set:
        all_ious.append(1.0)  # 正确预测，IoU为1.0（坐标完全相同）
    else:
        all_ious.append(0.0)  # 错误预测，IoU为0.0

mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
```

**关键点**：
- 如果预测的region索引在GT中，IoU = 1.0（因为引用的是同一个proposal region，坐标完全相同）
- 如果预测的region索引不在GT中，IoU = 0.0
- **不计算实际的bbox IoU**，因为模型只引用proposal，不生成新坐标

### 4. Precision、Recall、F1计算

两个脚本使用**标准公式**：

```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
```

## 两个脚本的差异

### eval_cervical.py (第一阶段)

**特点**：
- 单任务评估（只评估定位任务）
- 在`evaluate_model`函数中直接计算
- 所有样本的TP、FP、FN累加后计算总体指标

**代码位置**：
```189:205:scripts/eval_cervical.py
# 基于索引匹配计算TP, FP, FN
# pred_indices_set和gt_indices_set都是原始proposal region索引，直接比较
tp_batch = len(pred_indices_set & gt_indices_set)
fp_batch = len(pred_indices_set - gt_indices_set)
fn_batch = len(gt_indices_set - pred_indices_set)

tp += tp_batch
fp += fp_batch
fn += fn_batch

# 计算IoU
# 对于每个预测索引：如果在GT中，IoU为1.0（坐标完全相同）；否则IoU为0.0
for pred_idx in pred_indices_set:
    if pred_idx in gt_indices_set:
        all_ious.append(1.0)  # 正确预测，IoU为1.0
    else:
        all_ious.append(0.0)  # 错误预测，IoU为0.0
```

### eval_cervical_stage2.py (第二阶段)

**特点**：
- 多任务评估（支持4种任务类型）
- 有独立的`evaluate_localization_task`函数
- **按任务类型分别统计**（type1和type2分开计算指标）

**代码位置**：
```74:124:scripts/eval_cervical_stage2.py
def evaluate_localization_task(pred_dict: dict, gt_dict: dict, gt_regions: list) -> dict:
    """
    评估定位任务（类型1和类型2）
    """
    # 提取region索引
    pred_indices_set = set()
    for label, indices in pred_dict.items():
        pred_indices_set.update(indices)
    
    gt_indices_set = set()
    for label, indices in gt_dict.items():
        gt_indices_set.update(indices)
    
    # 计算TP, FP, FN
    tp = len(pred_indices_set & gt_indices_set)
    fp = len(pred_indices_set - gt_indices_set)
    fn = len(gt_indices_set - pred_indices_set)
    
    # 计算IoU（对于定位任务，正确预测的region IoU为1.0）
    all_ious = []
    for pred_idx in pred_indices_set:
        if pred_idx in gt_indices_set:
            all_ious.append(1.0)
        else:
            all_ious.append(0.0)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'pred_indices': sorted(pred_indices_set),
        'gt_indices': sorted(gt_indices_set)
    }
```

**关键差异**：
- `eval_cervical_stage2.py`按任务类型分别累加：
```python
# 在evaluate_model中
if task_type in ['type1', 'type2']:
    metrics = evaluate_localization_task(pred_dict, gt_dict, gt_regions)
    task_metrics[task_type]['tp'] += metrics['tp']  # 按类型累加
    task_metrics[task_type]['fp'] += metrics['fp']
    task_metrics[task_type]['fn'] += metrics['fn']
    task_metrics[task_type]['all_ious'].extend(...)
```

- 最后分别计算type1和type2的指标：
```python
for task_type in ['type1', 'type2']:
    tp = task_metrics[task_type]['tp']
    fp = task_metrics[task_type]['fp']
    fn = task_metrics[task_type]['fn']
    # 计算该任务类型的precision, recall, f1, mean_iou
```

## 计算逻辑总结

### 相同点

1. **索引提取方式相同**：都使用`parse_grounding_output`解析，然后提取索引集合
2. **TP/FP/FN计算逻辑相同**：都使用集合运算
3. **IoU计算逻辑相同**：正确预测=1.0，错误预测=0.0
4. **指标计算公式相同**：Precision、Recall、F1使用标准公式

### 不同点

1. **代码组织**：
   - `eval_cervical.py`：在主函数中直接计算
   - `eval_cervical_stage2.py`：封装在`evaluate_localization_task`函数中

2. **任务类型处理**：
   - `eval_cervical.py`：单任务，所有样本一起计算
   - `eval_cervical_stage2.py`：多任务，type1和type2分别计算指标

3. **指标输出**：
   - `eval_cervical.py`：输出总体指标
   - `eval_cervical_stage2.py`：输出各任务类型的独立指标

## 示例计算

假设：
- 预测索引集合：`{0, 1, 9, 10, 11}`
- GT索引集合：`{0, 1, 4, 9, 17}`

计算过程：
```
TP = len({0, 1, 9, 10, 11} & {0, 1, 4, 9, 17}) = len({0, 1, 9}) = 3
FP = len({0, 1, 9, 10, 11} - {0, 1, 4, 9, 17}) = len({10, 11}) = 2
FN = len({0, 1, 4, 9, 17} - {0, 1, 9, 10, 11}) = len({4, 17}) = 2

Precision = 3 / (3 + 2) = 0.6
Recall = 3 / (3 + 2) = 0.6
F1 = 2 * 0.6 * 0.6 / (0.6 + 0.6) = 0.6

IoU列表: [1.0, 1.0, 1.0, 0.0, 0.0]  # 对应索引0, 1, 9, 10, 11
Mean IoU = (1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 5 = 0.6
```

## 注意事项

1. **IoU的特殊性**：由于模型只引用proposal region，不生成新坐标，所以IoU只能是1.0或0.0，不能反映bbox重叠程度
2. **索引匹配即坐标匹配**：因为所有region都来自同一个proposal列表，索引匹配意味着引用的是同一个bbox
3. **集合运算的高效性**：使用Python的set操作，时间复杂度为O(n)，比逐一遍历更高效





















