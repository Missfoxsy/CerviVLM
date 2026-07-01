# 基于 VLM-FO1 的宫颈细胞两阶段训练：改进点、模型框架与数据流

本文档依据仓库内 `scripts/run_two_stage_and_eval.sh`、`scripts/train_stage1_align.py`、`scripts/train_stage2_cot.py`、`vlm_fo1/model/language_model/omchat_qwen2_5_vl.py`、`vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py` 及 `dataset/cervical_dataset.py` 梳理，用于说明**在本项目中对官方 VLM-FO1（Qwen2.5-VL 多模态）的扩展与数据管线**。

---

## 1. 基座与项目定位

- **基座模型**：`resources/VLM-FO1_Qwen2.5-VL-3B-v01`（OmChat 封装下的 Qwen2.5-VL 视觉-语言模型）。
- **应用场景**：宫颈细胞相关多任务（定位、区域描述、分类、CoT 诊断式输出等），通过**两阶段训练**先对齐区域与领域表达，再注入 Chain-of-Thought 风格推理能力。
- **与通用 VLM-FO1 的差异**：在保持主干（ViT 主塔 + 辅助塔 + 区域编码 HFRE + 投影到 LLM）的前提下，增加了**宫颈 JSON 数据管线**、**任务约束 prompt**、**层级 ROI 特征融合（含门控 gated）**及**两阶段 LoRA 策略**。

---

## 2. 相对官方流程的主要改进点

### 2.1 两阶段训练范式

| 阶段 | 脚本 | 配置示例 | 目标 |
|------|------|----------|------|
| Stage1（对齐） | `train_stage1_align.py` | `configs/train_config_stage1_lora.yaml` | 领域适配：定位/描述等，**LoRA + mm_projector / mm_projector_aux**（可训投影与区域路径） |
| Stage2（CoT） | `train_stage2_cot.py` | `configs/train_config_stageB_lora.yaml` | 在 Stage1 权重上注入 CoT：**仅 LoRA 作用于 q/k/v/o**，冻结视觉与 projector，减轻遗忘 |

Stage2 支持从 Stage1 **LoRA 目录**加载（`load_stageA_lora: true`），在 CPU 上 merge 后再训练，降低显存峰值。

### 2.2 HFRE 层级特征融合（Hierarchical Feature Fusion）

- **位置**：辅助视觉塔 + SimpleFPN 输出的多尺度特征上，对每个 ROI 做多层 `roi_align` 池化后，进入 `HierarchicalFeatureFusion`（`hybrid_finegrained_region_encoder.py`）。
- **结构**：粗→中→细三层（4 层 FPN 时：前两档均值作为粗层，后两档为中/细）；每层线性编码 + 相邻层 **MultiheadAttention**（query 为当前层，key/value 为上一层），再融合为 **2048 维**以兼容原 VT 区域分支。
- **融合模式**（`hierarchical_fusion_mode`）：
  - `legacy`：三层拼接后线性投影。
  - `task_adaptive`：按 `task_type`（localization / classification / reasoning）走不同头或层级选择器。
  - `gated`：**门控网络**对三层特征做 **Softmax 归一化权重**，逐层加权后拼接再投影；缓存 `last_gates` 便于可解释性分析。

启用条件（训练脚本中）：`use_hierarchical_fusion: true` 且 **`mm_use_simpleFPN_for_vt=true`**（否则脚本会跳过挂载）。

### 2.3 任务类型与 `task_id` 驱动区域编码

在 `encode_regions` 中，根据 `task_ids` 将样本映射为 `task_type`，供 HFRE 使用：

- `0, 1` → `localization`
- `3, 4` → `classification`
- 其余 → `reasoning`

训练时 `task_ids` 由 `CervicalDataset` 经 collator 传入模型前向。

### 2.4 数据侧：任务约束与多语言

`dataset/cervical_dataset.py` 在**不修改原始 JSON** 的前提下，按 `task_id` 为用户 prompt 追加**输出格式约束**（中/英 `TASK_CONSTRAINT_TEXTS`），减轻多任务混淆（如 type2 禁止 ground 模板、type3/4 只输出类别名、type5 三段式结构）。

### 2.5 评估与可视化

- `scripts/eval_stage2_cot.py`：多任务指标 + 可选层级注意力热力图导出。
- 门控模式下可汇总 **ROI 级门控 → 样本内平均 → 按任务/细胞类型聚合**（见评估脚本中的 gate 统计逻辑与 `gate_stats.json`）。

### 2.6 一键流水线

`scripts/run_two_stage_and_eval.sh`：

- `torchrun` 依次调用 Stage1 / Stage2 训练；
- 评估阶段调用 `eval_stage2_cot.py`，可通过环境变量切换 `START_FROM`、`CUDA_VISIBLE_DEVICES`、`HIERARCHICAL_FUSION_MODE`（如 `gated`）、评估 JSON 与输出目录等。

---

## 3. 模型框架（简化）

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：图像 + 文本 token +（可选）区域 bbox_list + task_ids      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────────┐
│ Qwen2.5-VL ViT     │     │ Vision Tower Aux    │
│ 主塔 encode_images │     │ 多尺度特征           │
│ → mm_projector     │     │ + HFRE (ROIAlign +  │
│ → 图像 token 嵌入   │     │  HierarchicalFusion) │
└────────────────────┘     │ → mm_projector_aux   │
         │                 └──────────┬───────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
         prepare_inputs_labels_for_qwen2_5_vl_multimodal
         （将图像/区域特征与文本 token 对齐，构造 input_embeds）
                      ▼
              Qwen2.5 LLM（+ Stage1/2 LoRA）
                      ▼
                 自回归 LM loss / generate
```

- **OmChatMetaModel**（`omchat_arch.py`）：组装 `vision_tower`、`vision_tower_aux`、`mm_projector`、`mm_projector_aux`、`object_vp_extractor`（HFREModule）。
- **OmChatQwen25VLForCausalLM**：实现 `encode_images`、`encode_regions` 及多模态 `prepare_inputs_labels_for_qwen2_5_vl_multimodal`。

---

## 4. 数据流程（训练）

```
JSON（image, conversations, regions, task_id）
        │
        ▼
CervicalDataset.__getitem__
  · 解析 user/assistant，按 task_id 追加任务约束文案
  · 读图、resize、bbox 校验与对齐辅助分支尺寸
  · tokenizer 构造 input_ids / labels（仅 assistant 段计 loss）
  · 输出：input_ids, labels, images, image_grid_thws,
          images_aux, bbox_list, task_id, assistant_text
        │
        ▼
DataCollator（Stage1 右填充；Stage2 左填充 + FlashAttention 兼容）
  · batch：input_ids, attention_mask, labels, images,
            image_grid_thws, images_aux, bbox_list, task_ids, ...
        │
        ▼
CervicalTrainerStage1.compute_loss
  · pop 出 task_ids 等，传入 model(**model_inputs)
        │
        ▼
模型内部：encode_images + encode_regions(task_ids)
  · 区域特征经 HFRE（含 gated 时写 last_gates）
  · 与文本嵌入融合后算交叉熵（标准 LM loss）
```

**注意**：若从 shell 直接 `python` 且未设置 `PYTHONPATH`，可能无法 `import dataset`；建议在项目根执行或 `export PYTHONPATH=<项目根>:$PYTHONPATH`。

---

## 5. 配置与脚本对应关系（当前常用）

| 项目 | 说明 |
|------|------|
| `run_two_stage_and_eval.sh` | `STAGE1_CONFIG` / `STAGE2_CONFIG`，`START_FROM=stage1\|stage2\|eval` |
| Stage1 | `use_hierarchical_fusion` + `hierarchical_fusion_mode: gated`，LoRA 含 `q/k/v/o/gate/up/down` + projector |
| Stage2 | 同上层级融合；**仅 lora_layers**；`base_model` 指向 Stage1 输出目录 |
| 评估 | `eval_stage2_cot.py`：`--use_hierarchical_fusion`、`--hierarchical_fusion_mode gated` |

---

## 6. 任务 ID 与数据阶段对照（与脚本注释一致）

**Stage1（对齐）**

- task 0：定位（ground/objects/region）
- task 1：按描述定位
- task 2：指定 region 病理描述

**Stage2（CoT 多任务，平衡集等）**

- task 0 / 1 / 2：同能力扩展
- task 4：来源分类等
- task 5：区域细胞类型 + 诊断依据（CoT 结构）

具体样本比例以所用 JSON 为准。

---

## 7. 文档维护

- 代码为准：若配置或脚本与本文不一致，以仓库内 `configs/*.yaml` 与 `scripts/*.py` 为准。
- 修订时可同步更新：层级融合模式、评估 `task_ids`、默认 checkpoint 路径。

---

*生成说明：本文档由代码阅读整理，用于组内对齐「VLM-FO1 宫颈扩展」的框架与数据流。*
