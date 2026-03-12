# VLM-FO1 宫颈任务创新点系统梳理

本文基于当前仓库代码实现，对你在 VLM-FO1 基础上的创新进行系统化归纳。  
重点覆盖：区域提案、HoVer-Net 病理特征、Path Feature Token 化注入、两阶段训练、多任务联合优化、可解释性与验证闭环。

---

## 1. 总体创新框架（端到端）

当前方案可抽象为 6 个串联环节：

1. **区域提案生成（上游检测器）**：为每张图像生成 `regions/bbox_list` 候选区域。  
2. **结构化病理特征抽取**：基于 HoVer-Net 输出（`json` + `mat`）和原图构建 ROI 级病理向量。  
3. **多模态序列构建**：将 `<regionX><regionfeat>` 注入文本上下文，使 LLM 可定位到候选 region。  
4. **Pathology Token 注入**：每个 ROI 形成 visual token + pathology token 并列输入 LLM。  
5. **两阶段训练**：阶段A做 Path Token 对齐，阶段B做多任务 LoRA 全流程学习。  
6. **评估/解释/对照实验**：多任务评估、attention 分析、特征-文本一致性验证、无辅助损失对照实验。

---

## 2. 创新点清单（按“是否代码落地”分层）

### A. 已在仓库代码明确落地

1. **ROI 级病理特征工程（HoVer-Net + Pap 染色适配）**  
   - 新增 9 维特征：实例形态 + 空间拥挤 + 核深染统计。  
   - 特征构造来自 HoVer-Net `json`（核 bbox/centroid/id）与 `mat`（`inst_map`），并融合原图亮度反转通道（`1 - max(RGB)`）适配巴氏染色。  
   - 代码：`scripts/build_path_features.py`。

2. **去重式预处理策略（按图像聚合 ROI）**  
   - `build_path_features` 先按 `image_path` 聚合，避免同图像在多任务样本重复提取，显著降低预处理冗余。  
   - 代码：`scripts/build_path_features.py` 中 `image_to_regions` 聚合逻辑。

3. **Path Feature 与 bbox 严格对齐机制**  
   - 数据集中对 bbox 做验证、过滤、resize、截断后，保持 `bbox_list_original` 同步过滤；再用 IoU 匹配缓存特征，确保 path feature 与最终 ROI 一一对应。  
   - 代码：`dataset/cervical_dataset.py`。

4. **Token-level Pathology Injection（由融合改并列 token）**  
   - 不再把病理特征直接融合到 visual feature；改为每个 ROI 生成并列 token：  
     - `visual_tokens`（来自 `mm_projector_aux`）  
     - `path_tokens`（`path_encoder -> mm_projector_path`）  
   - 使用 `path_type_embed` 显式区分 token 类型（0=visual, 1=pathology）。  
   - 采用 interleave 结构 `[ROI1_visual, ROI1_path, ROI2_visual, ROI2_path, ...]`。  
   - 代码：`vlm_fo1/model/language_model/omchat_qwen2_5_vl.py`。

5. **病理 token 模块懒初始化 + warmup 注册优化**  
   - `path_encoder/mm_projector_path/path_type_embed` 按输入维度懒初始化。  
   - 在训练脚本中增加 warmup forward，确保模块在 optimizer 构建前可见并参与训练。  
   - 代码：`omchat_qwen2_5_vl.py` + `scripts/train_cervical_stage2.py`。

6. **两阶段训练范式（Path 对齐 -> 多任务 LoRA）**  
   - 阶段A（`phase=paffe_align`）：冻结主干，仅训练 pathology token 相关模块。  
   - 阶段B（`phase=full`）：加载阶段A权重后进行 LoRA + 投影层/病理模块联合学习。  
   - 配套脚本与配置：  
     - `configs/train_config_stageA.yaml`  
     - `configs/train_config_stageB.yaml`  
     - `scripts/train_cervical_stage2.sh`  
     - `scripts/train_cervical_stage2_B.sh`

7. **多任务统一训练与任务感知辅助损失**  
   - 统一支持 task0~task4（定位、描述、分类）。  
   - 在阶段B引入两类辅助监督：  
     - `L_region`：对 `<regionX>` 命中 token 做加权监督（task0/1）  
     - `L_cls`：对类别词命中 token 做加权监督（task3/4）  
   - 支持全局权重 + 按 task 细粒度权重。  
   - 代码：`scripts/train_cervical_stage2.py`，配置在 `configs/train_config_stageB.yaml`。

8. **任务输出约束注入（降低多任务混淆）**  
   - 在数据管道内按 `task_id` 动态增强 user prompt：  
     - task2 强制“只输出描述文本，不要 `<ground>/<region*>`”  
     - task3/4 强制“只输出一个候选类别”  
   - 代码：`dataset/cervical_dataset.py`（评估脚本也有同策略）。

9. **训练可观测性与调试体系增强**  
   - 启动即检：检查 `task_id/assistant_text/path_features` 链路。  
   - 训练期诊断：打印每个窗口辅助损失均值、task 分布、mask 命中数、非空计数。  
   - 输入链路诊断：`InputDiag` 检查字段是否被 Trainer 链路裁掉。  
   - 代码：`scripts/train_cervical_stage2.py`。

10. **稳定性修复（多模态序列长度不一致）**  
    - 对辅助 CE 计算加入时序对齐（右对齐裁剪共同长度），解决 `logits` 与 `labels` 长度不一致导致的训练崩溃。  
    - 代码：`scripts/train_cervical_stage2.py`。

11. **可解释性与有效性闭环脚本**  
    - `scripts/validate_task4_path_features.py`：验证病理特征与文本描述关键词一致性（核深染/核增大/核不规则/核拥挤）。  
    - `scripts/analyze_token_attention.py`：重建真实 region token 位置，比较 visual/path token 注意力分布。  

12. **标准化对照实验入口**  
    - 新增“无辅助损失”实验脚本，通过生成临时配置将 region/cls 辅助项全部置零，输出到独立目录。  
    - 代码：`scripts/train_cervical_stage2_B_no_aux.sh`。

---

### B. 作为上游前置假设存在（接口已打通，训练代码不在当前仓）

1. **微调目标检测器生成 region proposals（你提到的 Grounding DINO 路线）**  
   - 当前训练/评估主流程假设每条样本已带 `regions[].bbox`，模型输出是对 proposal 索引的引用。  
   - 这一“候选区域先验”是体系关键创新之一。  
   - 在仓内可见的接口证据：  
     - `dataset/cervical_dataset.py` 使用 `item['regions']` 构造 `bbox_list`  
     - `vlm_fo1/mm_utils.py` 将每个 ROI 编码为 `<regionX><regionfeat>`  
     - `docs/evaluation_logic_comparison.md` 明确采用“索引匹配即坐标匹配”评估逻辑  
   - 注意：Grounding DINO 微调训练脚本本身不在此仓（当前可见的是 UPN/SAM3 proposal 推理示例脚本）。

---

## 3. 你这套方案相对原始 VLM-FO1 的核心增量

### 3.1 从“视觉主导”升级为“视觉+病理统计双通道”
- 原有 region token 主要依赖视觉编码。  
- 现在每个 ROI 注入结构化病理 token，LLM 可在 token 级别自主选择信息源。

### 3.2 从“单阶段微调”升级为“先对齐再任务学习”
- 阶段A先把病理 token 分布对齐到 LLM token 空间；  
- 阶段B再进行多任务 LoRA 优化，降低直接 joint training 的不稳定性。

### 3.3 从“黑箱训练”升级为“可诊断训练”
- 增加启动链路检查、辅助损失诊断、输入字段诊断、no-aux 对照脚本，训练可解释性显著增强。

### 3.4 从“功能验证”升级为“证据链验证”
- 有特征-文本一致性脚本（语义层）  
- 有 token attention 分析脚本（机制层）  
- 有多任务评估脚本（性能层）

---

## 4. 可用于论文/汇报的创新表述模板

可直接作为章节标题使用：

1. **Proposal-Referenced Multitask Cervical VLM Pipeline**  
2. **HoVer-Net-driven ROI Pathology Statistics for Pap-stained Cytology**  
3. **Token-level Pathology Injection with Type-aware Interleaving**  
4. **Two-stage Optimization: Pathology Token Alignment then LoRA Multitask Tuning**  
5. **Task-aware Auxiliary Token Supervision for Localization and Classification**  
6. **Closed-loop Validation via Feature-Text Consistency and Attention Attribution**

---

## 5. 当前建议的“创新点主次排序”

按“学术新颖性 + 工程落地价值”综合排序：

1. **Token-level Pathology Injection（并列 token + type embedding + interleave）**  
2. **两阶段训练机制（对齐阶段 + 多任务阶段）**  
3. **HoVer-Net ROI 病理特征体系（Pap 染色适配的 9 维统计）**  
4. **任务感知辅助损失（region/class token supervision）**  
5. **proposal 索引式训练评估范式（上游 detector + 下游索引引用）**  
6. **可解释性与验证闭环（attention + feature-text）**  

---

## 6. 代码证据索引（便于答辩/复现）

- 训练主流程：`scripts/train_cervical_stage2.py`  
- 数据与对齐：`dataset/cervical_dataset.py`  
- 病理特征构建：`scripts/build_path_features.py`  
- 模型注入实现：`vlm_fo1/model/language_model/omchat_qwen2_5_vl.py`  
- 两阶段配置：`configs/train_config_stageA.yaml`、`configs/train_config_stageB.yaml`  
- 训练脚本：`scripts/train_cervical_stage2.sh`、`scripts/train_cervical_stage2_B.sh`  
- no-aux 对照：`scripts/train_cervical_stage2_B_no_aux.sh`  
- 特征一致性验证：`scripts/validate_task4_path_features.py`  
- 注意力分析：`scripts/analyze_token_attention.py`  
- proposal 索引评估说明：`docs/evaluation_logic_comparison.md`

