# Gradio 演示

运行：

```bash
python gradio/app.py
```

功能：

- 聊天机器人形态，支持 `task_id=0,1,2,3,4,5`。
- `task_id=0,1`：定位任务，assistant 输出文本并在右侧返回 bbox 可视化图。
- `task_id=2,3,5`：在 `gr.Image` 上**单击两个对角点**得到轴对齐矩形 bbox（或填写 `x1 y1 x2 y2`），按数据集风格构造输入再推理。
- `task_id=4`：整图问答，不需要 ROI。

默认数据集路径是：

`/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json`

可在页面顶部修改。

## 模型路径（与 `scripts/eval_stage2_cot.py` 一致）

- **Base Model**：填 **Stage1 align** 的 LoRA 输出目录（例如 `outputs/stage1_align`；英文流程常用 `outputs/stage1_align_en`）。
- **Checkpoint**：填 **Stage2** 的 LoRA 目录（例如 `outputs/stage2_cot_balance` 或 `outputs/stage2_cot_balance_en`）。

内部调用与评估脚本相同的 `load_checkpoint_model`：当 Base **不是** 带 `vlm-fo1` 的官方目录时，会先加载默认基座 `resources/VLM-FO1_...`，再加载 Stage1 LoRA、`merge_and_unload()`，再叠 Stage2 LoRA。

**不要**把 `resources/VLM-FO1_...` 填在 Base 里作为「两阶段训练」的用法——该路径会走「官方权重 + 单层 LoRA」分支，等价于只训了 Stage2、未合并 Stage1。
