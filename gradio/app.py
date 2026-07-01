#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_stage2_cot import (
    DEFAULT_VLM_FO1_BASE,
    _is_official_vlm_fo1_model_dir,
    add_task_constraints_to_user_text,
    load_checkpoint_model,
)
from utils.data_utils import parse_grounding_output
from vlm_fo1.mm_utils import extract_predictions_to_bboxes, prepare_inputs


def _normalize_grounding_markup(text: str) -> str:
    """合并标签之间的空白/换行，便于正则解析（模型常在 </ground> 与 <objects> 之间换行）。"""
    s = text.replace("<object>", "<objects>").replace("</object>", "</objects>")
    return re.sub(r">\s*<", "><", s.strip())


def _model_output_to_pred_bboxes(output_text: str, bbox_list: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    从定位任务模型输出解析 <ground>...<objects><region*>，映射到 bbox_list 下标（与 mm_utils 中 region0.. 一致）。
    优先 parse_grounding_output（支持 <object>/<objects>），并处理跨行标签。
    """
    if not bbox_list or not (output_text or "").strip():
        return {}

    fixed = _normalize_grounding_markup(output_text)

    out: Dict[str, List[List[float]]] = {}
    gd = parse_grounding_output(fixed)
    for label, indices in gd.items():
        seen = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(bbox_list):
                seen.append(bbox_list[i])
        if seen:
            out[label.strip() or "target"] = seen

    if out:
        return out

    out = extract_predictions_to_bboxes(fixed, bbox_list)
    if out:
        return out

    # 兜底：仅列出 <regionN>（无完整 ground 块时）
    idxs = sorted({int(m) for m in re.findall(r"<region(\d+)>", fixed, flags=re.IGNORECASE)})
    boxes = [bbox_list[i] for i in idxs if 0 <= i < len(bbox_list)]
    if boxes:
        out["detection"] = boxes
    return out


@dataclass
class RuntimeState:
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    image_processors: Optional[object] = None
    model_name_for_template: Optional[str] = None
    base_model: str = ""
    checkpoint: str = ""
    device: str = "cuda"
    language: str = "zh"

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.image_processors is not None


def _load_dataset(dataset_path: str) -> List[dict]:
    if not dataset_path or not os.path.exists(dataset_path):
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _filter_indices_by_task(data: List[dict], task_id: int) -> List[int]:
    out = []
    for i, item in enumerate(data):
        try:
            if int(item.get("task_id", -999)) == int(task_id):
                out.append(i)
        except Exception:
            continue
    return out


def _extract_user_and_gt(item: dict) -> Tuple[str, str]:
    user_text, gt_text = "", ""
    for conv in item.get("conversations", []):
        if conv.get("from") == "user":
            user_text = conv.get("value", "").replace("<image>", "").strip()
        elif conv.get("from") == "assistant":
            gt_text = conv.get("value", "")
    return user_text, gt_text


def _draw_box(img: Image.Image, bbox: List[float], color: str = "red", width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def _draw_pred_bboxes(img: Image.Image, pred_dict: Dict[str, List[List[float]]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for label_name, boxes in pred_dict.items():
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 14)), str(label_name), fill="red")
    return out


def _parse_manual_bbox(s: str) -> Optional[List[int]]:
    """解析 "x1 y1 x2 y2" 或逗号分隔的四个数，返回 [x1,y1,x2,y2]。"""
    if not s or not str(s).strip():
        return None
    parts = re.split(r"[\s,;]+", str(s).strip())
    nums: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            nums.append(float(p))
        except ValueError:
            return None
        if len(nums) == 4:
            break
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _clip_bbox_to_image(bbox: List[int], size: Tuple[int, int]) -> List[int]:
    w, h = size
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([int(x1), int(x2)])
    y1, y2 = sorted([int(y1), int(y2)])
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def _parse_select_xy(evt: gr.SelectData) -> Optional[Tuple[int, int]]:
    """从 Image.select 的 SelectData 取 (x, y)，兼容不同 Gradio 版本。"""
    idx = getattr(evt, "index", None)
    if idx is None:
        return None
    if isinstance(idx, dict):
        x = idx.get("x", idx.get(0))
        y = idx.get("y", idx.get(1))
        if x is None or y is None:
            return None
        return int(x), int(y)
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        if idx[0] is None or idx[1] is None:
            return None
        return int(idx[0]), int(idx[1])
    return None


def _clean_assistant_text(text: str) -> str:
    """去掉聊天展示不需要的特殊标记（ChatML / Qwen-VL 等解码残留）。"""
    if not text:
        return ""
    s = str(text).strip()
    # 所有 <|...|> 片段
    s = re.sub(r"<\|[^|]+\|>", "", s)
    # 个别 tokenizer 会解出无尖括号的尾标
    for tok in ("<|im_end|>", "<|endoftext|>"):
        s = s.replace(tok, "")
    return s.strip()


def _build_message(image_path: str, user_text: str, bbox_list: List[List[int]]) -> List[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": user_text},
        ],
        "bbox_list": bbox_list,
    }]


def _ensure_runtime(
    runtime: RuntimeState,
    base_model: str,
    checkpoint: str,
    device: str,
    language: str,
) -> Tuple[RuntimeState, str]:
    if (
        runtime.ready
        and runtime.base_model == base_model
        and runtime.checkpoint == checkpoint
        and runtime.device == device
        and runtime.language == language
    ):
        return runtime, "模型已加载，可直接推理。"

    tokenizer, model, image_processors = load_checkpoint_model(base_model, checkpoint, device=device)
    runtime.tokenizer = tokenizer
    runtime.model = model
    runtime.image_processors = image_processors
    # 与 eval_stage2_cot.py main() 一致：仅当 base 为官方发布目录（最后一级名含 vlm-fo1）时用其作 chat 模板名
    bm = (base_model or "").strip()
    runtime.model_name_for_template = bm if bm and _is_official_vlm_fo1_model_dir(bm) else DEFAULT_VLM_FO1_BASE
    runtime.base_model = base_model
    runtime.checkpoint = checkpoint
    runtime.device = device
    runtime.language = language
    return runtime, "模型加载完成。"


def _build_demo_item_for_roi(item: dict, user_text: str, roi_bboxes: List[List[int]]) -> dict:
    first_bbox = roi_bboxes[0]
    # 构造与数据集相同风格的最小样本结构
    return {
        "id": item.get("id", "demo_item"),
        "task_id": item.get("task_id"),
        "image": item.get("image"),
        "regions": [{"bbox": first_bbox, "region_id": "region0"}],
        "conversations": [
            {"from": "user", "value": f"<image>\n{user_text}"},
            {"from": "assistant", "value": ""},
        ],
    }


def _run_model_once(
    runtime: RuntimeState,
    image_path: str,
    task_id: int,
    user_text: str,
    bbox_list: List[List[int]],
) -> str:
    prompt_text = add_task_constraints_to_user_text(user_text, task_id, language=runtime.language)
    messages = _build_message(image_path, prompt_text, bbox_list)

    gen_kwargs = prepare_inputs(
        model_name=runtime.model_name_for_template,
        model=runtime.model,
        image_processors=runtime.image_processors,
        tokenizer=runtime.tokenizer,
        messages=messages,
        device=runtime.device,
        max_tokens=512,
        temperature=0.0,
        do_sample=False,
        path_feature_root=None,
    )
    output_ids = runtime.model.generate(**gen_kwargs)
    # 必须与 scripts/eval_stage2_cot.py 一致：False，否则 <region0> 等会被当作 special token 吃掉，无法解析画框
    return runtime.tokenizer.decode(
        output_ids[0, gen_kwargs["inputs"].shape[1]:],
        skip_special_tokens=False,
    ).strip()


def on_load_sample(dataset_path: str, task_id: int, sample_rank: int):
    data = _load_dataset(dataset_path)
    if not data:
        return None, None, "未加载到数据，请检查 dataset 路径。", gr.update(choices=[], value=None), {"clicks": [], "bbox": None}

    indices = _filter_indices_by_task(data, task_id)
    if not indices:
        return None, None, f"task_id={task_id} 没有样本。", gr.update(choices=[], value=None), {"clicks": [], "bbox": None}

    rank = max(0, min(int(sample_rank) if sample_rank is not None else 0, len(indices) - 1))
    idx = indices[rank]
    item = data[idx]
    user_text, gt_text = _extract_user_and_gt(item)
    img = Image.open(item["image"]).convert("RGB")
    info = (
        f"已载入样本: idx={idx}, id={item.get('id', 'N/A')}, task_id={item.get('task_id')}\n"
        f"原问题: {user_text}\nGT: {gt_text[:180]}"
    )
    return img, item, info, gr.update(choices=list(range(len(indices))), value=rank), {"clicks": [], "bbox": None}


def on_roi_image_select(evt: gr.SelectData, sample_state: Optional[dict], roi_state: Optional[dict]):
    """在图像上连续点击两次，得到轴对齐矩形 bbox（与数据集 [x1,y1,x2,y2] 一致）。"""
    roi_state = dict(roi_state or {})
    roi_state.setdefault("clicks", [])
    roi_state.setdefault("bbox", None)

    if sample_state is None or not sample_state.get("image"):
        return gr.update(), roi_state

    xy = _parse_select_xy(evt)
    if xy is None:
        return gr.update(), roi_state

    base = Image.open(sample_state["image"]).convert("RGB")
    w, h = base.size
    x, y = xy[0], xy[1]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    clicks = list(roi_state["clicks"]) + [[x, y]]
    if len(clicks) < 2:
        roi_state = {"clicks": clicks, "bbox": None}
        preview = base.copy()
        dr = ImageDraw.Draw(preview)
        r = 5
        dr.ellipse([x - r, y - r, x + r, y + r], outline="#ff6a00", width=3)
        dr.text((x + 8, y - 8), "再点击对角点", fill="#ff6a00")
        return preview, roi_state

    (x1, y1), (x2, y2) = clicks[-2], clicks[-1]
    bbox = _clip_bbox_to_image([x1, y1, x2, y2], (w, h))
    if not bbox:
        roi_state = {"clicks": [], "bbox": None}
        return base, roi_state

    roi_state = {"clicks": [], "bbox": bbox}
    return _draw_box(base, bbox, color="#ff6a00", width=4), roi_state


def on_clear_roi_selection(sample_state: Optional[dict]):
    if sample_state is None or not sample_state.get("image"):
        return None, {"clicks": [], "bbox": None}
    img = Image.open(sample_state["image"]).convert("RGB")
    return img, {"clicks": [], "bbox": None}


def on_change_rank(dataset_path: str, task_id: int, sample_rank: int):
    return on_load_sample(dataset_path, task_id, sample_rank)


def on_send(
    chat_history: List[Tuple[str, str]],
    runtime: RuntimeState,
    base_model: str,
    checkpoint: str,
    device: str,
    language: str,
    item: dict,
    task_id: int,
    user_input: str,
    roi_state: Optional[dict],
    manual_bbox: str,
):
    chat_history = chat_history or []
    if item is None:
        chat_history.append(("", "请先加载一个样本。"))
        return chat_history, runtime, None

    runtime, _ = _ensure_runtime(runtime, base_model, checkpoint, device, language)
    image_path = item["image"]
    vis_img = None

    if task_id in (0, 1):
        # 定位任务：使用样本自带 region 池，输出框图
        regions = item.get("regions", [])
        bbox_list = [r["bbox"] for r in regions if isinstance(r.get("bbox"), list) and len(r.get("bbox")) == 4]
        output_text = _run_model_once(runtime, image_path, task_id, user_input, bbox_list)
        pred = _model_output_to_pred_bboxes(output_text, bbox_list)
        img = Image.open(image_path).convert("RGB")
        vis_img = _draw_pred_bboxes(img, pred) if pred else img
        reply = _clean_assistant_text(output_text)

    elif task_id in (2, 3, 5):
        # ROI：优先双击矩形；否则用手动坐标
        roi_boxes: List[List[int]] = []
        base = Image.open(image_path).convert("RGB")
        if roi_state and roi_state.get("bbox"):
            clipped = _clip_bbox_to_image(list(roi_state["bbox"]), base.size)
            if clipped:
                roi_boxes = [clipped]
        if not roi_boxes:
            mb = _parse_manual_bbox(manual_bbox or "")
            if mb:
                clipped = _clip_bbox_to_image(mb, base.size)
                if clipped:
                    roi_boxes = [clipped]
        if not roi_boxes:
            chat_history.append(
                ("", "请在左侧图像上**连续点击两个点**（矩形对角），或在下方填写 **x1 y1 x2 y2** 后再发送。")
            )
            return chat_history, runtime, None
        demo_item = _build_demo_item_for_roi(item, user_input, roi_boxes)
        output_text = _run_model_once(
            runtime,
            demo_item["image"],
            task_id,
            user_input if "<region0>" in user_input else f"{user_input}\n请基于<region0>回答。",
            [roi_boxes[0]],
        )
        vis_img = _draw_box(base, roi_boxes[0], color="orange")
        reply = _clean_assistant_text(output_text)

    elif task_id == 4:
        # 与 eval_stage2_cot.py 对齐：task4 虽是整图分类，也传入原始 regions 的 bbox_list
        # （模型输入构造链路要求非空 bbox_list，且 region token 对齐数据集格式）
        regions = item.get("regions", [])
        bbox_list = [r["bbox"] for r in regions if isinstance(r.get("bbox"), list) and len(r.get("bbox")) == 4]
        output_text = _run_model_once(runtime, image_path, task_id, user_input, bbox_list)
        reply = _clean_assistant_text(output_text)
    else:
        reply = f"暂不支持 task_id={task_id}"

    chat_history.append((user_input, reply))
    return chat_history, runtime, vis_img


def build_app() -> gr.Blocks:
    default_dataset = "/home/nas3/biod/shiyu/data/data_vl_20260115/vlm_fo1_data/dataset/balance/stage2_cot_val.json"
    # 与 scripts/eval_stage2_cot.sh 一致：base=Stage1（LoRA 目录），checkpoint=Stage2（LoRA）。
    # 勿把「官方 resources/VLM-FO1_...」填在 Base：最后一级目录名含 vlm-fo1 时会走「官方权重+单层 LoRA」，不会 merge Stage1。
    default_base = str(ROOT / "outputs" / "cervical_stage1_align_lora")
    default_ckpt = str(ROOT / "outputs" / "stage2_cot_balance")

    with gr.Blocks(title="VLM-FO1 Gradio Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## VLM-FO1 多任务演示（聊天版）")
        gr.Markdown(
            "支持 task_id=0,1,2,3,4,5；其中 **2/3/5** 需要矩形 ROI。\n\n"
            "**矩形 ROI**：在左侧图像上**连续单击两个点**（轴对齐矩形的两个对角），会出现橙色矩形；"
            "也可在下方填写 `x1 y1 x2 y2`（原图像素坐标）。"
            "若点击无反应，请升级 Gradio（≥4.30）或仅用手动坐标。\n\n"
            "**模型加载**：与 `scripts/eval_stage2_cot.py` / `eval_stage2_cot.sh` 相同——"
            "Base 填 **Stage1 align 输出目录**（含 adapter 的 LoRA），Checkpoint 填 **Stage2**。"
            "英文流程请改为 `outputs/stage1_align_en` + `outputs/stage2_cot_balance_en`。"
        )

        runtime_state = gr.State(RuntimeState())
        sample_state = gr.State(None)
        roi_state = gr.State({"clicks": [], "bbox": None})

        with gr.Row():
            dataset_path = gr.Textbox(label="数据集 JSON", value=default_dataset)
            task_id = gr.Dropdown(label="task_id", choices=[0, 1, 2, 3, 4, 5], value=0)
            sample_rank = gr.Dropdown(label="该任务内样本序号", choices=[0], value=0)
            btn_load = gr.Button("加载样本", variant="primary")

        with gr.Row():
            base_model = gr.Textbox(
                label="Base Model（Stage1 align，LoRA 目录）",
                value=default_base,
            )
            checkpoint = gr.Textbox(
                label="Checkpoint（Stage2，LoRA 目录）",
                value=default_ckpt,
            )
            device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda")
            language = gr.Dropdown(label="Language", choices=["zh", "en"], value="zh")

        sample_info = gr.Textbox(label="样本信息", lines=4)

        manual_bbox = gr.Textbox(
            label="手动 bbox（可选，原图像素 x1 y1 x2 y2，空格或逗号分隔）",
            placeholder="例: 120 80 360 420",
            lines=1,
        )

        with gr.Row():
            with gr.Column(scale=1):
                roi_image = gr.Image(
                    type="pil",
                    label="样本图像（task 2/3/5：单击两个对角点形成矩形）",
                    interactive=True,
                )
                with gr.Row():
                    btn_clear_roi = gr.Button("清除矩形选区")
            vis_output = gr.Image(label="Assistant 可视化输出", type="pil")

        chatbot = gr.Chatbot(label="聊天机器人", height=420)
        user_input = gr.Textbox(label="你的问题", placeholder="例如：请找出阳性宫颈细胞", lines=2)

        with gr.Row():
            btn_send = gr.Button("发送", variant="primary")
            btn_clear = gr.Button("清空对话")

        btn_load.click(
            fn=on_load_sample,
            inputs=[dataset_path, task_id, sample_rank],
            outputs=[roi_image, sample_state, sample_info, sample_rank, roi_state],
        )
        task_id.change(
            fn=on_load_sample,
            inputs=[dataset_path, task_id, sample_rank],
            outputs=[roi_image, sample_state, sample_info, sample_rank, roi_state],
        )
        sample_rank.change(
            fn=on_change_rank,
            inputs=[dataset_path, task_id, sample_rank],
            outputs=[roi_image, sample_state, sample_info, sample_rank, roi_state],
        )

        if hasattr(roi_image, "select"):
            roi_image.select(
                fn=on_roi_image_select,
                inputs=[sample_state, roi_state],
                outputs=[roi_image, roi_state],
            )
        btn_clear_roi.click(
            fn=on_clear_roi_selection,
            inputs=[sample_state],
            outputs=[roi_image, roi_state],
        )

        btn_send.click(
            fn=on_send,
            inputs=[
                chatbot,
                runtime_state,
                base_model,
                checkpoint,
                device,
                language,
                sample_state,
                task_id,
                user_input,
                roi_state,
                manual_bbox,
            ],
            outputs=[chatbot, runtime_state, vis_output],
        )
        btn_clear.click(lambda: ([], None), outputs=[chatbot, vis_output])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
