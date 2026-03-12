我们直接给出一个可落地、结构安全、理论自洽的 Pathology Token 方案，把你当前的
线性融合式 PA-FFE
改成
Token-level Pathology Injection
目标是：
● ✅ 不破坏原视觉 token 分布
● ✅ 保留定位能力
● ✅ 允许 LLM 自主决定是否使用病理统计
● ✅ 易于写成论文创新点
下面是完整重构方案。

一、核心思想：不要融合，改为“并列 token”
当前你是：
f_roi_fused = gate * f_path + (1-gate) * f_visual
改成：
[ROI_visual_token]
[ROI_pathology_token]
让它们在序列中并列存在。

二、整体结构修改
原结构（简化）：
Image → Vision Encoder → ROI Features → mm_projector → LLM
改成：
Image → Vision Encoder → ROI Visual Feature → mm_projector_visual → ROI_visual_token
Image → HoVer-Net → ROI Path Stats → path_mlp → mm_projector_path → ROI_path_token

Concatenate:
[ROI_visual_token_1]
[ROI_path_token_1]
[ROI_visual_token_2]
[ROI_path_token_2]
...
→ LLM

三、具体模块设计

1️⃣ Pathology Token Encoder
class PathTokenEncoder(nn.Module):
    def __init__(self, path_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(path_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
⚠️ 必须加 LayerNorm
⚠️ hidden_dim = vision hidden dim

2️⃣ 单独 projector（不要共用）
你现在 mm_projector 是共享的。
改成：
self.mm_projector_visual
self.mm_projector_path
否则会：
● 强迫两个分布对齐
● 引发梯度冲突

3️⃣ Token 拼接方式
在 encode_regions 里：
visual_tokens = self.mm_projector_visual(f_visual)
path_tokens = self.mm_projector_path(self.path_encoder(f_path))

# interleave
tokens = torch.stack([visual_tokens, path_tokens], dim=2)
tokens = tokens.view(B, -1, hidden_dim)
或者简单 concat：
tokens = torch.cat([visual_tokens, path_tokens], dim=1)
建议 interleave，语义更清晰。

四、关键设计：添加类型 Embedding
否则 LLM 不知道哪个是视觉 token。
加入：
self.type_embed = nn.Embedding(2, hidden_dim)
使用：
visual_tokens += self.type_embed(torch.zeros(...))
path_tokens += self.type_embed(torch.ones(...))
这是非常重要的。

五、输入顺序建议
每个 ROI 结构：
<roi_start>
<roi_visual>
<roi_path>
<roi_end>
你可以用 learnable token：
self.roi_start = nn.Parameter(...)
self.roi_end = nn.Parameter(...)
这样更结构化。

六、训练策略（关键）

阶段A：Path Token 对齐
训练：
● path_encoder
● mm_projector_path
冻结：
● vision encoder
● LLM
● mm_projector_visual
目标：
让 pathology token 学会进入 LLM token 空间。

阶段B：LoRA 微调
训练：
● LoRA
● path_encoder
● mm_projector_path
● mm_projector_visual
冻结：
● vision backbone

七、为什么这个结构比融合稳？
因为：
1️⃣ 不改变视觉 token 几何结构
视觉 token 分布保持原样
定位能力不会退化。

2️⃣ attention 自主选择
LLM 可以：
● 用视觉 token 做定位
● 用 path token 做分类
● 两者结合做描述

3️⃣ 避免 gate collapse
不再有：
f ≈ f_path
这种灾难。

伪代码实现如下：


一、模型结构重构

1️⃣ 新增模块
class PathologyTokenModule(nn.Module):
    def __init__(self, path_feat_dim, vision_hidden_dim, llm_hidden_dim):
        super().__init__()

        # 结构化病理特征编码
        self.path_encoder = nn.Sequential(
            nn.Linear(path_feat_dim, vision_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(vision_hidden_dim),
            nn.Linear(vision_hidden_dim, vision_hidden_dim)
        )

        # 单独 projector（不要和 visual 共用）
        self.mm_projector_path = nn.Linear(
            vision_hidden_dim, llm_hidden_dim
        )

        # token 类型 embedding
        self.type_embed = nn.Embedding(2, llm_hidden_dim)
        # 0 = visual
        # 1 = pathology
⚠️ 注意：
不要再使用 gate
不要再修改 visual ROI feature

二、修改 encode_regions
原逻辑（简化）：
f_visual = vision_encoder(roi)
f_visual = mm_projector(f_visual)
改成：
def encode_regions(self, roi_images, roi_path_features):

    # 1️⃣ 视觉特征
    f_visual = self.vision_encoder(roi_images)   # [B, N, C]
    visual_tokens = self.mm_projector_visual(f_visual)

    visual_tokens = visual_tokens + \
        self.type_embed(torch.zeros_like(visual_tokens[...,0], dtype=torch.long))

    # 2️⃣ 病理统计特征
    f_path = self.pathology_module.path_encoder(roi_path_features)

    path_tokens = self.pathology_module.mm_projector_path(f_path)

    path_tokens = path_tokens + \
        self.pathology_module.type_embed(
            torch.ones_like(path_tokens[...,0], dtype=torch.long)
        )

    # 3️⃣ interleave
    tokens = torch.stack([visual_tokens, path_tokens], dim=2)
    tokens = tokens.view(B, -1, hidden_dim)

    return tokens
结构变为：
ROI1_visual
ROI1_path
ROI2_visual
ROI2_path
...

三、forward 修改
原：
roi_tokens = encode_regions(...)
llm_input = concat(prompt_tokens, roi_tokens)
保持一致，只是 roi_tokens 已改变。

四、训练阶段修改

🔹 阶段A：Pathology Token 对齐阶段
# 冻结
freeze(self.vision_encoder)
freeze(self.llm)
freeze(self.mm_projector_visual)

# 训练
unfreeze(self.pathology_module.path_encoder)
unfreeze(self.pathology_module.mm_projector_path)
目标：
让 pathology token 分布对齐到 LLM token 空间。

🔹 阶段B：Full + LoRA
enable_lora(self.llm)

unfreeze(self.mm_projector_visual)
unfreeze(self.pathology_module.path_encoder)
unfreeze(self.pathology_module.mm_projector_path)
冻结：
freeze(self.vision_encoder)

五、重要：梯度检查
确保：
for name, param in model.named_parameters():
    print(name, param.requires_grad)
检查没有误训练视觉 backbone。

第二部分：完整模型结构图（论文版）
下面给你一个可以直接画图的结构框架。

📊 模型总览结构
┌───────────────────────────────┐
                     │          Qwen2.5-VL LLM        │
                     │     (with LoRA in stage B)     │
                     └───────────────▲────────────────┘
                                     │
                        Concatenated Multimodal Tokens
                                     │
     ┌────────────────────────────────┼────────────────────────────────┐
     │                                │                                │
┌──────────────┐               ┌──────────────┐               ┌──────────────┐
│ ROI Visual   │               │ ROI Path     │               │ Prompt/Text  │
│ Tokens       │               │ Tokens       │               │ Tokens       │
└──────▲───────┘               └──────▲───────┘               └──────────────┘
       │                               │
┌──────┴────────┐              ┌───────┴────────┐
│ mm_projector  │              │ mm_projector   │
│ (visual)      │              │ (path)         │
└──────▲────────┘              └───────▲────────┘
       │                               │
┌──────┴────────┐              ┌───────┴────────┐
│ Vision Encoder│              │ Path Encoder   │
│ (Qwen-VL)     │              │ (MLP + LN)     │
└──────▲────────┘              └───────▲────────┘
       │                               │
   ROI Images                  ROI Path Statistics
                                     │
                                HoVer-Net

📌 关键区别（论文中强调）
原 PA-FFE	Pathology Token
Feature-level fusion	Token-level injection
改变视觉 token 分布	保持视觉 token 几何
可能 gate collapse	无 collapse 风险
干扰定位	定位稳定