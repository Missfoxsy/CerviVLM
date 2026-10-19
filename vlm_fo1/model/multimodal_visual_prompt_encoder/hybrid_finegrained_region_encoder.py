import os
import torch
import torch.nn as nn
from typing import List, Tuple, Union
import torch.nn.functional as F
from torchvision.ops import roi_align
import math

from vlm_fo1.model.multimodal_visual_prompt_encoder.simple_fpn import SimpleFP


class AdaptiveGatingNetwork(nn.Module):
    """
    自适应门控：Sigmoid 独立门控 + 下界裁剪 + 行归一化（Σ=1），替代 Softmax。
    与逐层相乘后再 concat 的融合形式配合，保持混合特征总尺度稳定。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_levels: int = 4,
        dropout: float = 0.1,
        init_bias: float = 2.0,
        floor: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.floor = floor
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_levels),
        )
        # sigmoid(2.0)≈0.88，初始偏向“开启”，再经 floor 与归一化
        nn.init.constant_(self.gate_net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_net(x)
        gates_raw = torch.sigmoid(logits)
        span = 1.0 - self.floor
        gates_clipped = self.floor + span * gates_raw
        gate_sum = gates_clipped.sum(dim=1, keepdim=True) + 1e-8
        gates = gates_clipped / gate_sum
        return gates, gates_raw


class HierarchicalFeatureFusion(nn.Module):
    """
    Layer-wise hierarchical fusion for ROI features（由粗到细）。

    legacy：逐层 encoder + 跨层 MultiheadAttention + concat → Linear（无门控）。
    task_adaptive / gated：在 legacy 特征上叠加任务/样本门控（Sigmoid+floor+行归一化）。
    softmax_gate：与 task_adaptive 同结构（跨层 MHA + level_selector + task bias），
                  仅将门控改为 Softmax(z)（表格 Softmax Gate 对照）。
    mean：无跨层注意力，各层特征等权平均后再投影（Average 对照）。
    concat：无跨层注意力，拼接多层特征后线性融合（Concat + Linear 对照）。
    weighted：无跨层注意力，静态可学层级权重。
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        num_heads: int = 8,
        fusion_mode: str = "legacy",
        num_levels: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.num_levels = int(num_levels)

        # legacy | task_adaptive | softmax_gate | gated | mean | concat | weighted
        self.fusion_mode = fusion_mode
        self.level_encoders = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(self.num_levels)
        ])
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True,
            ) for _ in range(max(0, self.num_levels - 1))
        ])
        self.final_fusion = nn.Linear(self.num_levels * feature_dim, out_dim)
        self.level_selector = nn.Sequential(
            nn.Linear(self.num_levels * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_levels),
        )
        # Task-conditioned bias for per-task gating (type0..type5).
        # Initialize with mild, separable priors:
        # - task0/4: coarse-mid preference
        # - task2/3/5: finer-level preference
        # - task1: balanced
        task_bias_prior = torch.tensor(
            [
                [0.20, 0.10, -0.05, -0.15],  # task0: coarse/mid
                [0.00, 0.00, 0.00, 0.00],    # task1: balanced
                [-0.15, -0.05, 0.10, 0.20],  # task2: fine
                [-0.15, -0.05, 0.10, 0.20],  # task3: fine
                [0.20, 0.10, -0.05, -0.15],  # task4: coarse/mid
                [-0.15, -0.05, 0.10, 0.20],  # task5: fine
            ],
            dtype=torch.float32,
        )
        self.task_level_bias = nn.Parameter(task_bias_prior)
        # task_adaptive gates: use sigmoid + floor + normalize (instead of softmax) to allow more independent scaling.
        # Lower temp and lower floor usually produce sharper inter-level differences.
        self.task_adaptive_gate_temp = 0.8
        self.task_adaptive_gate_floor = 0.02
        self.task_adaptive_task_bias_scale = 2.0
        # Sample-adaptive hierarchical gates (Scheme B): Sigmoid + floor + row-norm（非 Softmax）
        gate_in_dim = self.num_levels * feature_dim
        self.gate_network = AdaptiveGatingNetwork(
            input_dim=gate_in_dim,
            hidden_dim=512,
            num_levels=self.num_levels,
            dropout=0.1,
        )

        # weighted 模式：全局静态层级权重（softmax，无样本/任务条件）
        self.static_level_logits = nn.Parameter(torch.zeros(self.num_levels))

        # Optional debug cache (last forward); can be read by probing scripts.
        self.last_level_features = None
        self.last_level_attn = None
        self.last_gates = None
        self.last_gates_raw = None

    @staticmethod
    def _normalize_sigmoid_gates(gate_logits: torch.Tensor, floor: float, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = max(float(temp), 1e-3)
        floor = float(max(0.0, min(0.9, floor)))
        gates_raw = torch.sigmoid(gate_logits / temp)
        span = 1.0 - floor
        gates_clipped = floor + span * gates_raw
        gates = gates_clipped / (gates_clipped.sum(dim=-1, keepdim=True) + 1e-8)
        return gates, gates_raw

    @staticmethod
    def _parse_task_id(task_type) -> int:
        """Map task tag to row index 0..5 for task_adaptive bias. Always returns a valid index.

        Accepts ``type0``..``type5`` and legacy aliases ``localization`` / ``classification`` / ``reasoning``.
        Unknown or invalid values fall back to ``1`` (与 task1 均衡先验一致).
        """
        if not isinstance(task_type, str):
            return 1
        t = task_type.strip().lower()
        if t.startswith("type"):
            try:
                v = int(t.replace("type", "", 1))
                return v if 0 <= v <= 5 else 1
            except Exception:
                return 1
        legacy = {"localization": 0, "classification": 2, "reasoning": 1}
        return legacy.get(t, 1)

    @staticmethod
    def _build_hierarchy_levels(roi_feats_per_level: List[torch.Tensor], num_levels: int) -> List[torch.Tensor]:
        """对齐到 num_levels：层数足够时取前 num_levels；不足则重复最后一层补齐。"""
        target = int(num_levels)
        n = len(roi_feats_per_level)
        if n == 0:
            raise ValueError("roi_feats_per_level is empty")
        if n >= target:
            return [roi_feats_per_level[i] for i in range(target)]
        out = list(roi_feats_per_level)
        while len(out) < target:
            out.append(out[-1])
        return out

    def forward(self, roi_feats_per_level: List[torch.Tensor], task_type: str = "type1") -> torch.Tensor:
        levels = self._build_hierarchy_levels(roi_feats_per_level, self.num_levels)
        target_dtype = self.level_encoders[0].weight.dtype

        # mean / concat / weighted：无跨层 MultiheadAttention 的短路径
        if self.fusion_mode in ("mean", "concat", "weighted"):
            encoded = []
            for i, feat in enumerate(levels):
                if feat.dtype != target_dtype:
                    feat = feat.to(target_dtype)
                encoded.append(self.level_encoders[i](feat))  # [N, D]

            self.last_level_features = [e.detach() for e in encoded]
            self.last_level_attn = []

            if self.fusion_mode == "mean":
                # Average：各层等权平均（再经 final_fusion 对齐到 out_dim）
                avg = torch.stack(encoded, dim=0).mean(dim=0)  # [N, D]
                pooled = avg.repeat(1, self.num_levels)  # [N, num_levels*D]
                fused = self.final_fusion(pooled)
                self.last_gates = None
                self.last_gates_raw = None
            elif self.fusion_mode == "concat":
                # Concat + Linear：拼接多层特征后线性融合（无跨层注意力、无门控）
                concat = torch.cat(encoded, dim=-1)  # [N, num_levels*D]
                fused = self.final_fusion(concat)
                self.last_gates = None
                self.last_gates_raw = None
            else:  # weighted
                w = torch.softmax(self.static_level_logits.to(target_dtype), dim=0)  # [num_levels]
                weighted = [encoded[i] * w[i] for i in range(self.num_levels)]
                weighted_concat = torch.cat(weighted, dim=-1)  # [N, num_levels*D]
                fused = self.final_fusion(weighted_concat)
                self.last_gates = w.detach().unsqueeze(0).expand(encoded[0].shape[0], -1)
                self.last_gates_raw = self.last_gates
            return fused

        # legacy / task_adaptive / softmax_gate / gated（含跨层注意力）
        level_features = []
        attn_maps = []
        prev = None

        for i, feat in enumerate(levels):
            if feat.dtype != target_dtype:
                feat = feat.to(target_dtype)
            # [N, D] -> [1, N, D]
            cur = self.level_encoders[i](feat).unsqueeze(0)
            if prev is not None:
                # 保留按 head 的权重 [B, H, Nq, Nk]，便于可视化；与 average=True 时前向计算一致。
                try:
                    cur, attn = self.level_attentions[i - 1](
                        query=cur,
                        key=prev,
                        value=prev,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                except TypeError:
                    cur, attn = self.level_attentions[i - 1](query=cur, key=prev, value=prev)
                attn_maps.append(attn)
            prev = cur
            level_features.append(cur.squeeze(0))

        self.last_level_features = [lf.detach() for lf in level_features]
        self.last_level_attn = [a.detach() for a in attn_maps]

        stacked = torch.cat(level_features, dim=-1)  # [N, 4D]
        if self.fusion_mode == "task_adaptive":
            task_id = self._parse_task_id(task_type)
            # Shared selector + task-specific additive bias => explicit per-task gates.
            gate_logits = self.level_selector(stacked)
            gate_logits = gate_logits + self.task_adaptive_task_bias_scale * self.task_level_bias[task_id : task_id + 1]
            gates, gates_raw = self._normalize_sigmoid_gates(
                gate_logits,
                floor=self.task_adaptive_gate_floor,
                temp=self.task_adaptive_gate_temp,
            )  # [N, 4], row-sum=1
            gated_concat = torch.cat(
                [level_features[i] * gates[:, i : i + 1] for i in range(self.num_levels)], dim=-1
            )
            # Task-adaptive path now uses unified dimensionality:
            # task_id only modulates gates; fusion head is shared to avoid branch capacity imbalance.
            fused = self.final_fusion(gated_concat)
            self.last_gates = gates.detach()
            self.last_gates_raw = gates_raw.detach()
        elif self.fusion_mode == "softmax_gate":
            # Softmax Gate ablation: identical to task_adaptive except α = Softmax(z/temp).
            task_id = self._parse_task_id(task_type)
            gate_logits = self.level_selector(stacked)
            gate_logits = gate_logits + self.task_adaptive_task_bias_scale * self.task_level_bias[task_id : task_id + 1]
            temp = max(float(self.task_adaptive_gate_temp), 1e-3)
            gates = torch.softmax(gate_logits / temp, dim=-1)  # [N, 4], Σ=1
            gated_concat = torch.cat(
                [level_features[i] * gates[:, i : i + 1] for i in range(self.num_levels)], dim=-1
            )
            fused = self.final_fusion(gated_concat)
            self.last_gates = gates.detach()
            self.last_gates_raw = gates.detach()
        elif self.fusion_mode == "gated":
            gates, gates_raw = self.gate_network(stacked)  # [N, 4], Σ row = 1
            gated_features = [
                level_features[i] * gates[:, i : i + 1] for i in range(self.num_levels)
            ]
            gated_concat = torch.cat(gated_features, dim=-1)  # [N, 4D]
            fused = self.final_fusion(gated_concat)           # [N, out_dim]
            self.last_gates = gates.detach()
            self.last_gates_raw = gates_raw.detach()
        else:  # legacy
            fused = self.final_fusion(stacked)           # [N, out_dim]
            self.last_gates = None
            self.last_gates_raw = None
        return fused


def visualize_hierarchical_attention(hierarchical_fusion: HierarchicalFeatureFusion, save_path: str = "attn_vis.png"):
    """
    Visualize cached cross-level attention maps from HierarchicalFeatureFusion.

    Note:
      - Call this AFTER a forward() pass where hierarchical fusion is enabled.
      - last_level_attn entries: [B, N, N]（旧）或 [B, H, N, N]（按 head，推荐）。
      - 当 ROI 数 N=1 时，每层注意力矩阵为 1×1，softmax 在单 key 上恒为 1.0，三张图会数值相同；
        此时应结合下方门控（last_gates）或换多样本（多 region）解读。
    """
    if hierarchical_fusion is None or not hasattr(hierarchical_fusion, "last_level_attn"):
        print("No hierarchical_fusion attention cache found.")
        return
    if not hierarchical_fusion.last_level_attn:
        print("No attention maps cached. Run forward() first.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
    except Exception as e:
        print(f"Failed to import matplotlib/seaborn for attention visualization: {e}")
        return

    attn_list = hierarchical_fusion.last_level_attn
    n_levels = len(attn_list)
    gates = getattr(hierarchical_fusion, "last_gates", None)
    show_gate_row = gates is not None and gates.numel() > 0

    n_rows = 2 if show_gate_row else 1
    fig = plt.figure(figsize=(max(5.5 * n_levels, 10), 7 if show_gate_row else 5))
    gs = gridspec.GridSpec(n_rows, n_levels, height_ratios=[2.2, 1.0] if show_gate_row else [1.0], hspace=0.4, wspace=0.35)

    degenerate_warn = False
    for i, attn in enumerate(attn_list):
        ax = fig.add_subplot(gs[0, i])
        t = attn
        head_std_msg = ""
        if t.dim() == 4:
            # [B, H, Nq, Nk]：主图用 head 平均；标题附带跨 head 差异提示
            t0 = t[0].float()
            attn_2d = t0.mean(dim=0).detach().cpu().numpy()
            if t0.shape[0] > 1:
                hs = t0.reshape(t0.shape[0], -1).std(dim=1).mean().item()
                head_std_msg = f" | mean head-std={hs:.4f}"
        elif t.dim() == 3:
            attn_2d = t[0].detach().float().cpu().numpy()
        else:
            print(f"Skip attn level {i}: unexpected rank {t.dim()}")
            continue

        nq, nk = int(attn_2d.shape[0]), int(attn_2d.shape[1])
        show_annot = max(nq, nk) <= 10
        degenerate = nq * nk <= 1 or (float(attn_2d.max()) - float(attn_2d.min()) < 1e-6 and nq * nk <= 4)
        if degenerate:
            degenerate_warn = True

        fmt = ".4f" if nq * nk <= 4 else ".2f"
        sns.heatmap(
            attn_2d,
            cmap="inferno",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.75},
            annot=show_annot,
            fmt=fmt,
        )
        ax.set_title(f"L{i + 1}→L{i + 2}  ({nq}×{nk}){head_std_msg}")
        ax.set_xlabel("key: prev-level ROI")
        ax.set_ylabel("query: cur-level ROI")

    if show_gate_row:
        axb = fig.add_subplot(gs[1, :])
        g = gates.detach().float().cpu().numpy()
        if g.ndim == 2 and g.shape[1] >= 4:
            mean_g = g.mean(axis=0)[:4]
            axb.bar(["L0", "L1", "L2", "L3"], mean_g, color="steelblue")
            axb.set_ylabel("gate (mean over ROIs)")
            axb.set_title("Level fusion gates (task_adaptive / gated)")
            axb.set_ylim(0.0, max(1.0, float(mean_g.max()) * 1.15))
        else:
            axb.text(0.5, 0.5, f"gates shape {getattr(g, 'shape', None)}", ha="center", va="center")

    if degenerate_warn:
        fig.suptitle(
            "Note: N=1 ROI → each cross-level attn is 1×1 softmax (=1.0); maps look identical. "
            "Use gate bars below or a sample with more regions.",
            fontsize=10,
            y=1.02,
        )
        print(
            "[visualize_hierarchical_attention] 单 ROI（或极小矩阵）：层级注意力热力图数值退化，"
            "已叠加门控条形图（若有）；多 region 样本上热力图更有区分度。"
        )

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention visualization saved to {save_path}")


def generate_2d_position_embedding(height, width, dim, device):
    """Generate a 2D positional encoding for a feature map.

    Args:
        height (int): Height of the feature map.
        width (int): Width of the feature map.
        dim (int): Dimensionality of the positional embedding (should match channel count).
        device: Torch device on which to allocate tensors.

    Returns:
        pos_embed (Tensor): Positional encoding of shape [H, W, dim].
    """
    # Generate grid coordinate vectors of length H and W
    y_pos = torch.arange(height, dtype=torch.float32, device=device)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)
    
    # Normalize grid values to [0, 1]
    y_pos = y_pos / height
    x_pos = x_pos / width
    
    # Create mesh grid (Y: rows, X: cols)
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
    
    scale = 2 * math.pi
    # Calculate positions for sine/cosine encoding
    quarter_dim = dim // 4
    dim_t = torch.arange(quarter_dim, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / quarter_dim) if quarter_dim > 0 else torch.tensor([1.0], device=device)
    
    # X direction encoding
    x_embed = x_grid.unsqueeze(-1) * scale  # [H, W, 1]
    pos_x = x_embed / dim_t  # [H, W, quarter_dim]
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # Alternating sin/cos
    
    # Y direction encoding
    y_embed = y_grid.unsqueeze(-1) * scale  # [H, W, 1]
    pos_y = y_embed / dim_t  # [H, W, quarter_dim]
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # Alternating sin/cos
    
    # Concatenate along the last dimension to make [H, W, dim]
    pos_embed = torch.cat([pos_y, pos_x], dim=-1)
    
    return pos_embed

def gen_sineembed_for_position(pos_tensor, dim_of_pos_feats):
    """Generate sine/cosine positional embedding for ROI position(s).

    Args:
        pos_tensor (Tensor): Shape [batch_size, N, 4] (format: [cx, cy, w, h] in normalized [0, 1])
        dim_of_pos_feats (int): Output embedding dimensionality (#positional channels).

    Returns:
        pos (Tensor): [batch_size, N, dim_of_pos_feats * (2, 4, ...)]
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        dim_of_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_of_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    # Generate encodings for cx, cy
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        # [cx, cy] input
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        # [cx, cy, w, h] input
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # Concatenate encodings for [cy, cx, w, h]
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class HFREModule(nn.Module):
    """Hybrid Finegrained Region Encoder (HFREModule).

    Handles multi-level ROI region features, optional position embedding, and feature combination for hybrid visual prompt encoding.

    Args:
        roi_output_size (Optional[int]): Output spatial size for ROIAlign.
        region_feature_dim (int): The output dimension for region features.
        apply_position_embedding (bool): Whether positional embedding is used in region features.
        pos_embedding_strategy (str): 'bbox_based', 'feature_map_based', or 'hybrid'.
        use_vt_region_feature_only (bool): Only use vision tower features, skip auxiliary.
        use_vision_tower_region_feature (bool): Whether to include vision tower region features.
        region_feature_combination (str): Combination method: 'concat', 'mean', etc.
        use_separate_mlp_for_regions (bool): Whether to MLP project each region type separately.
        apply_region_layer_norm (bool): Whether to apply layernorm to region features.
        vision_tower_region_feature_dim (int): #channels for vision-tower region feature.
        vision_tower_spatial_scale (float): Spatial scale for vision-tower (for roi_align).
        use_simpleFPN_for_vt (bool): Whether to use FPN on the vision-tower output.
        aux_vision_tower_region_feature_dims (List[int]): Channel dimensions of auxiliary features list.
        aux_vision_tower_spatial_scale (float): Spatial scale for auxiliary vision-tower features.
    """

    def __init__(
        self,
        roi_output_size: int = None,                  # Output spatial size for ROI region features
        region_feature_dim: int = 1024,               # Output dimension for final region feature
        apply_position_embedding: bool = False,       # Whether to apply position embedding
        pos_embedding_strategy: str = 'bbox_based',   # Strategy: 'bbox_based', 'feature_map_based', 'hybrid'
        use_vt_region_feature_only: bool = False,     # Use vision tower (VT) region features only
        use_vision_tower_region_feature: bool = False,# Use vision tower region features (with others)
        region_feature_combination: str = 'concat',   # How to combine aux and vt region features
        use_separate_mlp_for_regions: bool = False,   # MLP-per-region
        apply_region_layer_norm: bool = False,        # Apply LayerNorm

        # Primary vision tower related
        vision_tower_region_feature_dim: int = 5120,    # Dim of the VT region feature
        vision_tower_spatial_scale: float = 1/14,       # Spatial scale of the VT for roi_align
        use_simpleFPN_for_vt: bool = False,             # Use simpleFPN for vision tower

        # Auxiliary vision tower related
        aux_vision_tower_region_feature_dims: List[int] = [256, 512, 1024, 2048],
        aux_vision_tower_spatial_scale: float = None,    # Scale for aux VT
        use_hierarchical_fusion: bool = False,           # Enable layer-wise hierarchical fusion
        hierarchical_num_heads: int = 8,                 # Num heads for cross-level attention
        hierarchical_fusion_mode: str = "legacy",        # legacy | task_adaptive | gated
    ):
        super(HFREModule, self).__init__()
        self.roi_output_size = roi_output_size
        self.region_feature_dim = region_feature_dim
        self.apply_position_embedding = apply_position_embedding
        self.pos_embedding_strategy = pos_embedding_strategy
        self.use_vt_region_feature_only = use_vt_region_feature_only
        self.use_vision_tower_region_feature = use_vision_tower_region_feature
        self.region_feature_combination = region_feature_combination
        self.use_separate_mlp_for_regions = use_separate_mlp_for_regions
        self.apply_region_layer_norm = apply_region_layer_norm

        self.vision_tower_region_feature_dim = vision_tower_region_feature_dim
        self.vision_tower_spatial_scale = vision_tower_spatial_scale
        self.use_simpleFPN_for_vt = use_simpleFPN_for_vt

        self.aux_vision_tower_region_feature_dims = aux_vision_tower_region_feature_dims
        self.aux_vision_tower_spatial_scale = aux_vision_tower_spatial_scale
        self.use_hierarchical_fusion = use_hierarchical_fusion
        self.hierarchical_num_heads = hierarchical_num_heads
        self.hierarchical_fusion_mode = hierarchical_fusion_mode

        # Print configuration for debugging
        # print(f"output_size: {self.roi_output_size} use_vision_tower_region_feature: {self.use_vision_tower_region_feature} vision_tower_region_feature_dim: {self.vision_tower_region_feature_dim} "
        #       f"apply_position_embedding: {self.apply_position_embedding} region_feature_combination: {self.region_feature_combination} region_feature_dim: {self.region_feature_dim} use_vt_region_feature_only: {self.use_vt_region_feature_only} "
        #       f"use_simpleFPN_for_vt: {self.use_simpleFPN_for_vt} pos_embedding_strategy: {self.pos_embedding_strategy} "
        #       f"apply_region_layer_norm: {self.apply_region_layer_norm}")
        
        # Optional: FPN for the vision tower input if enabled
        self.fpn_num_levels = 4
        self.fpn_level_indices = [0, 1, 2, 3]
        self.vt_fpn_strides = [3.5, 7.0, 14.0, 28.0]
        self.vt_fpn_out_dim = 2048

        if self.use_simpleFPN_for_vt:
            self.simple_fpn = SimpleFP(out_channels=512, norm="LN", square_pad=0, dim=1280, stride=14)
            if self.use_hierarchical_fusion:
                # With SimpleFPN, each level ROI feature is [N, 512], and legacy VT output is [N, 2048]
                self.hierarchical_fusion = HierarchicalFeatureFusion(
                    feature_dim=512,
                    out_dim=2048,
                    num_heads=self.hierarchical_num_heads,
                    fusion_mode=self.hierarchical_fusion_mode,
                )

        # LayerNorm for auxiliary and VT region features, if enabled
        if self.apply_region_layer_norm:
            if self.use_vision_tower_region_feature:
                self.vt_region_norm = nn.LayerNorm(self.vision_tower_region_feature_dim)
            if not self.use_vt_region_feature_only:
                self.aux_region_norm = nn.LayerNorm(sum(self.aux_vision_tower_region_feature_dims))
            
        # Optionally, a projection MLP if using certain combination strategies
        if self.use_vision_tower_region_feature and self.region_feature_combination in ['mean', 'mean_sep_pos', 'mean_aux_pos', 'mean_sep_no_vt_pos']:
            self.vision_tower_region_feature_projector = nn.Sequential(
                nn.Linear(vision_tower_region_feature_dim, region_feature_dim),
                nn.GELU(),
                nn.Linear(region_feature_dim, region_feature_dim)
            )    
        
        # Two MLP heads if regions are projected separately (for concat mode)
        if self.use_vision_tower_region_feature and self.use_separate_mlp_for_regions:
            self.vt_region_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )
            self.aux_region_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )

    def _apply_feature_map_position_embedding(self, features):
        """Apply 2D position embedding to each feature map in a feature pyramid, if enabled.

        Args:
            features (List[Tensor]): Each is [B, C, H, W] per FPN level.

        Returns:
            List[Tensor]: Feature maps with position embedding applied, shape unchanged.
        """
        enhanced_features = []
        for level_idx, feature in enumerate(features):
            if self.apply_position_embedding and self.pos_embedding_strategy in ['feature_map_based', 'hybrid']:
                B, C, H, W = feature.shape

                # Generate position embedding matching channel dimension
                pos_embed = generate_2d_position_embedding(
                    H, W, C, feature.device
                )  # [H, W, C]
                
                # Reshape to [1, C, H, W] and add
                pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)
                feature = feature + pos_embed.to(feature.dtype)
            enhanced_features.append(feature)
        return enhanced_features

    def extract_vt_region_feature(self, multi_level_features, boxes: Union[torch.Tensor, List[torch.Tensor]], task_type: str = "type1") -> torch.Tensor:
        """Extract vision-tower region features via ROIAlign over FPN features, with spatial scaling.

        Args:
            multi_level_features (List[Tensor]): Per-FPN level features, [B, C, H, W].
            boxes (Union[Tensor, List[Tensor]]): ROI bounding boxes for roi_align.

        Returns:
            Tensor: [1, N, C=tower_channels]
        """
        if self.use_simpleFPN_for_vt:
            multi_level_features = self.simple_fpn(multi_level_features)
            level_indices = getattr(self, "fpn_level_indices", list(range(len(multi_level_features))))
            vt_strides = getattr(self, "vt_fpn_strides", [3.5, 7.0, 14.0, 28.0])
            roi_features_per_level = []
            for rel_idx, src_idx in enumerate(level_indices):
                if src_idx >= len(multi_level_features):
                    continue
                level_feature = multi_level_features[src_idx]
                stride = vt_strides[rel_idx] if rel_idx < len(vt_strides) else vt_strides[-1]
                current_spatial_scale = 1.0 / float(stride)
                level_roi_feat = roi_align(
                    level_feature.float(),
                    boxes,
                    output_size=self.roi_output_size,
                    spatial_scale=current_spatial_scale,
                )
                level_roi_feat = level_roi_feat.mean(dim=(2, 3))
                roi_features_per_level.append(level_roi_feat)
            if self.use_hierarchical_fusion and hasattr(self, "hierarchical_fusion"):
                out_box_feat = self.hierarchical_fusion(roi_features_per_level, task_type=task_type).unsqueeze(0)
            else:
                cat_feat = torch.cat(roi_features_per_level, dim=1)
                proj = getattr(self, "vt_level_fusion_proj", None)
                if proj is not None:
                    cat_feat = proj(cat_feat)
                out_box_feat = cat_feat.unsqueeze(0)
        else:
            # If not using FPN: concatenate all feature levels on channel axis and ROI-align once
            concat_multi_level_feature = []
            concat_multi_level_feature = torch.cat(multi_level_features, dim=1)

            out_box_feat = roi_align(
                concat_multi_level_feature.float(),
                boxes,
                output_size=self.roi_output_size,
                spatial_scale=self.vision_tower_spatial_scale,
            )
            # Pool per ROI for (1, N, C_total)
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
        return out_box_feat
    
    def __call__(
        self,
        aux_multi_level_features: List[torch.Tensor],
        aux_boxes: Union[torch.Tensor, List[torch.Tensor]],
        vt_multi_level_features = None,
        vt_boxes: Union[torch.Tensor, List[torch.Tensor]] = None,
        task_type: str = "type1",
    ) -> torch.Tensor:
        """Main forward. Extracts ROI region features with possible hybrid VT/aux, applies position embedding and combines as configured.

        Args:
            aux_multi_level_features (List[Tensor]): Auxiliary vision features (e.g., from FPN, [B, C, H, W]).
            aux_boxes (Union[Tensor, List[Tensor]]): ROIs in [N, 4] xyxy.
            vt_multi_level_features (optional): Vision tower features.
            vt_boxes (optional): Vision tower's box coordinates ([N, 4]).

        Returns:
            Tensor: Region features of shape [1, N, C], N=#ROIs.
        """
        if self.use_vt_region_feature_only:
            # Only use VT region features (skip aux completely)
            out_box_feat = self.extract_vt_region_feature(vt_multi_level_features, vt_boxes, task_type=task_type)

            if self.apply_position_embedding:
                # Add position embedding to VT region feature
                pos_boxes = vt_boxes[0]  # (N, 4)
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                original_img_width = vt_max_width / self.vision_tower_spatial_scale
                original_img_height = vt_max_height / self.vision_tower_spatial_scale
                # Normalize box coordinates by image size
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # Convert from (x1, y1, x2, y2) to (cx, cy, w, h)
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                # Add sine/cos position embedding
                pos_embed = gen_sineembed_for_position(pos_boxes.unsqueeze(0), self.region_feature_dim // 4)
                out_box_feat = out_box_feat + pos_embed
            return out_box_feat
        
        # Otherwise: hybrid mode (aux + possibly VT region features)
        aux_boxes[0] = aux_boxes[0].float()
        
        # Collect all auxiliary features at the same (max) spatial size for channel concat
        concat_multi_level_feature = []
        max_height = max([feature.shape[2] for feature in aux_multi_level_features])
        max_width = max([feature.shape[3] for feature in aux_multi_level_features])
        
        # Optionally apply 2D position encoding at the feature map level (before concat/roi_align)
        if self.pos_embedding_strategy in ['feature_map_based', 'hybrid']:
            # Option: compute stride info for each level for debugging/extension
            feature_strides = []
            for feature in aux_multi_level_features:
                stride = max_height / feature.shape[2]
                feature_strides.append(stride)
            aux_multi_level_features = self._apply_feature_map_position_embedding(
                aux_multi_level_features
            )
        
        # Interpolate all features to (max_height,max_width), then concat along channel
        for level, feature in enumerate(aux_multi_level_features):
            if level != 0:
                concat_multi_level_feature.append(
                    F.interpolate(
                        feature.float(),
                        size=(max_height, max_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                concat_multi_level_feature.append(feature.float())
        concat_multi_level_feature = torch.cat(concat_multi_level_feature, dim=1)

        # Extract region feature for all boxes using roi_align
        out_box_aux_feat = roi_align(
            concat_multi_level_feature,
            aux_boxes,
            output_size=self.roi_output_size,
            spatial_scale=self.aux_vision_tower_spatial_scale
        )
        
        # Pool H,W to get final shape (1, Nbox, C)
        out_box_aux_feat = out_box_aux_feat.mean(dim=(2, 3)).reshape(
            1, out_box_aux_feat.shape[0], out_box_aux_feat.shape[1]
        )

        if self.apply_region_layer_norm:
            out_box_aux_feat = self.aux_region_norm.float()(out_box_aux_feat)
        
        if self.use_vision_tower_region_feature:
            # If also using vision-tower features
            out_box_vt_feat = self.extract_vt_region_feature(vt_multi_level_features, vt_boxes, task_type=task_type)
            if self.apply_region_layer_norm:
                out_box_vt_feat = self.vt_region_norm.float()(out_box_vt_feat)
            if self.region_feature_combination in ['mean', 'mean_aux_pos']:
                # Combine by mean
                out_box_feat = (out_box_aux_feat + out_box_vt_feat) / 2
            elif self.region_feature_combination in ['concat', 'concat_aux_pos']:
                # Optionally MLP each before concat
                if self.use_separate_mlp_for_regions:
                    original_vt_dtype = out_box_vt_feat.dtype
                    original_aux_dtype = out_box_aux_feat.dtype
                    out_box_vt_feat = self.vt_region_mlp(out_box_vt_feat.to(self.vt_region_mlp[0].weight.dtype)).to(original_vt_dtype)
                    out_box_aux_feat = self.aux_region_mlp(out_box_aux_feat.to(self.aux_region_mlp[0].weight.dtype)).to(original_aux_dtype)
                out_box_feat = torch.cat([out_box_aux_feat, out_box_vt_feat], dim=-1)
            elif self.region_feature_combination in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                # Compute position embedding separately for aux and vt features
                # Use `aux_boxes` for aux and `vt_boxes` for vt
                vt_dim = 5120 if self.region_feature_combination == 'concat_sep_pos' else 2880

                # Aux region: positional embedding using aux_boxes
                aux_pos_boxes = aux_boxes[0].to(out_box_aux_feat.dtype)  # (N, 4)
                aux_original_img_width = max_width / self.aux_vision_tower_spatial_scale
                aux_original_img_height = max_height / self.aux_vision_tower_spatial_scale
                
                aux_pos_boxes[:, [0, 2]] = aux_pos_boxes[:, [0, 2]] / aux_original_img_width
                aux_pos_boxes[:, [1, 3]] = aux_pos_boxes[:, [1, 3]] / aux_original_img_height
                aux_pos_boxes[:, 2] = aux_pos_boxes[:, 2] - aux_pos_boxes[:, 0]
                aux_pos_boxes[:, 3] = aux_pos_boxes[:, 3] - aux_pos_boxes[:, 1]
                aux_pos_boxes[:, 0] = aux_pos_boxes[:, 0] + aux_pos_boxes[:, 2] / 2
                aux_pos_boxes[:, 1] = aux_pos_boxes[:, 1] + aux_pos_boxes[:, 3] / 2
                aux_pos_embed = gen_sineembed_for_position(
                    aux_pos_boxes.unsqueeze(0), 2880 // 4
                )
                out_box_aux_feat = out_box_aux_feat + aux_pos_embed
                
                # Only apply VT position embedding in these combos:
                # For *_no_vt_pos: skip vt feature position embedding
                if self.region_feature_combination in ['concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                    pass
                else:
                    # VT region: positional embedding using vt_boxes
                    vt_pos_boxes = vt_boxes[0].to(out_box_vt_feat.dtype)  # (N, 4)
                    vt_max_height = max([feature.shape[2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[3] for feature in vt_multi_level_features])
                    vt_original_img_width = vt_max_width / self.vision_tower_spatial_scale
                    vt_original_img_height = vt_max_height / self.vision_tower_spatial_scale

                    vt_pos_boxes[:, [0, 2]] = vt_pos_boxes[:, [0, 2]] / vt_original_img_width
                    vt_pos_boxes[:, [1, 3]] = vt_pos_boxes[:, [1, 3]] / vt_original_img_height
                    vt_pos_boxes[:, 2] = vt_pos_boxes[:, 2] - vt_pos_boxes[:, 0]
                    vt_pos_boxes[:, 3] = vt_pos_boxes[:, 3] - vt_pos_boxes[:, 1]
                    vt_pos_boxes[:, 0] = vt_pos_boxes[:, 0] + vt_pos_boxes[:, 2] / 2
                    vt_pos_boxes[:, 1] = vt_pos_boxes[:, 1] + vt_pos_boxes[:, 3] / 2
                    vt_pos_embed = gen_sineembed_for_position(
                        vt_pos_boxes.unsqueeze(0), vt_dim // 4
                    )
                    out_box_vt_feat = out_box_vt_feat + vt_pos_embed

                # Merge aux and vt region features (by cat or mean)
                if self.region_feature_combination in ['concat_sep_pos', 'concat_sep_no_vt_pos']:
                    out_box_feat = torch.cat([out_box_aux_feat, out_box_vt_feat], dim=-1)
                elif self.region_feature_combination in ['mean_sep_pos', 'mean_sep_no_vt_pos']:
                    out_box_feat = (out_box_aux_feat + out_box_vt_feat) / 2
                

        # If enabled: add single positional embedding (bbox-based, not separate for each region type)
        if self.apply_position_embedding and self.region_feature_combination not in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
            # Only apply if position embedding strategy matches
            apply_bbox_pos_embed = (self.pos_embedding_strategy == 'bbox_based' or self.pos_embedding_strategy == 'hybrid')
            
            if apply_bbox_pos_embed:
                # Use vt_boxes unless not enabled or configured otherwise
                if self.use_vision_tower_region_feature and vt_boxes is not None and self.region_feature_combination not in ['concat_aux_pos', 'mean_aux_pos']:
                    pos_boxes = vt_boxes[0]  # (N, 4)
                    vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                    vt_spatial_scale = self.vision_tower_spatial_scale
                    original_img_width = vt_max_width / vt_spatial_scale
                    original_img_height = vt_max_height / vt_spatial_scale
                else:
                    max_width = max([feature.shape[3] for feature in aux_multi_level_features])
                    max_height = max([feature.shape[2] for feature in aux_multi_level_features])
                    pos_boxes = aux_boxes[0]  # (N, 4)
                    original_img_width = max_width / self.aux_vision_tower_spatial_scale
                    original_img_height = max_height / self.aux_vision_tower_spatial_scale
                
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # Convert box to center format
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                pos_embed = gen_sineembed_for_position(
                    pos_boxes.unsqueeze(0), self.region_feature_dim // 4
                )
                out_box_feat = out_box_feat + pos_embed

        return out_box_feat