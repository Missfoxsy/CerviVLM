"""注意力实现选择（避免在无 flash-attn 的超算上硬编码 FlashAttention2）。"""

import os

_VALID = frozenset({"sdpa", "eager", "flash_attention_2"})


def resolve_attn_implementation(default: str = "sdpa") -> str:
    """
    由环境变量 ``VLM_FO1_ATTN_IMPLEMENTATION`` 选择实现。

    - ``sdpa``：PyTorch ``scaled_dot_product_attention``（推荐：无需安装 flash-attn 包）
    - ``eager``：标准注意力（最保守，略慢）
    - ``flash_attention_2``：需安装 flash-attn 且环境支持

    未设置或非法值时使用 ``default``（默认 ``sdpa``）。
    """
    raw = os.environ.get("VLM_FO1_ATTN_IMPLEMENTATION", default).strip().lower()
    return raw if raw in _VALID else default
