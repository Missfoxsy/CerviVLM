"""Environment compatibility shims for broken optional CUDA packages.

Problems seen in the covt env (torch 2.11 + mismatched optional wheels):
1) flash_attn CUDA extension ABI mismatch / missing libc10 → blocks transformers import
2) bitsandbytes imports ``triton.ops`` which was removed in newer triton → blocks peft/Trainer
3) torchaudio ABI mismatch → ``is_torchaudio_available()`` is True then ``import torchaudio``
   crashes inside transformers.audio_utils

This module must be imported **before** ``transformers`` / ``peft``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_HIDDEN_PACKAGES: set[str] = set()


def _ensure_torch_lib_on_ld_path() -> None:
    """Help native extensions resolve libtorch / libc10 at import time."""
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isdir(torch_lib):
            return
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if torch_lib not in parts:
            os.environ["LD_LIBRARY_PATH"] = torch_lib + ((":" + cur) if cur else "")
    except Exception:
        pass


def _install_find_spec_hide(prefixes: list[str]) -> None:
    """Make ``importlib.util.find_spec`` report None for broken optional packages."""
    for p in prefixes:
        _HIDDEN_PACKAGES.add(p)

    # Idempotent: always wrap the current find_spec once with a shared filter.
    if getattr(_install_find_spec_hide, "_wrapped", False):
        return

    _orig_find_spec = importlib.util.find_spec

    def _find_spec(name, package=None):  # type: ignore[no-untyped-def]
        if isinstance(name, str):
            for prefix in _HIDDEN_PACKAGES:
                if name == prefix or name.startswith(prefix + "."):
                    return None
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _find_spec  # type: ignore[assignment]
    _install_find_spec_hide._wrapped = True  # type: ignore[attr-defined]


def _purge_modules(*prefixes: str) -> None:
    for key in list(sys.modules):
        for prefix in prefixes:
            if key == prefix or key.startswith(prefix + "."):
                sys.modules.pop(key, None)
                break


def _hide_broken_flash_attn() -> bool:
    """Return True if flash_attn was hidden due to a broken CUDA extension."""
    try:
        if importlib.util.find_spec("flash_attn") is None:
            return False
        import flash_attn_2_cuda  # noqa: F401
        return False
    except Exception as exc:  # noqa: BLE001
        _purge_modules("flash_attn", "flash_attn_2_cuda")
        _install_find_spec_hide(["flash_attn"])
        print(
            f"[env_compat] Broken flash_attn ({type(exc).__name__}: {exc}); "
            "hiding package → transformers falls back to SDPA/eager."
        )
        return True


def _hide_broken_torchaudio() -> bool:
    """Hide ABI-broken torchaudio so transformers skips the optional import."""
    try:
        if importlib.util.find_spec("torchaudio") is None:
            return False
        import torchaudio  # noqa: F401
        return False
    except Exception as exc:  # noqa: BLE001
        _purge_modules("torchaudio")
        _install_find_spec_hide(["torchaudio"])
        print(
            f"[env_compat] Broken torchaudio ({type(exc).__name__}: {exc}); "
            "hiding package → transformers skips audio backends."
        )
        return True


def _stub_missing_triton_ops() -> bool:
    """Provide minimal stubs if bitsandbytes expects removed ``triton.ops``."""
    try:
        import triton.ops  # noqa: F401
        return False
    except Exception:
        pass

    ops = types.ModuleType("triton.ops")
    matmul = types.ModuleType("triton.ops.matmul_perf_model")

    def early_config_prune(*_a, **_k):
        return []

    def estimate_matmul_time(*_a, **_k):
        return 0.0

    matmul.early_config_prune = early_config_prune
    matmul.estimate_matmul_time = estimate_matmul_time
    ops.matmul_perf_model = matmul
    sys.modules["triton.ops"] = ops
    sys.modules["triton.ops.matmul_perf_model"] = matmul
    print("[env_compat] Stubbed missing triton.ops for bitsandbytes/peft import.")
    return True


def apply_env_compat() -> dict:
    if getattr(apply_env_compat, "_done", False):
        return getattr(apply_env_compat, "_info", {})

    _ensure_torch_lib_on_ld_path()
    info = {
        "flash_attn_hidden": _hide_broken_flash_attn(),
        "torchaudio_hidden": _hide_broken_torchaudio(),
        "triton_ops_stubbed": _stub_missing_triton_ops(),
    }
    apply_env_compat._done = True  # type: ignore[attr-defined]
    apply_env_compat._info = info  # type: ignore[attr-defined]
    return info


# Backwards-compatible alias used by train/eval entrypoints.
def disable_broken_flash_attn() -> bool:
    return bool(apply_env_compat().get("flash_attn_hidden"))


apply_env_compat()
