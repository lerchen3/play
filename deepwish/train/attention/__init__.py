"""Expose Triton kernels when available while keeping CPU-only tests importable."""

from __future__ import annotations

import importlib
import warnings

__all__ = []

_KERNEL_MODULES = [
    "casual_fwd",
    "casual_bwd",
    "select_fwd",
    "select_bwd",
]

for _name in _KERNEL_MODULES:
    try:
        _module = importlib.import_module(f".{_name}", __name__)
    except ImportError as exc:  # pragma: no cover - depends on Triton availability
        warnings.warn(
            f"train.attention: skipped loading Triton kernel '{_name}' ({exc}). "
            "PyTorch fallbacks will be used instead.",
            RuntimeWarning,
        )
        continue
    globals().update({k: getattr(_module, k) for k in getattr(_module, "__all__", [])})
    __all__.extend(getattr(_module, "__all__", []))

try:  # Re-export BaseAttention defined in train/base_attention.py
    from ..base_attention import BaseAttention as BaseAttention  # type: ignore
except ImportError:  # pragma: no cover - should not happen in normal execution
    BaseAttention = None  # type: ignore
else:
    __all__.append("BaseAttention")

