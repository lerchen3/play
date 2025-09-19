"""Qwen3 model package exposing the CUDA-only Triton-backed implementation."""

from .model import Qwen3Model, triton_cce_loss
from .training import Qwen3TrainModel

__all__ = [
    "Qwen3Model",
    "Qwen3TrainModel",
    "triton_cce_loss",
]
