"""Configuration objects for the custom Qwen3-Next implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import torch


def _build_layer_types(num_layers: int, pattern: Sequence[str]) -> Tuple[str, ...]:
    if not pattern:
        raise ValueError("Pattern used to build layer types cannot be empty")
    result = []
    idx = 0
    while len(result) < num_layers:
        result.append(pattern[idx % len(pattern)])
        idx += 1
    return tuple(result[:num_layers])


@dataclass(slots=True)
class Qwen3NextConfig:
    """Dataclass holding the hyper-parameters of the Qwen3-Next architecture."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_dropout: float = 0.0
    attention_bias: bool = False
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rms_norm_input_weight_decay: float = 0.2
    rope_theta: float = 1_000_0000.0
    partial_rotary_factor: float = 0.25
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    use_cache: bool = True
    dtype: Optional[torch.dtype] = torch.bfloat16
    mtp_depth: int = 0

    # Hybrid architecture parameters
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # MoE configuration
    num_experts: int = 512
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    decoder_sparse_step: int = 1
    mlp_only_layers: Tuple[int, ...] = field(default_factory=tuple)
    output_router_logits: bool = False
    router_aux_loss_coef: float = 1e-3

    # Layer layout overrides
    layer_types: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"got num_attention_heads={self.num_attention_heads}, num_key_value_heads={self.num_key_value_heads}"
            )
        if self.layer_types is None:
            # Repeat three linear layers followed by one full attention layer.
            pattern: Tuple[str, ...] = (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
            self.layer_types = _build_layer_types(self.num_hidden_layers, pattern)
        else:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    "layer_types must match num_hidden_layers; "
                    f"got {len(self.layer_types)} types for {self.num_hidden_layers} layers"
                )
            invalid = {t for t in self.layer_types if t not in {"linear_attention", "full_attention"}}
            if invalid:
                raise ValueError(f"Unsupported layer types found: {sorted(invalid)}")
        rotary = int(self.partial_rotary_factor * self.head_dim)
        if rotary <= 0 or rotary % 2:
            raise ValueError(
                "partial_rotary_factor must generate an even positive rotary dimension; "
                f"head_dim={self.head_dim}, partial_rotary_factor={self.partial_rotary_factor}"
            )
        if not (0.0 <= self.rms_norm_input_weight_decay < 1.0):
            raise ValueError(
                "rms_norm_input_weight_decay must be in [0, 1); "
                f"got {self.rms_norm_input_weight_decay}"
            )
        if self.mtp_depth < 0:
            raise ValueError(f"mtp_depth must be non-negative; got {self.mtp_depth}")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.partial_rotary_factor * self.head_dim)

    @property
    def device_dtype(self) -> torch.dtype:
        return self.dtype if self.dtype is not None else torch.get_default_dtype()

    def layers_with_moe(self) -> Tuple[int, ...]:
        if self.mlp_only_layers:
            keep = {int(idx) for idx in self.mlp_only_layers}
            return tuple(idx for idx in range(self.num_hidden_layers) if idx not in keep)
        if self.decoder_sparse_step <= 0:
            return tuple()
        return tuple(idx for idx in range(self.decoder_sparse_step - 1, self.num_hidden_layers, self.decoder_sparse_step))

    def layers_without_moe(self) -> Tuple[int, ...]:
        full = set(range(self.num_hidden_layers))
        return tuple(sorted(full.difference(self.layers_with_moe())))
