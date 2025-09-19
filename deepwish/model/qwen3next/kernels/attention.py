"""Scaled dot product attention powered by Triton matrix multiplies."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from model.decode_triton import single_query_attention

from .linear import triton_matmul
from train.attention.casual import attention as fa2_attention


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes scaled dot product attention using Triton kernels."""
    batch, seq_len, num_heads, head_dim = query.shape
    value_dim = value.shape[-1]

    if (
        seq_len == 1
        and single_query_attention is not None
        and query.device.type == "cuda"
        and not training
        and attention_mask is None
        and dropout_p == 0.0
    ):
        q_heads = query.permute(0, 2, 1, 3).contiguous()
        k_heads = key.permute(0, 2, 1, 3).contiguous()
        v_heads = value.permute(0, 2, 1, 3).contiguous()
        attn = single_query_attention(q_heads, k_heads, v_heads, scale)
        context = attn.permute(0, 2, 1, 3)
        return context.to(value.dtype), None
    q_flat = query.permute(0, 2, 1, 3).reshape(-1, seq_len, head_dim)
    k_flat = key.permute(0, 2, 1, 3).reshape(-1, seq_len, head_dim)
    v_flat = value.permute(0, 2, 1, 3).reshape(-1, seq_len, value_dim)

    # Decide whether to use the FA2 Triton kernel or fall back to the PyTorch reference path.
    supported_head_dims = {16, 32, 64, 128, 256}
    use_fa2 = (
        attention_mask is None
        and dropout_p == 0.0
        and query.device.type == "cuda"
        and head_dim in supported_head_dims
        and seq_len >= 32
    )

    if use_fa2:
        q_heads = query.permute(0, 2, 1, 3).contiguous()
        k_heads = key.permute(0, 2, 1, 3).contiguous()
        v_heads = value.permute(0, 2, 1, 3).contiguous()
        context = fa2_attention(q_heads, k_heads, v_heads, scale)
        context = context.permute(0, 2, 1, 3)
        return context.to(value.dtype), None

    # Fallback: explicit PyTorch implementation (supports masks/dropout/training).
    attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
    attn_scores = attn_scores.reshape(batch, num_heads, seq_len, seq_len)

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)

    attn_probs = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)

    attn_probs_flat = attn_probs.reshape(-1, seq_len, seq_len)
    context = torch.matmul(attn_probs_flat, v_flat)
    context = context.reshape(batch, num_heads, seq_len, value_dim)
    context = context.permute(0, 2, 1, 3)
    return context.to(value.dtype), attn_probs.to(value.dtype)
