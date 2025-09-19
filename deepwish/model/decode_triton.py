"""Triton kernels optimized for single-token autoregressive decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _single_query_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    scale,
    stride_qm,
    stride_qd,
    stride_km,
    stride_kn,
    stride_kd,
    stride_vm,
    stride_vn,
    stride_vd,
    stride_outm,
    stride_outd,
    seq_len,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + pid * stride_qm + offs_d * stride_qd)
    q_dtype = q.dtype
    q = q.to(tl.float32)
    scale_f = tl.full((), scale, tl.float32)

    neg_inf = -float("inf")
    m_i = tl.full((), neg_inf, tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len

        k = tl.load(
            k_ptr + pid * stride_km + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + pid * stride_vm + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale_f
        scores = tl.where(mask, scores, neg_inf)

        block_max = tl.max(scores, axis=0)
        new_m = tl.maximum(m_i, block_max)
        exp_old = tl.exp(m_i - new_m)
        probs = tl.exp(scores - new_m)
        probs = tl.where(mask, probs, 0.0)
        exp_new = tl.sum(probs, axis=0)

        acc = exp_old * acc + tl.sum(probs[:, None] * v, axis=0)
        l_i = exp_old * l_i + exp_new
        m_i = new_m

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    out = acc * inv_l
    out = out.to(q_dtype)
    tl.store(out_ptr + pid * stride_outm + offs_d * stride_outd, out)


def single_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    block_size: int = 128,
) -> torch.Tensor:
    """Compute attention for a single autoregressive token using Triton."""
    if query.ndim != 4:
        raise ValueError("query must have shape (batch, heads, tokens, dim)")
    if query.shape[2] != 1:
        raise ValueError("single_query_attention expects a single token (tokens==1)")
    if query.device.type != "cuda" or key.device.type != "cuda" or value.device.type != "cuda":
        raise RuntimeError("single_query_attention requires CUDA tensors")

    batch, heads, tokens, dim = query.shape
    seq_len = key.shape[2]
    if seq_len != value.shape[2]:
        raise ValueError("key and value sequence lengths must match")

    q_flat = query.reshape(batch * heads, dim).contiguous()
    k_flat = key.reshape(batch * heads, seq_len, dim).contiguous()
    v_flat = value.reshape(batch * heads, seq_len, dim).contiguous()
    out_flat = torch.empty_like(q_flat)

    grid = (q_flat.shape[0],)
    _single_query_attention_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        scale,
        q_flat.stride(0),
        q_flat.stride(1),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        seq_len,
        BLOCK_N=block_size,
        HEAD_DIM=dim,
    )

    return out_flat.view(batch, heads, tokens, dim)


__all__ = ["single_query_attention"]
