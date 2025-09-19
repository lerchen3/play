"""Triton implementation of the recurrent gated delta rule."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - executed when Triton is unavailable
    triton = None
    tl = None
else:

    @triton.jit
    def _gated_delta_step(
        q_ptr,
        k_ptr,
        v_ptr,
        decay_ptr,
        beta_ptr,
        state_ptr,
        out_ptr,
        stride_qm,
        stride_qd,
        stride_km,
        stride_kd,
        stride_vm,
        stride_vd,
        stride_state_m,
        stride_state_k,
        stride_state_v,
        stride_out_m,
        stride_out_d,
        stride_decay,
        stride_beta,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid = tl.program_id(0)

        q_row = q_ptr + pid * stride_qm
        k_row = k_ptr + pid * stride_km
        v_row = v_ptr + pid * stride_vm
        state_row = state_ptr + pid * stride_state_m
        out_row = out_ptr + pid * stride_out_m

        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        q = tl.load(q_row + offs_k * stride_qd, mask=offs_k < D_K, other=0.0).to(tl.float32)
        k = tl.load(k_row + offs_k * stride_kd, mask=offs_k < D_K, other=0.0).to(tl.float32)
        v = tl.load(v_row + offs_v * stride_vd, mask=offs_v < D_V, other=0.0).to(tl.float32)
        decay = tl.load(decay_ptr + pid * stride_decay).to(tl.float32)
        beta = tl.load(beta_ptr + pid * stride_beta).to(tl.float32)

        state_offsets = offs_k[:, None] * stride_state_k + offs_v[None, :] * stride_state_v
        mask_state = (offs_k[:, None] < D_K) & (offs_v[None, :] < D_V)
        state = tl.load(state_row + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

        state = state * decay
        kv_mem = tl.sum(state * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], axis=0)

        tl.store(state_row + state_offsets, state, mask=mask_state)
        tl.store(out_row + offs_v * stride_out_d, out, mask=offs_v < D_V)


def _torch_gated_delta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, d_k = key.shape
    d_v = value.shape[-1]
    state_cur = state.to(torch.float32)
    outputs = []
    for t in range(seq_len):
        decay_t = decay[:, t].to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        key_t = key[:, t].to(torch.float32).unsqueeze(-1)  # (B, H, D_K, 1)
        value_t = value[:, t].to(torch.float32)  # (B, H, D_V)
        beta_t = beta[:, t].to(torch.float32).unsqueeze(-1)  # (B, H, 1)
        query_t = query[:, t].to(torch.float32).unsqueeze(-1)  # (B, H, D_K, 1)

        state_cur = state_cur * decay_t
        kv_mem = (state_cur * key_t).sum(dim=2)  # (B, H, D_V)
        delta = (value_t - kv_mem) * beta_t  # (B, H, D_V)
        state_cur = state_cur + key_t * delta.unsqueeze(-2)
        out_t = (state_cur * query_t).sum(dim=2)  # (B, H, D_V)
        outputs.append(out_t)

    out_stack = torch.stack(outputs, dim=1)  # (B, S, H, D_V)
    return out_stack, state_cur


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies the recurrent gated delta rule with an optional persistent state."""
    batch, seq_len, num_heads, d_k = key.shape
    d_v = value.shape[-1]

    if state is None:
        state = torch.zeros(batch, num_heads, d_k, d_v, dtype=torch.float32, device=query.device)
    else:
        state = state.to(torch.float32)

    if triton is None or not query.is_cuda:
        outputs, final_state = _torch_gated_delta(query, key, value, decay, beta, state)
        return outputs.to(query.dtype), final_state

    outputs = torch.empty(batch, seq_len, num_heads, d_v, dtype=torch.float32, device=query.device)
    state_view = state.view(-1, d_k, d_v).contiguous()
    block_k = 1 << (d_k - 1).bit_length()
    block_v = 1 << (d_v - 1).bit_length()

    for t in range(seq_len):
        q_step = query[:, t].reshape(-1, d_k).contiguous().to(torch.float32)
        k_step = key[:, t].reshape(-1, d_k).contiguous().to(torch.float32)
        v_step = value[:, t].reshape(-1, d_v).contiguous().to(torch.float32)
        decay_step = decay[:, t].reshape(-1).contiguous().to(torch.float32)
        beta_step = beta[:, t].reshape(-1).contiguous().to(torch.float32)
        out_step = torch.empty(q_step.shape[0], d_v, device=query.device, dtype=torch.float32)

        grid = (q_step.shape[0],)
        _gated_delta_step[grid](
            q_step,
            k_step,
            v_step,
            decay_step,
            beta_step,
            state_view,
            out_step,
            q_step.stride(0),
            q_step.stride(1),
            k_step.stride(0),
            k_step.stride(1),
            v_step.stride(0),
            v_step.stride(1),
            state_view.stride(0),
            state_view.stride(1),
            state_view.stride(2),
            out_step.stride(0),
            out_step.stride(1),
            decay_step.stride(0),
            beta_step.stride(0),
            D_K=d_k,
            D_V=d_v,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
        )

        outputs[:, t] = out_step.view(batch, num_heads, d_v)

    return outputs.to(query.dtype), state_view.view_as(state)


def gated_delta_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token update for the gated delta rule (autoregressive inference)."""

    if query.ndim != 3:
        raise ValueError("query must have shape (batch, heads, d_k)")

    batch, num_heads, d_k = query.shape
    d_v = value.shape[-1]

    if state is None:
        state = torch.zeros(batch, num_heads, d_k, d_v, dtype=torch.float32, device=query.device)
    else:
        state = state.to(torch.float32)

    if triton is None or not query.is_cuda:
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        v = value.unsqueeze(1)
        decay_step = decay.unsqueeze(1)
        beta_step = beta.unsqueeze(1)
        outputs, new_state = _torch_gated_delta(q, k, v, decay_step, beta_step, state)
        return outputs[:, 0].to(query.dtype), new_state

    q_step = query.contiguous().to(torch.float32).view(-1, d_k)
    k_step = key.contiguous().to(torch.float32).view(-1, d_k)
    v_step = value.contiguous().to(torch.float32).view(-1, d_v)
    decay_step = decay.contiguous().to(torch.float32).view(-1)
    beta_step = beta.contiguous().to(torch.float32).view(-1)

    state_view = state.view(-1, d_k, d_v).contiguous()
    out_step = torch.empty_like(v_step)

    block_k = 1 << (d_k - 1).bit_length()
    block_v = 1 << (d_v - 1).bit_length()

    grid = (q_step.shape[0],)
    _gated_delta_step[grid](
        q_step,
        k_step,
        v_step,
        decay_step,
        beta_step,
        state_view,
        out_step,
        q_step.stride(0),
        q_step.stride(1),
        k_step.stride(0),
        k_step.stride(1),
        v_step.stride(0),
        v_step.stride(1),
        state_view.stride(0),
        state_view.stride(1),
        state_view.stride(2),
        out_step.stride(0),
        out_step.stride(1),
        decay_step.stride(0),
        beta_step.stride(0),
        D_K=d_k,
        D_V=d_v,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )

    outputs = out_step.view(batch, num_heads, d_v)
    return outputs.to(query.dtype), state_view.view_as(state)
