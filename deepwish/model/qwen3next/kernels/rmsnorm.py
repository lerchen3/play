"""RMSNorm kernels and modules."""
from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - executed when Triton is unavailable
    triton = None
    tl = None
else:

    @triton.jit
    def _zero_centered_rmsnorm_fwd(
        x_ptr,
        weight_ptr,
        out_ptr,
        eps,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row_ptr = x_ptr + row * stride_xm
        out_row_ptr = out_ptr + row * stride_ym

        accum = 0.0
        for col_start in range(0, N, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            vals = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
            accum += tl.sum(vals * vals, axis=0)

        mean = accum / N
        scale = tl.rsqrt(mean + eps)

        for col_start in range(0, N, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            vals = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
            weights = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out = vals * scale * (1.0 + weights)
            tl.store(out_row_ptr + cols * stride_yn, out, mask=mask)


    @triton.jit
    def _gated_rmsnorm_fwd(
        x_ptr,
        gate_ptr,
        weight_ptr,
        out_ptr,
        eps,
        stride_xm,
        stride_xn,
        stride_gm,
        stride_gn,
        stride_ym,
        stride_yn,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row_ptr = x_ptr + row * stride_xm
        gate_row_ptr = gate_ptr + row * stride_gm
        out_row_ptr = out_ptr + row * stride_ym

        accum = 0.0
        for col_start in range(0, N, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            vals = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
            accum += tl.sum(vals * vals, axis=0)

        mean = accum / N
        scale = tl.rsqrt(mean + eps)

        for col_start in range(0, N, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            vals = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
            weights = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            gates = tl.load(gate_row_ptr + cols * stride_gn, mask=mask, other=0.0).to(tl.float32)
            silu = gates * tl.sigmoid(gates)
            out = vals * scale * weights * silu
            tl.store(out_row_ptr + cols * stride_yn, out, mask=mask)


if triton is not None:  # pragma: no cover - CUDA path requires Triton

    def _launch_zero_centered(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_mat = x.reshape(-1, x.shape[-1])
        out = torch.empty_like(x_mat, dtype=torch.float32)
        BLOCK_SIZE = min(256, 1 << (x.shape[-1] - 1).bit_length())
        grid = (x_mat.shape[0],)
        _zero_centered_rmsnorm_fwd[grid](
            x_mat,
            weight,
            out,
            eps,
            x_mat.stride(0),
            x_mat.stride(1),
            out.stride(0),
            out.stride(1),
            x.shape[-1],
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out.reshape_as(x_mat)


    def _launch_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_mat = x.reshape(-1, x.shape[-1])
        gate_mat = gate.reshape(-1, gate.shape[-1])
        out = torch.empty_like(x_mat, dtype=torch.float32)
        BLOCK_SIZE = min(256, 1 << (x.shape[-1] - 1).bit_length())
        grid = (x_mat.shape[0],)
        _gated_rmsnorm_fwd[grid](
            x_mat,
            gate_mat,
            weight,
            out,
            eps,
            x_mat.stride(0),
            x_mat.stride(1),
            gate_mat.stride(0),
            gate_mat.stride(1),
            out.stride(0),
            out.stride(1),
            x.shape[-1],
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out.reshape_as(x_mat)


class ZeroCenteredRMSNorm(torch.nn.Module):
    """Implementation of the zero-centered RMSNorm variant used in Qwen3-Next."""

    def __init__(self, dim: int, eps: float = 1e-6, input_weight_decay: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.input_weight_decay = input_weight_decay
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        if self.input_weight_decay != 0.0:
            values = values * (1.0 - self.input_weight_decay)
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self._normalize(hidden_states)
        normed = normed * self.weight.float()
        return normed.to(hidden_states.dtype)


class GatedRMSNorm(torch.nn.Module):
    """RMS normalization with a learned gate multiplier."""

    def __init__(self, dim: int, eps: float = 1e-6, input_weight_decay: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.input_weight_decay = input_weight_decay
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        if self.input_weight_decay != 0.0:
            values = values * (1.0 - self.input_weight_decay)
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.weight.float()
        normalized = normalized * torch.nn.functional.silu(gate.float())
        return normalized.to(hidden_states.dtype)
