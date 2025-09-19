"""Triton-powered linear algebra helpers."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - executed when Triton is unavailable
    triton = None
    tl = None
else:

    @triton.jit
    def _matmul_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
            w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)

            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(x, w)

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        out_ptrs = out_ptr + (offs_m[:, None] * stride_ym) + (offs_n[None, :] * stride_yn)
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def triton_matmul(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Matrix multiply helper using the custom Triton kernel."""
    if triton is None or not x.is_cuda or not weight.is_cuda:
        return F.linear(x, weight, bias)

    x = _ensure_contiguous(x)
    weight = _ensure_contiguous(weight)
    if bias is not None:
        bias = _ensure_contiguous(bias)

    m, k = x.shape
    n = weight.shape[0]

    out = torch.empty((m, n), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _matmul_kernel[grid](  # type: ignore[index]
        x,
        weight,
        bias if bias is not None else weight,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.to(x.dtype)


class TritonLinear(torch.nn.Module):
    """Linear layer leveraging the Triton matrix multiply kernel."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        limit = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -limit, limit)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_matrix = x.reshape(-1, x.shape[-1])
        out = triton_matmul(x_matrix, self.weight, self.bias)
        return out.reshape(*original_shape, self.out_features)
