
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@dataclass(frozen=True)
class _KernelConfig:
    block_b: int
    block_v: int
    block_d: int
    group_b: int


def _select_block_sizes(B: int, V: int, D: int, block_size: int, block_k: int) -> _KernelConfig:
    """Heuristically pick tiling dimensions that balance occupancy and reuse."""
    block_b_candidates = [128, 96, 64, 48, 32, 16]
    block_v_candidates = [256, 192, 128, 96, 64, 48, 32]
    block_d_candidates = [128, 96, 64, 48, 32, 24, 16]

    def _choose(candidates: list[int], hint: int, limit: int, minimum: int) -> int:
        hint = max(hint, minimum)
        limit = max(limit, minimum)
        for value in candidates:
            if value <= hint and value <= limit:
                return value
        return minimum

    block_b = _choose(block_b_candidates, max(block_size, 16), B if B > 0 else 1, 16)
    block_v = _choose(block_v_candidates, max(block_size * 2, 32), V if V > 0 else 1, 32)
    block_d = _choose(block_d_candidates, max(block_k, 16), D if D > 0 else 1, 16)
    group_b = 8 if B >= block_b * 8 else 1
    return _KernelConfig(block_b=block_b, block_v=block_v, block_d=block_d, group_b=group_b)


@triton.jit
def _indexed_neg_dot_forward_kernel(
    E_ptr,
    C_ptr,
    targets_ptr,
    out_ptr,
    B,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    targets = tl.load(targets_ptr + offs_b, mask=mask_b, other=0).to(tl.int32)

    accum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)

    e_ptrs = E_ptr + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C_ptr + (targets[:, None] * stride_cv + offs_d[None, :] * stride_cd)

    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e = tl.load(e_ptrs, mask=mask_b[:, None], other=0.0)
            c = tl.load(c_ptrs, mask=mask_b[:, None], other=0.0).to(e.dtype)
        else:
            mask_d = (offs_d + d_block * BLOCK_D) < D
            mask = mask_b[:, None] & mask_d[None, :]
            e = tl.load(e_ptrs, mask=mask, other=0.0)
            c = tl.load(c_ptrs, mask=mask, other=0.0).to(e.dtype)

        prod = (e * c).to(tl.float32)
        accum += tl.sum(prod, axis=1)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.store(out_ptr + offs_b, -accum, mask=mask_b)


_indexed_neg_dot_forward_kernel = triton.heuristics(
    {"EVEN_D": lambda meta: meta["D"] % meta["BLOCK_D"] == 0}
)(_indexed_neg_dot_forward_kernel)


@triton.jit
def _lse_forward_kernel(
    E_ptr,
    C_ptr,
    lse_ptr,
    locks_ptr,
    B,
    V,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_lse,
    num_locks,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)

    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = tl.minimum(num_pid_b - first_pid_b, GROUP_B)

    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_b = offs_b < B
    mask_v = offs_v < V

    offs_d = tl.arange(0, BLOCK_D)

    e_ptrs = E_ptr + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C_ptr + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e = tl.load(e_ptrs, mask=mask_b[:, None], other=0.0)
            c = tl.load(c_ptrs, mask=mask_v[None, :], other=0.0).to(e.dtype)
        else:
            current_d = d_block * BLOCK_D
            mask_e = mask_b[:, None] & ((offs_d[None, :] + current_d) < D)
            mask_c = mask_v[None, :] & ((offs_d[:, None] + current_d) < D)
            e = tl.load(e_ptrs, mask=mask_e, other=0.0)
            c = tl.load(c_ptrs, mask=mask_c, other=0.0).to(e.dtype)

        accum = tl.dot(e, c, accum, input_precision="ieee")

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    mask = mask_b[:, None] & mask_v[None, :]
    logits = tl.where(mask, accum, -float("inf"))

    tile_max = tl.max(logits, axis=1)
    logits_shift = logits - tile_max[:, None]
    logits_shift = tl.where(mask, logits_shift, -float("inf"))
    tile_sum = tl.sum(tl.exp(logits_shift), axis=1)
    tile_valid = mask_b & (tile_sum > 0)
    tile_lse = tile_max + tl.log(tile_sum)
    tile_lse = tl.where(tile_valid, tile_lse, -float("inf"))

    locks = locks_ptr
    lock_idx = pid_b % num_locks
    lock_ptr = locks + lock_idx

    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    lse_ptrs = lse_ptr + offs_b * stride_lse
    prev_lse = tl.load(lse_ptrs, mask=mask_b, other=-float("inf"))
    prev_is_inf = prev_lse == -float("inf")
    tile_is_inf = tile_lse == -float("inf")

    max_val = tl.maximum(prev_lse, tile_lse)
    exp_prev = tl.where(prev_is_inf, 0.0, tl.exp(prev_lse - max_val))
    exp_tile = tl.where(tile_is_inf, 0.0, tl.exp(tile_lse - max_val))
    combined = max_val + tl.log(exp_prev + exp_tile)
    combined = tl.where(tile_is_inf, prev_lse, combined)
    combined = tl.where(prev_is_inf, tile_lse, combined)
    combined = tl.where(tile_valid, combined, prev_lse)

    tl.store(lse_ptrs, combined, mask=mask_b)
    tl.atomic_xchg(lock_ptr, 0)


_lse_forward_kernel = triton.heuristics(
    {"EVEN_D": lambda meta: meta["D"] % meta["BLOCK_D"] == 0}
)(_lse_forward_kernel)


@triton.jit
def _cce_backward_kernel(
    E_ptr,
    C_ptr,
    targets_ptr,
    lse_ptr,
    grad_scale,
    dE_ptr,
    dC_ptr,
    B,
    V,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_tb,
    stride_deb,
    stride_ded,
    stride_dcv,
    stride_dcd,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)

    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = tl.minimum(num_pid_b - first_pid_b, GROUP_B)

    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_b = offs_b < B
    mask_v = offs_v < V

    offs_d = tl.arange(0, BLOCK_D)

    e_ptrs = E_ptr + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C_ptr + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        current_d = d_block * BLOCK_D
        mask_e = mask_b[:, None] & ((offs_d[None, :] + current_d) < D)
        mask_c = mask_v[None, :] & ((offs_d[:, None] + current_d) < D)

        e = tl.load(e_ptrs, mask=mask_e, other=0.0)
        c = tl.load(c_ptrs, mask=mask_c, other=0.0).to(e.dtype)
        accum = tl.dot(e, c, accum, input_precision="ieee")

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    mask = mask_b[:, None] & mask_v[None, :]
    logits = tl.where(mask, accum, -float("inf"))

    lse_vals = tl.load(lse_ptr + offs_b, mask=mask_b, other=0.0)

    logits_shift = logits - lse_vals[:, None]
    logits_shift = tl.where(mask, logits_shift, -float("inf"))
    probs = tl.exp(logits_shift)

    targets = tl.load(targets_ptr + offs_b * stride_tb, mask=mask_b, other=0).to(tl.int32)
    target_mask = mask & (offs_v[None, :] == targets[:, None])
    grad_logits = probs - tl.where(target_mask, 1.0, 0.0)
    grad_logits = grad_logits * grad_scale
    grad_logits = tl.where(mask, grad_logits, 0.0)

    e_back_ptrs = E_ptr + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_back_ptrs = C_ptr + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        current_d = d_block * BLOCK_D
        mask_e = mask_b[:, None] & ((offs_d[None, :] + current_d) < D)
        mask_c = mask_v[:, None] & ((offs_d[None, :] + current_d) < D)
        mask_c_load = mask_v[None, :] & ((offs_d[:, None] + current_d) < D)

        e = tl.load(e_back_ptrs, mask=mask_e, other=0.0).to(grad_logits.dtype)
        c = tl.load(c_back_ptrs, mask=mask_c_load, other=0.0).to(grad_logits.dtype)

        grad_e_tile = tl.dot(grad_logits, tl.trans(c), input_precision="ieee")
        grad_c_tile = tl.dot(tl.trans(grad_logits), e, input_precision="ieee")

        dE_ptrs_tile = dE_ptr + (
            offs_b[:, None] * stride_deb + (offs_d[None, :] + current_d) * stride_ded
        )
        dC_ptrs_tile = dC_ptr + (
            offs_v[:, None] * stride_dcv + (offs_d[None, :] + current_d) * stride_dcd
        )

        tl.atomic_add(dE_ptrs_tile, grad_e_tile, mask=mask_e)
        tl.atomic_add(dC_ptrs_tile, grad_c_tile, mask=mask_c)

        e_back_ptrs += BLOCK_D * stride_ed
        c_back_ptrs += BLOCK_D * stride_cd


def _ensure_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _infer_kernel_config(
    E: torch.Tensor, C: torch.Tensor, block_size: int, block_k: int
) -> _KernelConfig:
    B, D = E.shape
    V = C.size(0)
    return _select_block_sizes(B, V, D, block_size, block_k)


class _TritonCCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        E: torch.Tensor,
        C: torch.Tensor,
        targets: torch.Tensor,
        block_size: int = 128,
        block_k: int = 64,
    ) -> torch.Tensor:
        if E.ndim != 2 or C.ndim != 2:
            raise ValueError("E and C must be rank-2 tensors")
        if targets.ndim != 1:
            raise ValueError("targets must be a 1D tensor")
        if E.size(1) != C.size(1):
            raise ValueError("Embedding dimension mismatch between E and C")

        if not (E.is_cuda and C.is_cuda and targets.is_cuda):
            logits = (E @ C.t()).to(E.dtype)
            return F.cross_entropy(logits, targets, reduction="mean")

        E_ctg = _ensure_contiguous(E, "E")
        C_ctg = _ensure_contiguous(C, "C")
        targets_ctg = _ensure_contiguous(targets, "targets")

        B, D = E_ctg.shape
        V = C_ctg.size(0)
        cfg = _infer_kernel_config(E_ctg, C_ctg, block_size, block_k)

        lse = torch.full((B,), -float("inf"), dtype=torch.float32, device=E_ctg.device)
        neg_dot = torch.zeros((B,), dtype=torch.float32, device=E_ctg.device)
        num_locks = max(1, min(1024, (B + cfg.block_b - 1) // cfg.block_b))
        if num_locks == 0:
            num_locks = 1
        locks = torch.zeros((num_locks,), dtype=torch.uint32, device=E_ctg.device)

        def lse_grid(meta):
            return (
                triton.cdiv(B, meta["BLOCK_B"]) * triton.cdiv(V, meta["BLOCK_V"]),
            )

        _lse_forward_kernel[lse_grid](
            E_ctg,
            C_ctg,
            lse,
            locks,
            B,
            V,
            D,
            E_ctg.stride(0),
            E_ctg.stride(1),
            C_ctg.stride(0),
            C_ctg.stride(1),
            lse.stride(0),
            locks.size(0),
            BLOCK_B=cfg.block_b,
            BLOCK_V=cfg.block_v,
            BLOCK_D=cfg.block_d,
            GROUP_B=cfg.group_b,
            num_warps=8,
            num_stages=2,
        )

        def dot_grid(meta):
            return (
                triton.cdiv(B, meta["BLOCK_B"]),
            )

        _indexed_neg_dot_forward_kernel[dot_grid](
            E_ctg,
            C_ctg,
            targets_ctg,
            neg_dot,
            B,
            D,
            E_ctg.stride(0),
            E_ctg.stride(1),
            C_ctg.stride(0),
            C_ctg.stride(1),
            BLOCK_B=cfg.block_b,
            BLOCK_D=cfg.block_d,
            num_warps=8,
            num_stages=2,
        )

        loss = (lse + neg_dot).mean()

        ctx.save_for_backward(E_ctg, C_ctg, targets_ctg, lse)
        ctx.cfg = cfg
        ctx.shape = (B, V, D)

        return loss.to(E.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        E, C, targets, lse = ctx.saved_tensors
        B, V, D = ctx.shape
        cfg = ctx.cfg

        if grad_output.numel() != 1:
            raise ValueError("grad_output must be scalar")

        grad_scale = grad_output.item() / B

        dE = torch.zeros_like(E, dtype=torch.float32)
        dC = torch.zeros_like(C, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(B, meta["BLOCK_B"]) * triton.cdiv(V, meta["BLOCK_V"]),
            )

        _cce_backward_kernel[grid](
            E,
            C,
            targets,
            lse,
            grad_scale,
            dE,
            dC,
            B,
            V,
            D,
            E.stride(0),
            E.stride(1),
            C.stride(0),
            C.stride(1),
            targets.stride(0),
            dE.stride(0),
            dE.stride(1),
            dC.stride(0),
            dC.stride(1),
            BLOCK_B=cfg.block_b,
            BLOCK_V=cfg.block_v,
            BLOCK_D=cfg.block_d,
            GROUP_B=cfg.group_b,
            num_warps=4,
        )

        grad_E = dE.to(dtype=E.dtype) if ctx.needs_input_grad[0] else None
        grad_C = dC.to(dtype=C.dtype) if ctx.needs_input_grad[1] else None

        return grad_E, grad_C, None, None, None


def triton_cut_cross_entropy(
    E: torch.Tensor,
    C: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 128,
    block_k: int = 64,
) -> torch.Tensor:
    if not (E.is_cuda and C.is_cuda and targets.is_cuda):
        logits = (E @ C.t()).to(E.dtype)
        return F.cross_entropy(logits, targets, reduction="mean")
    return _TritonCCEFunction.apply(E, C, targets, block_size, block_k)


class TritonCCE(_TritonCCEFunction):
    """Alias to expose ``apply`` for external call-sites."""

    pass


__all__ = ["TritonCCE", "triton_cut_cross_entropy"]
