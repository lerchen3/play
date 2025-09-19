
"""Benchmark NSA component Triton kernels (forward/backward) against PyTorch baselines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.attention.casual import attention as triton_causal
from train.attention.select import select_attention as triton_select
from train.attention.topk import topk_indices as triton_topk


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class BenchResult:
    name: str
    torch_fwd: float
    triton_fwd: float
    torch_bwd: Optional[float] = None
    triton_bwd: Optional[float] = None

    @property
    def fwd_ratio(self) -> float:
        return self.triton_fwd / self.torch_fwd if self.torch_fwd else float("inf")

    @property
    def bwd_ratio(self) -> Optional[float]:
        if self.torch_bwd is None or self.triton_bwd is None or self.torch_bwd == 0.0:
            return None
        return self.triton_bwd / self.torch_bwd


def _time_loop(fn: Callable[[], None], iters: int = 10) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _torch_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, window_size: int = 0) -> torch.Tensor:
    B, H, T, D = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    base_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    if window_size and window_size > 0:
        idx = torch.arange(T, device=q.device)
        within = (idx[:, None] - idx[None, :]) < window_size
        mask = base_mask & within
    else:
        mask = base_mask
    scores = scores.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def benchmark_causal(window_size: int, B: int = 2, H: int = 8, T: int = 1024, D: int = 64, dtype: torch.dtype = torch.float16,
                     iters: int = 10) -> BenchResult:
    torch.manual_seed(0)
    q_base = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    k_base = torch.randn_like(q_base)
    v_base = torch.randn_like(q_base)
    sm_scale = 1.0 / math.sqrt(D)

    def run_triton(for_backward: bool = False):
        q = q_base.detach().clone().requires_grad_(for_backward)
        k = k_base.detach().clone().requires_grad_(for_backward)
        v = v_base.detach().clone().requires_grad_(for_backward)
        out = triton_causal(q, k, v, sm_scale, window_size)
        if for_backward:
            out.sum().backward()

    def run_torch(for_backward: bool = False):
        q = q_base.detach().clone().requires_grad_(for_backward)
        k = k_base.detach().clone().requires_grad_(for_backward)
        v = v_base.detach().clone().requires_grad_(for_backward)
        out = _torch_causal(q, k, v, sm_scale, window_size)
        if for_backward:
            out.sum().backward()

    # Warm-up (compilation & cache effects)
    run_triton(False)
    run_torch(False)
    run_triton(True)
    run_torch(True)

    fwd_triton = _time_loop(lambda: run_triton(False), iters)
    fwd_torch = _time_loop(lambda: run_torch(False), iters)
    bwd_triton = _time_loop(lambda: run_triton(True), iters)
    bwd_torch = _time_loop(lambda: run_torch(True), max(1, iters // 2))  # torch baseline is slow; fewer iters ok
    return BenchResult(
        name=f"casual_attn(window={window_size})",
        torch_fwd=fwd_torch,
        triton_fwd=fwd_triton,
        torch_bwd=bwd_torch,
        triton_bwd=bwd_triton,
    )


def _build_select_inputs(B: int, G: int, T: int, D: int, N_ctx: int, cmp_blk_size: int, Kmax: int,
                          dtype: torch.dtype = torch.float16):
    q = torch.randn(B, G, T, D, device='cuda', dtype=dtype)
    k_full = torch.randn(B, G, N_ctx, D, device='cuda', dtype=dtype)
    v_full = torch.randn_like(k_full)
    max_block = max(1, N_ctx // cmp_blk_size)
    block_idx = torch.randint(0, max_block, (B, G, T, Kmax), device='cuda', dtype=torch.int32)
    block_count = torch.randint(1, Kmax + 1, (B, G, T), device='cuda', dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(D)
    return q, k_full, v_full, block_idx, block_count, sm_scale


def _torch_select(q, k_full, v_full, sm_scale, block_idx, block_count, cmp_blk_size):
    """Vectorized PyTorch reference that mirrors block selection logic."""
    B, G, T, D = q.shape
    _, _, N_ctx, _ = k_full.shape
    device = q.device
    Kmax = block_idx.shape[-1]

    block_idx = block_idx.clamp_min(0)
    max_block = max(1, (N_ctx + cmp_blk_size - 1) // cmp_blk_size)
    block_idx = block_idx.clamp_max(max_block - 1)

    block_offsets = torch.arange(cmp_blk_size, device=device).view(1, 1, 1, 1, cmp_blk_size)
    token_indices = block_idx.unsqueeze(-1) * cmp_blk_size + block_offsets
    token_indices = token_indices.clamp_max(N_ctx - 1)

    valid_blocks = torch.arange(Kmax, device=device).view(1, 1, 1, Kmax)
    valid_blocks = valid_blocks < block_count.unsqueeze(-1)
    token_mask = valid_blocks.unsqueeze(-1).expand_as(token_indices)

    token_indices_flat = token_indices.reshape(B, G, T, -1)
    token_mask_flat = token_mask.reshape(B, G, T, -1)

    k_full_exp = k_full.unsqueeze(2).expand(B, G, T, N_ctx, D)
    gather_idx = token_indices_flat.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    gathered_k = torch.gather(k_full_exp, 3, gather_idx)
    gathered_v = torch.gather(v_full.unsqueeze(2).expand(B, G, T, N_ctx, D), 3, gather_idx)

    gathered_k = gathered_k.reshape(B, G, T, -1, D)
    gathered_v = gathered_v.reshape(B, G, T, -1, D)

    scores = torch.sum(q.unsqueeze(3) * gathered_k, dim=-1) * sm_scale
    scores = scores.masked_fill(~token_mask_flat, float('-inf'))
    probs = torch.softmax(scores.float(), dim=-1).to(gathered_v.dtype)
    probs = torch.nan_to_num(probs)

    out = torch.sum(probs.unsqueeze(-1) * gathered_v, dim=3)
    return out


def benchmark_select(B: int = 2, G: int = 4, T: int = 256, D: int = 32, N_ctx: int = 1024,
                     cmp_blk_size: int = 32, Kmax: int = 12, dtype: torch.dtype = torch.float16,
                     iters: int = 5) -> BenchResult:
    torch.manual_seed(0)
    inputs = _build_select_inputs(B, G, T, D, N_ctx, cmp_blk_size, Kmax, dtype)
    q_base, k_full_base, v_full_base, block_idx_base, block_count_base, sm_scale = inputs

    def run_triton(for_backward: bool = False):
        q = q_base.detach().clone().requires_grad_(for_backward)
        k = k_full_base.detach().clone().requires_grad_(for_backward)
        v = v_full_base.detach().clone().requires_grad_(for_backward)
        block_idx = block_idx_base.clone()
        block_count = block_count_base.clone()
        out = triton_select(q, k, v, sm_scale, block_idx, block_count, cmp_blk_size)
        if for_backward:
            out.sum().backward()

    def run_torch(for_backward: bool = False):
        q = q_base.detach().clone().requires_grad_(for_backward)
        k = k_full_base.detach().clone().requires_grad_(for_backward)
        v = v_full_base.detach().clone().requires_grad_(for_backward)
        block_idx = block_idx_base.clone()
        block_count = block_count_base.clone()
        out = _torch_select(q, k, v, sm_scale, block_idx, block_count, cmp_blk_size)
        if for_backward:
            out.sum().backward()

    run_triton(False)
    run_torch(False)
    run_triton(True)
    run_torch(True)

    fwd_triton = _time_loop(lambda: run_triton(False), iters)
    fwd_torch = _time_loop(lambda: run_torch(False), max(1, iters // 2))
    bwd_triton = _time_loop(lambda: run_triton(True), iters)
    bwd_torch = _time_loop(lambda: run_torch(True), 1)
    return BenchResult(
        name="select_attention",
        torch_fwd=fwd_torch,
        triton_fwd=fwd_triton,
        torch_bwd=bwd_torch,
        triton_bwd=bwd_triton,
    )


def benchmark_topk(B: int = 2, G: int = 4, T: int = 256, D: int = 32, N_ctx: int = 1024,
                   top_k: int = 16, dtype: torch.dtype = torch.float16, iters: int = 10) -> BenchResult:
    torch.manual_seed(0)
    q_group = torch.randn(B, G, T, D, device='cuda', dtype=dtype)
    k_full = torch.randn(B, G, N_ctx, D, device='cuda', dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)
    row_max = torch.arange(T, device='cuda').view(1, 1, T).repeat(B, G, 1)

    def run_triton():
        q = q_group.detach().clone()
        k = k_full.detach().clone()
        rm = row_max.clone()
        triton_topk(q, k, sm_scale, top_k, rm)

    def run_torch():
        q = q_group.detach().clone().float()
        k = k_full.detach().clone().float()
        rm = row_max.clone()
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        mask = torch.arange(N_ctx, device=q.device)
        mask = mask.view(1, 1, 1, N_ctx).expand_as(scores)
        cap = rm.view(B, G, T, 1)
        scores = scores.masked_fill(mask > cap, float('-inf'))
        torch.topk(scores, top_k, dim=-1)

    # Warm-up to exclude compilation overhead
    run_triton()
    run_torch()
    fwd_triton = _time_loop(run_triton, iters)
    fwd_torch = _time_loop(run_torch, max(1, iters // 2))
    return BenchResult(
        name="topk_indices",
        torch_fwd=fwd_torch,
        triton_fwd=fwd_triton,
    )


def main():
    results = []
    results.append(benchmark_causal(window_size=0, T=1024))
    results.append(benchmark_causal(window_size=128, T=1024))
    results.append(benchmark_select())
    results.append(benchmark_topk())

    print("NSA Component Benchmarks (times in ms, lower is better)")
    print("name, torch_fwd, triton_fwd, ratio_fwd, torch_bwd, triton_bwd, ratio_bwd")
    for r in results:
        bwd_ratio = r.bwd_ratio
        print(
            f"{r.name}, {r.torch_fwd:.3f}, {r.triton_fwd:.3f}, {r.fwd_ratio:.2f}, "
            f"{(r.torch_bwd if r.torch_bwd is not None else float('nan')):.3f}, "
            f"{(r.triton_bwd if r.triton_bwd is not None else float('nan')):.3f}, "
            f"{(bwd_ratio if bwd_ratio is not None else float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
