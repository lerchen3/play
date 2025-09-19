"""Benchmark Triton kernels against PyTorch baselines and write a Markdown report."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from model.cce import triton_cce_loss
from model.qwen3next.kernels import (
    GatedRMSNorm,
    TritonLinear,
    ZeroCenteredRMSNorm,
    gated_delta_rule,
    gated_delta_step,
    scaled_dot_product_attention,
    triton_matmul,
)
from model.qwen3next.kernels.deltanet import _torch_gated_delta  # type: ignore
from train.attention.nsa import NativeSparseAttention
from train.attention.casual import _attention as casual_attention
from train.attention.select import select_attention
from train.attention.topk import topk_indices


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class Benchmark:
    name: str
    torch_fwd_ms: float
    triton_fwd_ms: float
    torch_bwd_ms: Optional[float]
    triton_bwd_ms: Optional[float]

    @property
    def speedup(self) -> float:
        return self.torch_fwd_ms / self.triton_fwd_ms if self.triton_fwd_ms else float("inf")

    @property
    def backward_speedup(self) -> float:
        if self.torch_bwd_ms is None or self.triton_bwd_ms is None:
            return float("inf")
        if self.triton_bwd_ms == 0.0:
            return float("inf")
        return self.torch_bwd_ms / self.triton_bwd_ms


BackwardSetup = Callable[[], Tuple[Callable[[], None], Callable[[], None]]]

SetupReturn = Tuple[
    Callable[[], None],
    Callable[[], None],
    Optional[BackwardSetup],
    Optional[BackwardSetup],
    Dict[str, Any],
]


@dataclass
class SweepDefinition:
    name: str
    param_name: str
    values: List[int]
    setup: Callable[..., SetupReturn]
    warmup: int
    iters: int
    kwargs: Dict[str, Any]


@dataclass
class SweepResult:
    name: str
    param_name: str
    values: List[int]
    benchmarks: List[Benchmark]
    configs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_name": self.param_name,
            "values": self.values,
            "benchmarks": [asdict(b) for b in self.benchmarks],
            "configs": self.configs,
        }


def _time_callable(fn: Callable[[], None], label: str, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters
    return avg_ms


def _time_backward(
    prepare: BackwardSetup,
    label: str,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    for _ in range(warmup):
        forward, backward = prepare()
        forward()
        torch.cuda.synchronize()
        backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0

    for _ in range(iters):
        forward, backward = prepare()
        forward()
        torch.cuda.synchronize()
        start.record()
        backward()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    return total_ms / iters


def _default_sweeps() -> List[SweepDefinition]:
    return [
        SweepDefinition(
            name="TritonLinear",
            param_name="dim",
            values=[512, 1024, 2048, 4096],
            setup=_setup_linear,
            warmup=5,
            iters=10,
            kwargs={"batch": 64},
        ),
        SweepDefinition(
            name="triton_matmul",
            param_name="dim",
            values=[512, 1024, 2048, 4096],
            setup=_setup_matmul,
            warmup=3,
            iters=5,
            kwargs={},
        ),
        SweepDefinition(
            name="ZeroCenteredRMSNorm",
            param_name="dim",
            values=[512, 1024, 2048, 4096],
            setup=_setup_zero_centered_rms,
            warmup=3,
            iters=10,
            kwargs={"batch": 4096},
        ),
        SweepDefinition(
            name="GatedRMSNorm",
            param_name="dim",
            values=[512, 1024, 2048, 4096],
            setup=_setup_gated_rms,
            warmup=3,
            iters=10,
            kwargs={"batch": 4096},
        ),
        SweepDefinition(
            name="scaled_dot_product_attention",
            param_name="seq",
            values=[256, 512, 1024, 2048],
            setup=_setup_attention,
            warmup=3,
            iters=5,
            kwargs={"batch": 4, "heads": 16, "head_dim": 128},
        ),
        SweepDefinition(
            name="gated_delta_rule",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_delta_rule,
            warmup=1,
            iters=5,
            kwargs={"batch": 2, "heads": 4, "d_k": 16, "d_v": 32},
        ),
        SweepDefinition(
            name="gated_delta_step",
            param_name="d_model",
            values=[128, 256, 512, 1024],
            setup=_setup_delta_step,
            warmup=2,
            iters=20,
            kwargs={"batch": 2, "heads": 4},
        ),
        SweepDefinition(
            name="nsa_attention",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_nsa,
            warmup=2,
            iters=5,
            kwargs={"d_model": 128, "n_q_heads": 4, "n_kv_heads": 2, "d_head": 32, "dtype": torch.float16},
        ),
        SweepDefinition(
            name="nsa_compressed_attention",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_nsa_compressed_attention,
            warmup=2,
            iters=5,
            kwargs={"n_q_heads": 4, "n_kv_heads": 2, "d_head": 32},
        ),
        SweepDefinition(
            name="nsa_selected_attention",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_nsa_selected_attention,
            warmup=2,
            iters=5,
            kwargs={"n_q_heads": 4, "n_kv_heads": 2, "d_head": 32},
        ),
        SweepDefinition(
            name="nsa_sliding_attention",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_nsa_sliding_attention,
            warmup=2,
            iters=5,
            kwargs={"n_q_heads": 4, "n_kv_heads": 2, "d_head": 32, "window_size": 32},
        ),
        SweepDefinition(
            name="nsa_topk_selection",
            param_name="seq_len",
            values=[256, 512, 1024, 2048],
            setup=_setup_nsa_topk,
            warmup=2,
            iters=5,
            kwargs={"n_q_heads": 4, "n_kv_heads": 2, "d_head": 32},
        ),
        SweepDefinition(
            name="triton_cce_loss",
            param_name="vocab",
            values=[2048, 4096, 8192, 16384],
            setup=_setup_cce,
            warmup=1,
            iters=5,
            kwargs={"batch": 4, "seq": 1024, "dim": 1024, "block_size": 128},
        ),
    ]


def _benchmark_pair(
    name: str,
    torch_fwd: Callable[[], None],
    triton_fwd: Callable[[], None],
    torch_bwd: Optional[BackwardSetup] = None,
    triton_bwd: Optional[BackwardSetup] = None,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> Benchmark:
    # Pre-run once to trigger kernel compilation and ensure deterministic caches.
    torch_fwd()
    triton_fwd()
    torch.cuda.synchronize()
    torch_fwd_ms = _time_callable(torch_fwd, f"{name}_torch_fwd", warmup=warmup, iters=iters)
    triton_fwd_ms = _time_callable(triton_fwd, f"{name}_triton_fwd", warmup=warmup, iters=iters)

    torch_bwd_ms = None
    if torch_bwd is not None:
        warm_forward, warm_backward = torch_bwd()
        warm_forward()
        torch.cuda.synchronize()
        warm_backward()
        torch_bwd_ms = _time_backward(torch_bwd, f"{name}_torch_bwd", warmup=warmup, iters=iters)

    triton_bwd_ms = None
    if triton_bwd is not None:
        warm_forward, warm_backward = triton_bwd()
        warm_forward()
        torch.cuda.synchronize()
        warm_backward()
        triton_bwd_ms = _time_backward(triton_bwd, f"{name}_triton_bwd", warmup=warmup, iters=iters)

    return Benchmark(name, torch_fwd_ms, triton_fwd_ms, torch_bwd_ms, triton_bwd_ms)


def _run_sweep(device: torch.device, sweep: SweepDefinition) -> SweepResult:
    benchmarks: List[Benchmark] = []
    configs: List[Dict[str, Any]] = []
    for value in sweep.values:
        print(f"Benchmarking {sweep.name}: {sweep.param_name}={value}")
        torch_fwd, triton_fwd, torch_bwd, triton_bwd, cfg = sweep.setup(
            device, value, **sweep.kwargs
        )
        bench_raw = _benchmark_pair(
            f"{sweep.name}_{sweep.param_name}_{value}",
            torch_fwd,
            triton_fwd,
            torch_bwd,
            triton_bwd,
            warmup=sweep.warmup,
            iters=sweep.iters,
        )
        benchmarks.append(
            Benchmark(
                name=sweep.name,
                torch_fwd_ms=bench_raw.torch_fwd_ms,
                triton_fwd_ms=bench_raw.triton_fwd_ms,
                torch_bwd_ms=bench_raw.torch_bwd_ms,
                triton_bwd_ms=bench_raw.triton_bwd_ms,
            )
        )
        cfg_with_meta = {
            **cfg,
            sweep.param_name: value,
            "warmup": sweep.warmup,
            "iters": sweep.iters,
        }
        configs.append(cfg_with_meta)
    return SweepResult(
        name=sweep.name,
        param_name=sweep.param_name,
        values=sweep.values,
        benchmarks=benchmarks,
        configs=configs,
    )


def _power_of_two_floor(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _prepare_nsa_inputs(
    device: torch.device,
    seq_len: int,
    *,
    batch: int = 2,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    cmp_blk_size: int = 32,
    cmp_stride: int = 16,
    slc_top_n: int = 4,
    window_size: int = 32,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    torch.manual_seed(100 + seq_len)
    B = batch
    T = seq_len
    G = n_kv_heads
    Hg = n_q_heads // n_kv_heads
    scale = (d_head) ** -0.5

    q = torch.randn(B, n_q_heads, T, d_head, device=device, dtype=dtype)
    q_groups = q.view(B, G, Hg, T, d_head)
    q_group = q_groups.sum(dim=2).contiguous()
    q_slc = q_groups.mean(dim=2).contiguous()

    block_starts = list(range(0, max(1, T - cmp_blk_size + 1), cmp_stride))
    if 0 not in block_starts:
        block_starts = [0] + block_starts
    nb = len(block_starts)
    block_starts_tensor = torch.tensor(block_starts, device=device, dtype=torch.int32)

    token_positions = torch.arange(T, device=device, dtype=torch.int32)
    row_max_block = torch.bucketize(token_positions, block_starts_tensor, right=True) - 1
    row_max_block = row_max_block.clamp(min=0, max=nb - 1)
    row_max = row_max_block.view(1, 1, T).expand(B, G, T).contiguous()

    Kcmp_g = torch.randn(B, G, nb, d_head, device=device, dtype=dtype)
    Vcmp_g = torch.randn_like(Kcmp_g)

    topk_candidate = max(1, min(slc_top_n, nb))
    topk = min(_power_of_two_floor(topk_candidate), nb)

    scores = torch.matmul(
        q_group.to(torch.float32),
        Kcmp_g.transpose(-1, -2).to(torch.float32),
    ) * scale
    mask = torch.arange(nb, device=device).view(1, 1, 1, nb) > row_max.unsqueeze(-1)
    scores = scores.masked_fill(mask, float("-inf"))
    top_idx = torch.topk(scores, k=topk, dim=-1).indices.to(torch.int32)
    top_idx = torch.minimum(top_idx, row_max.unsqueeze(-1))

    forced_sink = torch.zeros(B, G, T, 1, device=device, dtype=torch.int32)
    last_two = torch.stack(
        [
            torch.full((B, G, T), max(0, nb - 2), device=device, dtype=torch.int32),
            torch.full((B, G, T), max(0, nb - 1), device=device, dtype=torch.int32),
        ],
        dim=-1,
    )
    sel_blocks_all = torch.cat([forced_sink, last_two, top_idx], dim=-1)

    blk_mask = torch.zeros(B, G, T, nb, dtype=torch.bool, device=device)
    blk_mask.scatter_(dim=3, index=sel_blocks_all, src=torch.ones_like(sel_blocks_all, dtype=torch.bool))
    block_count = blk_mask.sum(dim=-1).to(torch.int32)
    Kmax = min(nb, sel_blocks_all.shape[-1])
    idx_range = torch.arange(nb, device=device, dtype=torch.int32)
    idx_exp = idx_range.view(1, 1, 1, nb).expand_as(blk_mask)
    masked = torch.where(blk_mask, idx_exp, torch.full_like(idx_exp, nb))
    sorted_idx, _ = torch.sort(masked, dim=-1)
    block_idx = sorted_idx[..., :Kmax]
    block_idx = torch.where(block_idx == nb, torch.zeros_like(block_idx), block_idx)
    block_count = torch.minimum(block_count, torch.tensor(Kmax, device=device, dtype=torch.int32))

    Kslc_full = torch.randn(B, G, T, d_head, device=device, dtype=dtype)
    Vslc_full = torch.randn_like(Kslc_full)

    k_win_kv = torch.randn(B, T, G, d_head, device=device, dtype=dtype)
    v_win_kv = torch.randn_like(k_win_kv)
    k_win_full = k_win_kv.permute(0, 2, 1, 3).repeat_interleave(Hg, dim=1).contiguous()
    v_win_full = v_win_kv.permute(0, 2, 1, 3).repeat_interleave(Hg, dim=1).contiguous()
    q_win_full = q.view(B, n_q_heads, T, d_head).contiguous()

    return {
        "q_group": q_group,
        "q_slc": q_slc,
        "q_win_full": q_win_full,
        "Kcmp_g": Kcmp_g,
        "Vcmp_g": Vcmp_g,
        "row_max": row_max,
        "scale": scale,
        "topk": topk,
        "block_idx": block_idx,
        "block_count": block_count,
        "Kslc_full": Kslc_full,
        "Vslc_full": Vslc_full,
        "k_win_full": k_win_full,
        "v_win_full": v_win_full,
        "window_size": window_size,
        "cmp_blk_size": cmp_blk_size,
        "nb": nb,
        "block_starts": block_starts,
        "dtype": dtype,
        "batch": batch,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
    }

def _setup_linear(
    device: torch.device,
    dim: int,
    *,
    batch: int = 64,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(0)
    in_dim = dim
    out_dim = dim
    base_input = torch.randn(batch, in_dim, device=device, dtype=dtype)
    triton_linear = TritonLinear(in_dim, out_dim, bias=True).to(device, dtype)

    weight = triton_linear.weight.detach().to(dtype)
    bias = triton_linear.bias.detach().to(dtype) if triton_linear.bias is not None else None

    grad_out = torch.randn(batch, out_dim, device=device, dtype=dtype)

    def torch_fwd() -> None:
        x = base_input.clone()
        out = x @ weight.t()
        if bias is not None:
            out = out + bias

    def triton_fwd() -> None:
        x = base_input.clone()
        triton_linear(x)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_input.clone().detach().requires_grad_(True)
        w = weight.clone().detach().requires_grad_(True)
        b = bias.clone().detach().requires_grad_(True) if bias is not None else None
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = x @ w.t()
            if b is not None:
                out = out + b
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_input.clone()
        g_out = grad_out.clone()
        w = weight.clone()
        b = bias.clone() if bias is not None else None

        def forward() -> None:
            pass

        def backward() -> None:
            grad_input = g_out @ w
            grad_weight = g_out.t() @ x
            accum = grad_input.sum() + grad_weight.sum()
            if b is not None:
                grad_bias = g_out.sum(dim=0)
                accum = accum + grad_bias.sum()
            accum.item()

        return forward, backward

    config = {
        "batch": batch,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "dim": dim,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_matmul(
    device: torch.device,
    dim: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(1)
    rows = cols = shared = dim
    left = torch.randn(rows, shared, device=device, dtype=dtype)
    right = torch.randn(cols, shared, device=device, dtype=dtype)

    grad_out = torch.randn(rows, cols, device=device, dtype=dtype)

    def torch_fwd() -> None:
        a = left.clone()
        b = right.clone()
        torch.matmul(a, b.t())

    def triton_fwd() -> None:
        a = left.clone()
        b = right.clone()
        triton_matmul(a, b)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        a = left.clone().detach().requires_grad_(True)
        b = right.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = torch.matmul(a, b.t())
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        a = left.clone()
        b = right.clone()
        g_out = grad_out.clone()
        def forward() -> None:
            pass

        def backward() -> None:
            grad_a = g_out @ b
            grad_b = g_out.t() @ a
            (grad_a.sum() + grad_b.sum()).item()

        return forward, backward

    config = {
        "rows": rows,
        "cols": cols,
        "shared": shared,
        "dim": dim,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_zero_centered_rms(
    device: torch.device,
    dim: int,
    *,
    batch: int = 8192,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(2)
    eps = 1e-6
    decay = 0.2
    base_hidden = torch.randn(batch, dim, device=device, dtype=dtype)
    module = ZeroCenteredRMSNorm(dim, eps=eps, input_weight_decay=decay).to(device, dtype)

    grad_out = torch.randn(batch, dim, device=device, dtype=torch.float32)

    def _reference(x: torch.Tensor) -> torch.Tensor:
        values = x.float() * (1.0 - decay)
        centered = values - values.mean(dim=-1, keepdim=True)
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        out = centered * torch.rsqrt(var + eps)
        out = out * module.weight.float()
        return out

    def torch_fwd() -> None:
        x = base_hidden.clone()
        _ = _reference(x)

    def triton_fwd() -> None:
        x = base_hidden.clone()
        module(x)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_hidden.clone().detach().requires_grad_(True)
        weight = module.weight.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            values = x.float() * (1.0 - decay)
            centered = values - values.mean(dim=-1, keepdim=True)
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            out = centered * torch.rsqrt(var + eps)
            out = out * weight.float()
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_hidden.clone().detach().requires_grad_(True)
        weight = module.weight.clone().detach()
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            values = x.float() * (1.0 - decay)
            centered = values - values.mean(dim=-1, keepdim=True)
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            out = centered * torch.rsqrt(var + eps)
            state["out"] = out * weight.float()

        def backward() -> None:
            grad_input = torch.autograd.grad(
                (state["out"] * grad_out).sum(),
                x,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_input is not None:
                _ = grad_input

        return forward, backward

    config = {
        "dim": dim,
        "batch": batch,
        "dtype": str(dtype),
        "eps": eps,
        "decay": decay,
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_gated_rms(
    device: torch.device,
    dim: int,
    *,
    batch: int = 8192,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(3)
    eps = 1e-6
    decay = 0.2
    base_hidden = torch.randn(batch, dim, device=device, dtype=dtype)
    gate = torch.randn(batch, dim, device=device, dtype=dtype)
    module = GatedRMSNorm(dim, eps=eps, input_weight_decay=decay).to(device, dtype)

    grad_out = torch.randn(batch, dim, device=device, dtype=torch.float32)

    def _reference(x: torch.Tensor, gate_tensor: torch.Tensor) -> torch.Tensor:
        values = x.float() * (1.0 - decay)
        centered = values - values.mean(dim=-1, keepdim=True)
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(var + eps)
        normalized = normalized * module.weight.float()
        normalized = normalized * torch.nn.functional.silu(gate_tensor.float())
        return normalized

    def torch_fwd() -> None:
        x = base_hidden.clone()
        g = gate.clone()
        _ = _reference(x, g)

    def triton_fwd() -> None:
        x = base_hidden.clone()
        module(x, gate)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_hidden.clone().detach().requires_grad_(True)
        g = gate.clone().detach().requires_grad_(True)
        w = module.weight.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            values = x.float() * (1.0 - decay)
            centered = values - values.mean(dim=-1, keepdim=True)
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            normalized = centered * torch.rsqrt(var + eps)
            normalized = normalized * w.float()
            normalized = normalized * torch.nn.functional.silu(g.float())
            state["loss"] = (normalized * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = base_hidden.clone().detach().requires_grad_(True)
        g = gate.clone().detach()
        weight = module.weight.clone().detach()
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            values = x.float() * (1.0 - decay)
            centered = values - values.mean(dim=-1, keepdim=True)
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            normalized = centered * torch.rsqrt(var + eps)
            normalized = normalized * weight.float()
            normalized = normalized * torch.nn.functional.silu(g.float())
            state["normalized"] = normalized

        def backward() -> None:
            grad_input = torch.autograd.grad(
                (state["normalized"] * grad_out).sum(),
                x,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_input is not None:
                _ = grad_input

        return forward, backward

    config = {
        "dim": dim,
        "batch": batch,
        "dtype": str(dtype),
        "eps": eps,
        "decay": decay,
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_attention(
    device: torch.device,
    seq: int,
    *,
    batch: int = 4,
    heads: int = 16,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(4)
    scale = 1.0 / math.sqrt(head_dim)
    query = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    mask = torch.triu(torch.ones(seq, seq, device=device), diagonal=1).bool()

    grad_out = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float32)

    def torch_fwd() -> None:
        q = query.clone().permute(0, 2, 1, 3).float()
        k = key.clone().permute(0, 2, 1, 3).float()
        v = value.clone().permute(0, 2, 1, 3).float()
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        torch.matmul(probs, v)

    def triton_fwd() -> None:
        q = query.clone()
        k = key.clone()
        v = value.clone()
        scaled_dot_product_attention(q, k, v, scale)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().permute(0, 2, 1, 3).detach().requires_grad_(True)
        k = key.clone().permute(0, 2, 1, 3).detach().requires_grad_(True)
        v = value.clone().permute(0, 2, 1, 3).detach().requires_grad_(True)
        grad = grad_out.clone()
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            out = torch.matmul(probs, v)
            state["loss"] = (out * grad).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().permute(0, 2, 1, 3).detach().requires_grad_(True)
        k = key.clone().permute(0, 2, 1, 3).detach()
        v = value.clone().permute(0, 2, 1, 3).detach()
        grad = grad_out.clone()
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            state["out"] = torch.matmul(probs, v.float())

        def backward() -> None:
            grad_input = torch.autograd.grad(
                (state["out"] * grad).sum(),
                q,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_input is not None:
                _ = grad_input

        return forward, backward

    config = {
        "batch": batch,
        "seq": seq,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "causal": True,
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_nsa(
    device: torch.device,
    seq_len: int,
    *,
    d_model: int = 128,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    cmp_blk_size: int = 32,
    cmp_stride: int = 16,
    slc_top_n: int = 4,
    window_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(7)
    cfg = {
        "d_model": d_model,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "seq_len": seq_len,
        "cmp_blk_size": cmp_blk_size,
        "cmp_stride": cmp_stride,
        "slc_top_n": slc_top_n,
        "window_size": window_size,
    }

    batch, seq = 2, cfg["seq_len"]
    x_base = torch.randn(batch, seq, cfg["d_model"], device=device, dtype=dtype)
    q_base = torch.randn(batch, cfg["n_q_heads"], seq, cfg["d_head"], device=device, dtype=dtype)
    grad_out = torch.randn(batch, seq, cfg["n_q_heads"] * cfg["d_head"], device=device, dtype=torch.float32)

    nsa_module = NativeSparseAttention(**cfg).to(device, dtype=dtype)
    scale = 1.0 / math.sqrt(cfg["d_head"])

    gate_w = nsa_module.gate.weight.detach().to(torch.float32)
    gate_b = nsa_module.gate.bias.detach().to(torch.float32)
    w_k_cmp = nsa_module.W_k_cmp.weight.detach().to(torch.float32)
    w_v_cmp = nsa_module.W_v_cmp.weight.detach().to(torch.float32)
    w_k_slc = nsa_module.W_k_slc.weight.detach().to(torch.float32)
    w_v_slc = nsa_module.W_v_slc.weight.detach().to(torch.float32)
    w_k_win = nsa_module.W_k_win.weight.detach().to(torch.float32)
    w_v_win = nsa_module.W_v_win.weight.detach().to(torch.float32)
    block_pos = nsa_module.block_pos.detach().to(torch.float32)

    n_q = cfg["n_q_heads"]
    n_kv = cfg["n_kv_heads"]
    d_head = cfg["d_head"]

    causal_mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
    idx = torch.arange(seq, device=device)
    window = cfg["window_size"]
    if window is not None and window > 0:
        dist = idx[:, None] - idx[None, :]
        window_mask = dist >= window
        causal_mask = causal_mask | window_mask

    def _dense_attention(q_heads: torch.Tensor, k_heads: torch.Tensor, v_heads: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(v_heads.dtype)
        return torch.matmul(probs, v_heads)

    def _torch_reference(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        q32 = q.to(torch.float32)

        gate_logits = torch.nn.functional.linear(x32, gate_w, gate_b)
        gates = torch.sigmoid(gate_logits).view(batch, seq, n_q, 3).permute(0, 2, 1, 3)
        g_cmp, g_slc, g_win = gates[..., 0], gates[..., 1], gates[..., 2]

        k_cmp = torch.nn.functional.linear(x32, w_k_cmp).view(batch, seq, n_kv, d_head)
        v_cmp = torch.nn.functional.linear(x32, w_v_cmp).view(batch, seq, n_kv, d_head)
        k_slc = torch.nn.functional.linear(x32, w_k_slc).view(batch, seq, n_kv, d_head)
        v_slc = torch.nn.functional.linear(x32, w_v_slc).view(batch, seq, n_kv, d_head)
        k_win = torch.nn.functional.linear(x32, w_k_win).view(batch, seq, n_kv, d_head)
        v_win = torch.nn.functional.linear(x32, w_v_win).view(batch, seq, n_kv, d_head)

        # Apply learned block positional encodings for compressed branch
        cmp_blocks = []
        cmp_values = []
        block_starts = list(range(0, max(1, seq - cfg["cmp_blk_size"] + 1), cfg["cmp_stride"]))
        if 0 not in block_starts:
            block_starts = [0] + block_starts
        for start in block_starts:
            end = min(start + cfg["cmp_blk_size"], seq)
            kblk = k_cmp[:, start:end]
            vblk = v_cmp[:, start:end]
            pad = cfg["cmp_blk_size"] - (end - start)
            if pad > 0:
                pad_shape = (batch, pad, n_kv, d_head)
                kblk = torch.cat([kblk, torch.zeros(pad_shape, device=device, dtype=kblk.dtype)], dim=1)
                vblk = torch.cat([vblk, torch.zeros(pad_shape, device=device, dtype=vblk.dtype)], dim=1)
            kblk = kblk + block_pos.view(1, cfg["cmp_blk_size"], 1, d_head)
            vblk = vblk + block_pos.view(1, cfg["cmp_blk_size"], 1, d_head)
            ksum, vsum = nsa_module._compress_block_pair(kblk, vblk)
            cmp_blocks.append(ksum)
            cmp_values.append(vsum)
        k_cmp_summary = torch.stack(cmp_blocks, dim=1)  # (B, nb, n_kv, d)
        v_cmp_summary = torch.stack(cmp_values, dim=1)

        nb = k_cmp_summary.shape[1]
        k_cmp_heads = k_cmp_summary.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        v_cmp_heads = v_cmp_summary.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        k_cmp_heads = k_cmp_heads.repeat_interleave(seq // nb + 1, dim=2)[..., :seq, :]
        v_cmp_heads = v_cmp_heads.repeat_interleave(seq // nb + 1, dim=2)[..., :seq, :]

        k_slc_heads = k_slc.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        v_slc_heads = v_slc.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        k_win_heads = k_win.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        v_win_heads = v_win.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)

        q_heads = q32
        base_mask = causal_mask
        cmp_context = _dense_attention(q_heads, k_cmp_heads, v_cmp_heads, base_mask)
        slc_context = _dense_attention(q_heads, k_slc_heads, v_slc_heads, base_mask)
        win_mask = causal_mask
        win_context = _dense_attention(q_heads, k_win_heads, v_win_heads, win_mask)

        out = (
            g_cmp.unsqueeze(-1) * cmp_context
            + g_slc.unsqueeze(-1) * slc_context
            + g_win.unsqueeze(-1) * win_context
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, n_q * d_head)
        return out.to(x.dtype)

    target_dtype = nsa_module.gate.weight.dtype

    def torch_fwd() -> None:
        x = x_base.clone().to(target_dtype)
        q = q_base.clone().to(target_dtype)
        _torch_reference(x, q)

    def triton_fwd() -> None:
        nsa_module.reset_cache()
        x = x_base.clone().to(target_dtype)
        q = q_base.clone().to(target_dtype)
        nsa_module(x, q)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        x = x_base.clone().to(target_dtype).detach().requires_grad_(True)
        q = q_base.clone().to(target_dtype).detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = _torch_reference(x, q)
            state["loss"] = (out.to(torch.float32) * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        nsa_module.zero_grad(set_to_none=True)
        nsa_module.reset_cache()
        x = x_base.clone().to(target_dtype).detach().requires_grad_(True)
        q = q_base.clone().to(target_dtype).detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = nsa_module(x, q)
            state["loss"] = (out.to(torch.float32) * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    config = {
        **cfg,
        "batch": batch,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_nsa_compressed_attention(
    device: torch.device,
    seq_len: int,
    *,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    cmp_blk_size: int = 32,
    cmp_stride: int = 16,
    slc_top_n: int = 4,
    dtype: torch.dtype = torch.float16,
) -> SetupReturn:
    inputs = _prepare_nsa_inputs(
        device,
        seq_len,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        cmp_blk_size=cmp_blk_size,
        cmp_stride=cmp_stride,
        slc_top_n=slc_top_n,
        dtype=dtype,
    )
    q_group = inputs["q_group"]
    Kcmp_g = inputs["Kcmp_g"]
    Vcmp_g = inputs["Vcmp_g"]
    row_max = inputs["row_max"].to(torch.int32)
    scale = inputs["scale"]
    nb = Kcmp_g.shape[2]

    mask = torch.arange(nb, device=device).view(1, 1, 1, nb) > row_max.unsqueeze(-1)

    def torch_fwd() -> None:
        scores = torch.matmul(q_group.to(torch.float32), Kcmp_g.transpose(-1, -2).to(torch.float32)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        _ = torch.matmul(probs, Vcmp_g.to(torch.float32))

    def triton_fwd() -> None:
        casual_attention.apply(q_group, Kcmp_g, Vcmp_g, scale, 0, row_max)

    config = {
        "seq_len": seq_len,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "cmp_blk_size": cmp_blk_size,
        "cmp_stride": cmp_stride,
        "slc_top_n": slc_top_n,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, None, None, config


def _setup_nsa_selected_attention(
    device: torch.device,
    seq_len: int,
    *,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    cmp_blk_size: int = 32,
    cmp_stride: int = 16,
    slc_top_n: int = 4,
    dtype: torch.dtype = torch.float16,
) -> SetupReturn:
    inputs = _prepare_nsa_inputs(
        device,
        seq_len,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        cmp_blk_size=cmp_blk_size,
        cmp_stride=cmp_stride,
        slc_top_n=slc_top_n,
        dtype=dtype,
    )
    q_slc = inputs["q_slc"]
    Kslc_full = inputs["Kslc_full"]
    Vslc_full = inputs["Vslc_full"]
    block_idx = inputs["block_idx"].to(torch.int32)
    block_count = inputs["block_count"].to(torch.int32)
    scale = inputs["scale"]
    block_starts = inputs["block_starts"]
    cmp_blk_size = inputs["cmp_blk_size"]
    B, G, T, _ = q_slc.shape
    Kmax = block_idx.shape[-1]
    seq = Kslc_full.shape[2]

    block_starts_tensor = torch.tensor(block_starts, device=device, dtype=torch.int32)
    positions = torch.arange(seq, device=device, dtype=torch.int32)
    block_ranges = (
        (positions.unsqueeze(0) >= block_starts_tensor.unsqueeze(1))
        & (positions.unsqueeze(0) < (block_starts_tensor + cmp_blk_size).unsqueeze(1))
    )
    block_ranges = block_ranges.to(torch.bool)

    block_ranges_exp = block_ranges.view(1, 1, 1, block_ranges.shape[0], seq).expand(B, G, T, -1, -1)
    idx_expanded = block_idx.long().unsqueeze(-1).expand(-1, -1, -1, -1, seq)
    selected_ranges = torch.gather(block_ranges_exp, 3, idx_expanded)
    count_mask = torch.arange(Kmax, device=device, dtype=torch.int32).view(1, 1, 1, Kmax, 1) < block_count.unsqueeze(-1).unsqueeze(-1)
    selected_ranges = selected_ranges & count_mask
    keep_mask = selected_ranges.any(dim=3)
    t_idx = torch.arange(T, device=device).view(1, 1, T, 1)
    pos_idx = torch.arange(seq, device=device).view(1, 1, 1, seq)
    causal_mask = pos_idx <= t_idx
    keep_mask = keep_mask & causal_mask

    def torch_fwd() -> None:
        scores = torch.matmul(q_slc.to(torch.float32), Kslc_full.transpose(-1, -2).to(torch.float32)) * scale
        scores = scores.masked_fill(~keep_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        _ = torch.matmul(probs, Vslc_full.to(torch.float32))

    def triton_fwd() -> None:
        select_attention(q_slc, Kslc_full, Vslc_full, scale, block_idx, block_count, cmp_blk_size)

    config = {
        "seq_len": seq_len,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "cmp_blk_size": cmp_blk_size,
        "cmp_stride": cmp_stride,
        "slc_top_n": slc_top_n,
        "dtype": str(dtype),
        "Kmax": Kmax,
    }

    grad_out = torch.randn_like(q_slc, dtype=torch.float32)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = q_slc.clone().detach().requires_grad_(True)
        k = Kslc_full.clone().detach().requires_grad_(True)
        v = Vslc_full.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            scores = torch.matmul(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)) * scale
            scores = scores.masked_fill(~keep_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v.to(torch.float32))
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = q_slc.clone().detach().requires_grad_(True)
        k = Kslc_full.clone().detach().requires_grad_(True)
        v = Vslc_full.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = select_attention(q, k, v, scale, block_idx, block_count, cmp_blk_size)
            state["loss"] = (out.to(torch.float32) * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_nsa_sliding_attention(
    device: torch.device,
    seq_len: int,
    *,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    window_size: int = 32,
    dtype: torch.dtype = torch.float16,
) -> SetupReturn:
    inputs = _prepare_nsa_inputs(
        device,
        seq_len,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        window_size=window_size,
        dtype=dtype,
    )
    q_win = inputs["q_win_full"]
    k_win = inputs["k_win_full"]
    v_win = inputs["v_win_full"]
    scale = inputs["scale"]
    T = q_win.shape[2]

    idx = torch.arange(T, device=device)
    causal = idx[None, None, :, None] >= idx[None, None, None, :]
    if window_size and window_size > 0:
        causal = causal & ((idx[None, None, :, None] - idx[None, None, None, :]) < window_size)

    def torch_fwd() -> None:
        scores = torch.matmul(q_win.to(torch.float32), k_win.transpose(-1, -2).to(torch.float32)) * scale
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        _ = torch.matmul(probs, v_win.to(torch.float32))

    def triton_fwd() -> None:
        casual_attention.apply(q_win, k_win, v_win, scale, window_size)

    config = {
        "seq_len": seq_len,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "window_size": window_size,
        "dtype": str(dtype),
    }

    grad_out = torch.randn_like(q_win, dtype=torch.float32)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = q_win.clone().detach().requires_grad_(True)
        k = k_win.clone().detach().requires_grad_(True)
        v = v_win.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            scores = torch.matmul(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)) * scale
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v.to(torch.float32))
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = q_win.clone().detach().requires_grad_(True)
        k = k_win.clone().detach().requires_grad_(True)
        v = v_win.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out = casual_attention.apply(q, k, v, scale, window_size)
            state["loss"] = (out.to(torch.float32) * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_nsa_topk(
    device: torch.device,
    seq_len: int,
    *,
    n_q_heads: int = 4,
    n_kv_heads: int = 2,
    d_head: int = 32,
    cmp_blk_size: int = 32,
    cmp_stride: int = 16,
    slc_top_n: int = 4,
    dtype: torch.dtype = torch.float16,
) -> SetupReturn:
    inputs = _prepare_nsa_inputs(
        device,
        seq_len,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        cmp_blk_size=cmp_blk_size,
        cmp_stride=cmp_stride,
        slc_top_n=slc_top_n,
        dtype=dtype,
    )
    q_group = inputs["q_group"]
    Kcmp_g = inputs["Kcmp_g"]
    row_max = inputs["row_max"].to(torch.int32)
    scale = inputs["scale"]
    nb = Kcmp_g.shape[2]
    topk = inputs["topk"]

    mask = torch.arange(nb, device=device).view(1, 1, 1, nb) > row_max.unsqueeze(-1)

    def torch_fwd() -> None:
        scores = torch.matmul(q_group.to(torch.float32), Kcmp_g.transpose(-1, -2).to(torch.float32)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        _ = torch.topk(scores, k=topk, dim=-1).indices

    def triton_fwd() -> None:
        topk_indices(q_group, Kcmp_g, scale, int(topk), row_max)

    config = {
        "seq_len": seq_len,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "cmp_blk_size": cmp_blk_size,
        "cmp_stride": cmp_stride,
        "slc_top_n": slc_top_n,
        "topk": int(topk),
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, None, None, config


def _setup_delta_rule(
    device: torch.device,
    seq_len: int,
    *,
    batch: int = 2,
    heads: int = 4,
    d_k: int = 16,
    d_v: int = 32,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(5)
    query = torch.randn(batch, seq_len, heads, d_k, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seq_len, heads, d_v, device=device, dtype=dtype)
    decay = torch.rand(batch, seq_len, heads, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=dtype)

    grad_out = torch.randn(batch, seq_len, heads, d_v, device=device, dtype=torch.float32)

    def torch_fwd() -> None:
        state = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        _torch_gated_delta(query, key, value, decay, beta, state)

    def triton_fwd() -> None:
        gated_delta_rule(query, key, value, decay=decay, beta=beta)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().detach().requires_grad_(True)
        k = key.clone().detach().requires_grad_(True)
        v = value.clone().detach().requires_grad_(True)
        dec = decay.clone().detach().requires_grad_(True)
        bet = beta.clone().detach().requires_grad_(True)
        state_tensor = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out, _ = _torch_gated_delta(q, k, v, dec, bet, state_tensor)
            state["loss"] = (out * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().detach().requires_grad_(True)
        k = key.clone().detach()
        v = value.clone().detach()
        dec = decay.clone().detach()
        bet = beta.clone().detach()
        state_tensor = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out, _ = _torch_gated_delta(q, k, v, dec, bet, state_tensor)
            state["out"] = out

        def backward() -> None:
            grad_input = torch.autograd.grad(
                (state["out"] * grad_out).sum(),
                q,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_input is not None:
                _ = grad_input

        return forward, backward

    config = {
        "batch": batch,
        "seq_len": seq_len,
        "heads": heads,
        "d_k": d_k,
        "d_v": d_v,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_delta_step(
    device: torch.device,
    d_model: int,
    *,
    batch: int = 2,
    heads: int = 4,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(6)
    d_k = d_model
    d_v = d_model
    query = torch.randn(batch, heads, d_k, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, heads, d_v, device=device, dtype=dtype)
    decay = torch.rand(batch, heads, device=device, dtype=dtype)
    beta = torch.rand(batch, heads, device=device, dtype=dtype)

    grad_out = torch.randn(batch, heads, d_v, device=device, dtype=torch.float32)

    def torch_fwd() -> None:
        state = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        _torch_gated_delta(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            decay.unsqueeze(1),
            beta.unsqueeze(1),
            state,
        )

    def triton_fwd() -> None:
        gated_delta_step(query, key, value, decay=decay, beta=beta)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().detach().requires_grad_(True)
        k = key.clone().detach().requires_grad_(True)
        v = value.clone().detach().requires_grad_(True)
        dec = decay.clone().detach().requires_grad_(True)
        bet = beta.clone().detach().requires_grad_(True)
        state_tensor = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out, _ = _torch_gated_delta(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                dec.unsqueeze(1),
                bet.unsqueeze(1),
                state_tensor,
            )
            state["loss"] = (out[:, 0] * grad_out).sum()

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        q = query.clone().detach().requires_grad_(True)
        state_tensor = torch.zeros(batch, heads, d_k, d_v, device=device, dtype=torch.float32)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            out, _ = _torch_gated_delta(
                q.unsqueeze(1),
                key.unsqueeze(1),
                value.unsqueeze(1),
                decay.unsqueeze(1),
                beta.unsqueeze(1),
                state_tensor,
            )
            state["out"] = out[:, 0]

        def backward() -> None:
            grad_input = torch.autograd.grad(
                (state["out"] * grad_out).sum(),
                q,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_input is not None:
                _ = grad_input

        return forward, backward

    config = {
        "batch": batch,
        "heads": heads,
        "d_model": d_model,
        "d_k": d_k,
        "d_v": d_v,
        "dtype": str(dtype),
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _setup_cce(
    device: torch.device,
    vocab: int,
    *,
    batch: int = 4,
    seq: int = 256,
    dim: int = 256,
    dtype: torch.dtype = torch.float32,
    block_size: int = 64,
) -> Tuple[Callable[[], None], Callable[[], None], BackwardSetup, BackwardSetup, dict]:
    torch.manual_seed(7)
    hidden = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    weight = torch.randn(vocab, dim, device=device, dtype=dtype)
    targets = torch.randint(0, vocab, (batch, seq), device=device, dtype=torch.long)

    def torch_fwd() -> None:
        hidden_flat = hidden.view(-1, dim)
        logits = hidden_flat @ weight.t()
        torch.nn.functional.cross_entropy(logits, targets.view(-1))

    def triton_fwd() -> None:
        triton_cce_loss(hidden, weight, targets, block_size=block_size)

    def torch_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        h = hidden.clone().detach().requires_grad_(True)
        w = weight.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            logits = h.view(-1, dim) @ w.t()
            state["loss"] = torch.nn.functional.cross_entropy(logits, targets.view(-1))

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    def triton_bwd_prepare() -> Tuple[Callable[[], None], Callable[[], None]]:
        h = hidden.clone().detach().requires_grad_(True)
        w = weight.clone().detach().requires_grad_(True)
        state: Dict[str, torch.Tensor] = {}

        def forward() -> None:
            state["loss"] = triton_cce_loss(h, w, targets, block_size=block_size)

        def backward() -> None:
            state["loss"].backward()

        return forward, backward

    config = {
        "batch": batch,
        "seq": seq,
        "dim": dim,
        "vocab": vocab,
        "dtype": str(dtype),
        "block_size": block_size,
    }

    return torch_fwd, triton_fwd, torch_bwd_prepare, triton_bwd_prepare, config


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _format_speedup(value: float) -> str:
    if value == float("inf"):
        return "—"
    return f"{value:.2f}x"


def _write_markdown_report(results: List[SweepResult], report_path: Path) -> None:
    lines = [
        "# Triton vs. PyTorch Kernel Benchmark Sweeps",
        "",
        "Timings collected with CUDA events (averaged per iteration).",
        "",
        "PyTorch baselines use unfused reference implementations; Triton kernels are forward-optimized.",
        "Backward timings rely on PyTorch autograd when Triton backward kernels are unavailable.",
    ]

    for sweep in results:
        lines += [
            "",
            f"## {sweep.name} vs. PyTorch ({sweep.param_name} sweep)",
            "",
            "| {param} | Torch FWD (ms) | Triton FWD (ms) | FWD Speedup | Torch BWD (ms) | Triton BWD (ms) | BWD Speedup |".format(
                param=sweep.param_name
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for value, bench in zip(sweep.values, sweep.benchmarks):
            torch_bwd = _format_optional(bench.torch_bwd_ms)
            triton_bwd = _format_optional(bench.triton_bwd_ms)
            lines.append(
                f"| {value} | {bench.torch_fwd_ms:.2f} | {bench.triton_fwd_ms:.2f} | {_format_speedup(bench.speedup)} | {torch_bwd} | {triton_bwd} | {_format_speedup(bench.backward_speedup)} |"
            )

        lines.append("")
        lines.append("Configs:")
        for cfg in sweep.configs:
            cfg_items = ", ".join(f"{k}={v}" for k, v in cfg.items())
            lines.append(f"- {cfg_items}")

    lines += [
        "",
        "## Environment",
        f"* PyTorch {torch.__version__}",
        f"* CUDA device: {torch.cuda.get_device_name(0)}",
    ]

    report_path.write_text("\n".join(lines))


def _write_json_report(results: List[SweepResult], json_path: Path) -> None:
    payload = {sweep.name: sweep.to_dict() for sweep in results}
    json_path.write_text(json.dumps(payload, indent=2))


def _generate_plots(results: List[SweepResult], plot_dir: Path) -> None:
    for sweep in results:
        x_vals = [int(v) for v in sweep.values]
        torch_fwd = [bench.torch_fwd_ms for bench in sweep.benchmarks]
        triton_fwd = [bench.triton_fwd_ms for bench in sweep.benchmarks]
        torch_bwd = [bench.torch_bwd_ms for bench in sweep.benchmarks]
        triton_bwd = [bench.triton_bwd_ms for bench in sweep.benchmarks]
        speedup_fwd = [bench.speedup if bench.speedup != float("inf") else float("nan") for bench in sweep.benchmarks]
        speedup_bwd = [
            bench.backward_speedup if bench.backward_speedup != float("inf") else float("nan")
            for bench in sweep.benchmarks
        ]

        has_backward = all(b is not None for b in torch_bwd) and all(b is not None for b in triton_bwd)

        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
        ax0, ax1 = axes

        ax0.plot(x_vals, torch_fwd, marker="o", label="PyTorch FWD")
        ax0.plot(x_vals, triton_fwd, marker="o", label="Triton FWD")
        if has_backward:
            ax0.plot(x_vals, torch_bwd, marker="^", linestyle="--", label="PyTorch BWD")
            ax0.plot(x_vals, triton_bwd, marker="^", linestyle="--", label="Triton BWD")
        ax0.set_ylabel("Time (ms)")
        ax0.set_title(f"{sweep.name} timings vs {sweep.param_name}")
        ax0.grid(True, which="both", alpha=0.3)
        ax0.legend()

        ax1.plot(x_vals, speedup_fwd, marker="o", label="Forward speedup")
        if has_backward:
            ax1.plot(x_vals, speedup_bwd, marker="^", linestyle="--", label="Backward speedup")
        ax1.axhline(1.0, color="grey", linestyle=":" , label="Parity")
        ax1.axhline(0.6, color="red", linestyle="--", linewidth=0.8, label="0.6× target")
        ax1.set_ylabel("PyTorch / Triton")
        ax1.set_xlabel(sweep.param_name)
        ax1.set_xticks(x_vals)
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()

        fig_path = plot_dir / f"{sweep.name}.png"
        fig.savefig(fig_path)
        plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemError("CUDA device required to benchmark Triton kernels")
    device = torch.device("cuda")

    sweeps = _default_sweeps()
    sweep_results = [_run_sweep(device, sweep) for sweep in sweeps]

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "triton_vs_torch.md"
    json_path = report_dir / "triton_vs_torch.json"

    _write_markdown_report(sweep_results, report_path)
    _write_json_report(sweep_results, json_path)

    plot_dir = Path("plots/triton_speedups")
    plot_dir.mkdir(exist_ok=True)
    _generate_plots(sweep_results, plot_dir)

    print(f"Report saved to {report_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
