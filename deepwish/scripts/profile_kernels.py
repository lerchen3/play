"""Benchmark Triton kernels against PyTorch baselines.

This script measures the latency of the fused attention kernel
(`train.kernels.fa2.attention`) and the Triton cross-entropy kernel
(`train.kernels.cce.TritonCCE`). It compares each against a straightforward
PyTorch implementation using large-context defaults (seq_len=8192,
d_model=1024) to highlight relative performance.

Usage:
    python scripts/profile_kernels.py [--seq-len 8192 --d-model 1024]

The benchmarks expect a CUDA-capable GPU and synchronize around timed blocks to
ensure accurate measurements.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.kernels.fa2 import attention as triton_attention
from model.cce import triton_cce_loss


@dataclass
class BenchmarkResult:
    torch_ms: float
    triton_ms: float

    @property
    def speedup(self) -> float:
        if self.triton_ms == 0:
            return float("inf")
        return self.torch_ms / self.triton_ms


def _benchmark(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Time a callable using CUDA events and return average milliseconds."""
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
    return start.elapsed_time(end) / iters


def benchmark_attention(seq_len: int, d_model: int, *, batch_size: int = 1, n_heads: int = 16) -> BenchmarkResult:
    device = torch.device("cuda")
    dtype = torch.float16
    head_dim = d_model // n_heads
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((batch_size, n_heads, seq_len, head_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    @torch.no_grad()
    def torch_forward() -> None:
        logits = torch.matmul(q, k.transpose(-2, -1)).to(torch.float32)
        logits = logits * scale
        logits = logits.masked_fill(~causal_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        _ = torch.matmul(probs.to(dtype), v)

    @torch.no_grad()
    def triton_forward() -> None:
        _ = triton_attention(q, k, v, scale)

    torch_ms = _benchmark(torch_forward)
    triton_ms = _benchmark(triton_forward)
    return BenchmarkResult(torch_ms=torch_ms, triton_ms=triton_ms)


def benchmark_cce(seq_len: int, d_model: int, *, vocab_size: int = 32000, batch_size: int = 1) -> BenchmarkResult:
    device = torch.device("cuda")
    dtype = torch.float32
    tokens = batch_size * seq_len

    embeddings = torch.randn((tokens, d_model), device=device, dtype=dtype, requires_grad=True)
    classifier = torch.randn((vocab_size, d_model), device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab_size, (tokens,), device=device, dtype=torch.long)

    def _zero_grads() -> None:
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        if classifier.grad is not None:
            classifier.grad.zero_()

    def torch_ce() -> None:
        _zero_grads()
        logits = embeddings @ classifier.t()
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()

    def triton_ce() -> None:
        _zero_grads()
        loss = triton_cce_loss(embeddings, classifier, targets)
        loss.backward()

    torch_ms = _benchmark(torch_ce, warmup=3, iters=5)
    triton_ms = _benchmark(triton_ce, warmup=3, iters=5)
    return BenchmarkResult(torch_ms=torch_ms, triton_ms=triton_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels against PyTorch baselines")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length / tokens per sample")
    parser.add_argument("--d-model", type=int, default=1024, help="Model hidden dimension")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size for CE benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarks")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemError("CUDA device required to profile Triton kernels")

    args = parse_args()

    attn = benchmark_attention(args.seq_len, args.d_model, batch_size=args.batch_size)
    cce = benchmark_cce(args.seq_len, args.d_model, vocab_size=args.vocab, batch_size=args.batch_size)

    def fmt(result: BenchmarkResult) -> str:
        speedup = result.speedup
        if speedup >= 1:
            ratio = f"{speedup:.2f}x faster"
        else:
            ratio = f"{1/speedup:.2f}x slower"
        return f"torch {result.torch_ms:.2f} ms | triton {result.triton_ms:.2f} ms | {ratio}"

    print("Attention (seq_len={}, d_model={}, batch={}):".format(args.seq_len, args.d_model, args.batch_size))
    print("  " + fmt(attn))
    print("Cross-Entropy (tokens={}, vocab_size={}):".format(args.seq_len * args.batch_size, args.vocab))
    print("  " + fmt(cce))


if __name__ == "__main__":
    main()
