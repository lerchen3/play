#!/usr/bin/env python
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Estimate chunk-merge epsilon statistics")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output", type=Path, default=Path("analysis/chunk_merge_stats.json"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=5)
    return parser.parse_args()


def load_layer_batches(layer_dir: Path, max_batches: int) -> List[Path]:
    files = sorted(layer_dir.glob("batch_*.pt"))
    if max_batches:
        files = files[:max_batches]
    return files


def repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    batch, heads, seq_len, dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(2).expand(batch, heads, repeat, seq_len, dim)
    return hidden_states.reshape(batch, heads * repeat, seq_len, dim)


def chunk_error(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    ridge: float = 1e-4,
) -> Tuple[float, float]:
    if queries.numel() == 0:
        return 0.0, 0.0
    q = queries.to(torch.float32)
    k = keys.to(torch.float32)
    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(q, k.transpose(0, 1)) * scale
    targets = torch.logsumexp(logits, dim=1)
    q_t = q.transpose(0, 1)
    cov = torch.matmul(q_t, q)
    cov = cov + ridge * torch.eye(q.shape[1], dtype=torch.float32)
    rhs = torch.matmul(q_t, targets)
    k_center = torch.linalg.solve(cov, rhs)
    merged = torch.exp(torch.matmul(q, k_center) * scale)
    original = torch.exp(logits).sum(dim=1)
    err = torch.mean((original - merged) ** 2).item()
    return float(err), float(torch.norm(k_center).item())


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    stats = torch.load(args.stats_path)
    sigma_q = stats["sigma_q"]
    num_layers, num_q_heads, head_dim, _ = sigma_q.shape
    num_kv_heads = stats["cov_k"].shape[1]
    group_size = num_q_heads // num_kv_heads

    dump_root = args.dump_root
    all_errors: List[float] = []
    per_length: Dict[int, List[float]] = {length: [] for length in range(2, 11)}

    for layer_idx in range(num_layers):
        layer_dir = dump_root / f"layer_{layer_idx:02d}"
        batch_files = load_layer_batches(layer_dir, args.max_batches)
        if not batch_files:
            continue

        for batch_path in batch_files:
            payload = torch.load(batch_path, map_location="cpu")
            q = payload["q"]  # (batch, seq, num_heads, head_dim)
            k = payload["k"]  # (batch, seq, kv_heads, head_dim)
            batch, seq_len, _, _ = q.shape

            q = q.permute(0, 2, 1, 3).contiguous()  # (batch, num_heads, seq, dim)
            k = k.permute(0, 2, 1, 3).contiguous()  # (batch, kv_heads, seq, dim)

            for _ in range(args.samples // max(1, len(batch_files))):
                length = rng.randint(2, 10)
                start = rng.randint(0, seq_len - length - 1)
                kv_head = rng.randint(0, num_kv_heads - 1)
                sample_batch = rng.randint(0, batch - 1)

                keys = k[sample_batch, kv_head, start : start + length]
                q_slice = q[sample_batch, kv_head * group_size : (kv_head + 1) * group_size]
                queries = q_slice.reshape(-1, head_dim)

                err, _ = chunk_error(queries, keys, head_dim)
                per_length[length].append(err)
                all_errors.append(err)

    median = float(torch.median(torch.tensor(all_errors)) if all_errors else 0.0)
    percentile_75 = float(torch.quantile(torch.tensor(all_errors), 0.75) if all_errors else 0.0)

    summary = {
        "median": median,
        "p75": percentile_75,
        "per_length": {str(k): v for k, v in per_length.items()},
        "total_samples": len(all_errors),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
