#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute per-layer Q and K statistics from dumps.")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def load_metadata(dump_root: Path) -> Dict[str, object]:
    meta_path = dump_root / "metadata.pt"
    if meta_path.exists():
        return torch.load(meta_path)
    return {}


def infer_layers(dump_root: Path) -> List[Path]:
    layers = sorted(path for path in dump_root.iterdir() if path.is_dir() and path.name.startswith("layer_"))
    if not layers:
        raise RuntimeError(f"No layer directories found under {dump_root}")
    return layers


def main() -> None:
    args = parse_args()
    dump_root = args.dump_root
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    layers = infer_layers(dump_root)
    sample_file = torch.load(next(iter(layers[0].glob("batch_*.pt"))), map_location="cpu")

    batch_size, seq_len, num_q_heads, head_dim = sample_file["q"].shape
    num_kv_heads = sample_file["k"].shape[2]

    sigma_q_all = []
    cov_k_all = []
    k_mean_all = []

    tokens_per_batch = batch_size * seq_len
    metadata = load_metadata(dump_root)

    for layer_dir in layers:
        sigma_accum = torch.zeros((num_q_heads, head_dim, head_dim), dtype=torch.float64)
        kk_accum = torch.zeros((num_kv_heads, head_dim, head_dim), dtype=torch.float64)
        k_sum = torch.zeros((num_kv_heads, head_dim), dtype=torch.float64)
        total_tokens = 0

        batch_files = sorted(layer_dir.glob("batch_*.pt"))
        if not batch_files:
            raise RuntimeError(f"No batch files found in {layer_dir}")

        for batch_path in batch_files:
            payload = torch.load(batch_path, map_location="cpu")

            q = payload["q"].to(torch.float32)
            q = q.permute(2, 0, 1, 3).reshape(num_q_heads, -1, head_dim)
            sigma_batch = torch.einsum("hnd,hne->hde", q, q).to(torch.float64)
            sigma_accum += sigma_batch

            k = payload["k"].to(torch.float32)
            k = k.permute(2, 0, 1, 3).reshape(num_kv_heads, -1, head_dim)
            k_sum += k.sum(dim=1, dtype=torch.float64)
            kk_batch = torch.einsum("hnd,hne->hde", k, k).to(torch.float64)
            kk_accum += kk_batch

            total_tokens += k.shape[1]

        sigma_q = sigma_accum / total_tokens
        k_mean = k_sum / total_tokens
        cov_k = kk_accum / total_tokens - torch.einsum("hd,he->hde", k_mean, k_mean)

        sigma_q_all.append(sigma_q.to(torch.float32))
        cov_k_all.append(cov_k.to(torch.float32))
        k_mean_all.append(k_mean.to(torch.float32))

    sigma_q_tensor = torch.stack(sigma_q_all)  # (num_layers, num_q_heads, dim, dim)
    cov_k_tensor = torch.stack(cov_k_all)  # (num_layers, num_kv_heads, dim, dim)
    k_mean_tensor = torch.stack(k_mean_all)  # (num_layers, num_kv_heads, dim)

    stats = {
        "sigma_q": sigma_q_tensor,
        "cov_k": cov_k_tensor,
        "k_mean": k_mean_tensor,
        "tokens_per_batch": tokens_per_batch,
        "total_batches": metadata.get("total_batches"),
        "batch_size": metadata.get("batch_size", batch_size),
        "seq_len": metadata.get("seq_len", seq_len),
    }
    torch.save(stats, output_path)


if __name__ == "__main__":
    main()
