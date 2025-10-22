#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute L2 distance stats in whitened space")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output-path", type=Path, default=Path("analysis/whitened_chunk_stats.json"))
    parser.add_argument("--num-layers", type=int, default=5, help="Sample first N layers")
    parser.add_argument("--num-batches", type=int, default=10, help="Sample first N batches per layer")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[2, 3, 4, 5, 8, 10])
    parser.add_argument("--samples-per-chunk-size", type=int, default=500)
    return parser.parse_args()


def compute_whitening_matrices(stats_path: Path, num_layers: int, num_kv_heads: int) -> torch.Tensor:
    """Compute sigma_q^(-1/2) for whitening"""
    stats = torch.load(stats_path)
    sigma_q = stats["sigma_q"].to(torch.float64)  # (layers, q_heads, dim, dim)
    
    group_size = sigma_q.shape[1] // num_kv_heads
    head_dim = sigma_q.shape[2]
    
    whitening_matrices = torch.zeros((num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32)
    
    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            q_indices = slice(kv_head * group_size, (kv_head + 1) * group_size)
            sigma_group = 0.5 * (sigma_q[layer, q_indices] + sigma_q[layer, q_indices].transpose(-1, -2))
            sigma_avg = sigma_group.mean(dim=0)
            
            # Eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(sigma_avg)
            eigvals = torch.clamp(eigvals, min=1e-8)
            
            # Compute sigma^(-1/2)
            sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
            whitening_matrices[layer, kv_head] = sigma_inv_sqrt.to(torch.float32)
    
    return whitening_matrices


def whiten_keys(keys: torch.Tensor, whitening_matrix: torch.Tensor) -> torch.Tensor:
    """Apply whitening: keys @ W^T"""
    return torch.matmul(keys, whitening_matrix.T)


def compute_chunk_l2_distance(keys_whitened: torch.Tensor) -> float:
    """Compute mean L2 distance from center in whitened space"""
    center = keys_whitened.mean(dim=0)
    distances = torch.norm(keys_whitened - center.unsqueeze(0), dim=1)
    return distances.mean().item()


def main() -> None:
    args = parse_args()
    
    # Load first batch to get dimensions
    sample_file = torch.load(args.dump_root / "layer_00" / "batch_0000.pt")
    num_kv_heads = sample_file["k"].shape[2]
    head_dim = sample_file["k"].shape[3]
    
    print(f"Computing whitening matrices for {args.num_layers} layers, {num_kv_heads} heads...")
    whitening_matrices = compute_whitening_matrices(args.stats_path, args.num_layers, num_kv_heads)
    
    # Collect L2 distance statistics
    all_distances = {chunk_size: [] for chunk_size in args.chunk_sizes}
    
    for layer_idx in range(args.num_layers):
        print(f"Processing layer {layer_idx}...")
        layer_dir = args.dump_root / f"layer_{layer_idx:02d}"
        batch_paths = sorted(layer_dir.glob("batch_*.pt"))[:args.num_batches]
        
        for batch_path in batch_paths:
            batch = torch.load(batch_path)
            k = batch["k"]  # (batch, seq, kv_heads, head_dim)
            batch_size, seq_len = k.shape[:2]
            
            for kv_head in range(num_kv_heads):
                W = whitening_matrices[layer_idx, kv_head]
                
                # Extract all keys for this head and whiten them
                keys = k[:, :, kv_head, :].reshape(-1, head_dim).to(torch.float32)
                keys_whitened = whiten_keys(keys, W)
                
                # Sample random chunks of each size
                num_keys = keys_whitened.shape[0]
                for chunk_size in args.chunk_sizes:
                    if num_keys < chunk_size:
                        continue
                    
                    samples_needed = args.samples_per_chunk_size // (args.num_layers * args.num_batches * num_kv_heads)
                    samples_needed = max(1, samples_needed)
                    
                    for _ in range(samples_needed):
                        # Sample random contiguous chunk
                        start_idx = torch.randint(0, num_keys - chunk_size + 1, (1,)).item()
                        chunk_keys = keys_whitened[start_idx:start_idx + chunk_size]
                        
                        # Compute L2 distance
                        l2_dist = compute_chunk_l2_distance(chunk_keys)
                        all_distances[chunk_size].append(l2_dist)
    
    # Compute statistics
    stats = {}
    all_values = []
    
    for chunk_size in sorted(args.chunk_sizes):
        distances = all_distances[chunk_size]
        if distances:
            values_tensor = torch.tensor(distances)
            stats[f"chunk_{chunk_size}"] = {
                "mean": float(values_tensor.mean()),
                "median": float(values_tensor.median()),
                "std": float(values_tensor.std()),
                "min": float(values_tensor.min()),
                "max": float(values_tensor.max()),
                "p25": float(values_tensor.quantile(0.25)),
                "p75": float(values_tensor.quantile(0.75)),
                "p90": float(values_tensor.quantile(0.90)),
                "p95": float(values_tensor.quantile(0.95)),
                "num_samples": len(distances),
            }
            all_values.extend(distances)
            print(f"Chunk size {chunk_size}: median={stats[f'chunk_{chunk_size}']['median']:.4f}, "
                  f"mean={stats[f'chunk_{chunk_size}']['mean']:.4f}")
    
    # Overall statistics
    if all_values:
        all_tensor = torch.tensor(all_values)
        stats["overall"] = {
            "mean": float(all_tensor.mean()),
            "median": float(all_tensor.median()),
            "std": float(all_tensor.std()),
            "p25": float(all_tensor.quantile(0.25)),
            "p75": float(all_tensor.quantile(0.75)),
            "p90": float(all_tensor.quantile(0.90)),
            "p95": float(all_tensor.quantile(0.95)),
        }
        
        print(f"\nOverall median L2 distance: {stats['overall']['median']:.4f}")
        print(f"Overall mean L2 distance: {stats['overall']['mean']:.4f}")
        
        # Suggest epsilon values for compression ratios 0.05 to 0.5
        # Lower epsilon = more merging = lower compression ratio
        median = stats["overall"]["median"]
        p25 = stats["overall"]["p25"]
        p75 = stats["overall"]["p75"]
        p90 = stats["overall"]["p90"]
        
        suggested_epsilons = [
            p25 * 0.5,   # Very aggressive merging (~0.05 compression)
            p25,         # Aggressive merging (~0.1 compression)
            median * 0.7, # Moderate merging (~0.2 compression)
            median,      # Balanced (~0.3 compression)
            p75,         # Conservative (~0.5 compression)
        ]
        
        stats["suggested_epsilons"] = [float(e) for e in suggested_epsilons]
        print(f"\nSuggested epsilon values: {[f'{e:.4f}' for e in suggested_epsilons]}")
    
    # Save results
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved statistics to {args.output_path}")


if __name__ == "__main__":
    main()

