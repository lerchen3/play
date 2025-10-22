#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("Plot L2 distance heatmap for whitened keys")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output-path", type=Path, default=Path("analysis/whitened_key_l2_heatmap.png"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=512, help="Max sequence length to plot")
    return parser.parse_args()


def compute_whitening_matrix(stats_path: Path, layer_idx: int, kv_head: int, num_kv_heads: int):
    """Compute sigma_q^(-1/2) for whitening"""
    stats = torch.load(stats_path)
    sigma_q = stats["sigma_q"].to(torch.float64)
    
    num_q_heads = sigma_q.shape[1]
    group_size = num_q_heads // num_kv_heads
    head_dim = sigma_q.shape[2]
    
    q_indices = slice(kv_head * group_size, (kv_head + 1) * group_size)
    sigma_group = 0.5 * (sigma_q[layer_idx, q_indices] + sigma_q[layer_idx, q_indices].transpose(-1, -2))
    sigma_avg = sigma_group.mean(dim=0)
    
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(sigma_avg)
    eigvals = torch.clamp(eigvals, min=1e-8)
    
    # Compute sigma^(-1/2)
    sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
    return sigma_inv_sqrt.to(torch.float32)


def main():
    args = parse_args()
    
    # Load batch
    batch_path = args.dump_root / f"layer_{args.layer:02d}" / f"batch_{args.batch_idx:04d}.pt"
    batch = torch.load(batch_path)
    
    # Get keys for specified head
    k = batch['k']  # (batch, seq, kv_heads, dim)
    num_kv_heads = k.shape[2]
    
    # Extract keys for this head and batch
    keys = k[0, :, args.kv_head, :].float()  # (seq, dim)
    seq_len = min(keys.shape[0], args.max_seq)
    keys = keys[:seq_len]
    
    print(f"Processing layer {args.layer}, KV head {args.kv_head}")
    print(f"Keys shape: {keys.shape}")
    
    # Compute whitening matrix
    whitening_matrix = compute_whitening_matrix(args.stats_path, args.layer, args.kv_head, num_kv_heads)
    
    # Whiten keys
    keys_whitened = torch.matmul(keys, whitening_matrix.T)
    
    # Compute L2 distance matrix
    print("Computing L2 distance matrix...")
    dist_matrix = torch.cdist(keys_whitened, keys_whitened, p=2).numpy()
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full heatmap
    im1 = ax1.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Key Index')
    ax1.set_ylabel('Key Index')
    ax1.set_title(f'L2 Distance (Whitened Keys)\nLayer {args.layer}, KV Head {args.kv_head}')
    plt.colorbar(im1, ax=ax1, label='L2 Distance')
    
    # Zoomed view (first 128x128)
    zoom_size = min(128, seq_len)
    im2 = ax2.imshow(dist_matrix[:zoom_size, :zoom_size], cmap='viridis', aspect='auto')
    ax2.set_xlabel('Key Index')
    ax2.set_ylabel('Key Index')
    ax2.set_title(f'L2 Distance (Whitened Keys) - Zoomed\nFirst {zoom_size}x{zoom_size}')
    plt.colorbar(im2, ax=ax2, label='L2 Distance')
    
    plt.tight_layout()
    
    # Save
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {args.output_path}")
    
    # Print statistics
    print(f"\nL2 Distance Statistics (whitened keys):")
    print(f"  Min: {dist_matrix[dist_matrix > 0].min():.4f}")
    print(f"  Max: {dist_matrix.max():.4f}")
    print(f"  Mean: {dist_matrix.mean():.4f}")
    print(f"  Median: {np.median(dist_matrix):.4f}")
    print(f"  Std: {dist_matrix.std():.4f}")
    
    # Check diagonal neighbors
    diag_dists = []
    for i in range(seq_len - 1):
        diag_dists.append(dist_matrix[i, i+1])
    print(f"\nDiagonal neighbor distances (i to i+1):")
    print(f"  Mean: {np.mean(diag_dists):.4f}")
    print(f"  Median: {np.median(diag_dists):.4f}")
    print(f"  Min: {np.min(diag_dists):.4f}")
    print(f"  Max: {np.max(diag_dists):.4f}")
    
    plt.close()


if __name__ == "__main__":
    main()

