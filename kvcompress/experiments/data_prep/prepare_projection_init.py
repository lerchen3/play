#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare projection initializations and var_dir plots.")
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--plot-path", type=Path, default=Path("analysis/var_dir_curve.png"))
    return parser.parse_args()


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-1, -2))


def main() -> None:
    args = parse_args()
    stats = torch.load(args.stats_path)
    sigma_q = stats["sigma_q"].to(torch.float64)  # (layers, q_heads, dim, dim)
    cov_k = stats["cov_k"].to(torch.float64)  # (layers, kv_heads, dim, dim)

    num_layers, num_q_heads, head_dim, _ = sigma_q.shape
    _, num_kv_heads, _, _ = cov_k.shape
    group_size = num_q_heads // num_kv_heads

    eigenvectors = torch.zeros((num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32)
    var_dir = torch.zeros((num_layers, num_kv_heads, head_dim), dtype=torch.float64)

    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            q_indices = slice(kv_head * group_size, (kv_head + 1) * group_size)
            sigma_group = symmetrize(sigma_q[layer, q_indices].mean(dim=0))
            cov = symmetrize(cov_k[layer, kv_head])

            # Compute sigma_q^(1/2) via eigendecomposition
            eigvals_q, eigvecs_q = torch.linalg.eigh(sigma_group)
            eigvals_q = torch.clamp(eigvals_q, min=1e-10)  # numerical stability
            sigma_sqrt = eigvecs_q @ torch.diag(torch.sqrt(eigvals_q)) @ eigvecs_q.transpose(0, 1)
            
            # Compute the scaled key covariance: sigma_q^(1/2) @ Cov(k) @ sigma_q^(1/2)
            # After query whitening, this is what matters for attention
            scaled_cov_k = sigma_sqrt @ cov @ sigma_sqrt
            scaled_cov_k = symmetrize(scaled_cov_k)
            
            # Get eigendecomposition of the scaled covariance (for eigenvalues)
            scaled_eigvals, _ = torch.linalg.eigh(scaled_cov_k)
            scaled_eigvals = torch.clamp(scaled_eigvals, min=0.0)
            
            # For initialization, use plain key covariance eigenvectors (simpler/more stable)
            cov_eigvals, cov_eigvecs = torch.linalg.eigh(cov)
            
            # Sort both by the scaled eigenvalues (what matters theoretically)
            order = torch.argsort(scaled_eigvals, descending=True)
            var_sorted = scaled_eigvals[order]
            eigvecs_init = cov_eigvecs[:, torch.argsort(cov_eigvals, descending=True)]

            eigenvectors[layer, kv_head] = eigvecs_init.to(torch.float32)
            var_dir[layer, kv_head] = var_sorted

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "eigenvectors": eigenvectors,
            "var_dir": var_dir,
        },
        output_dir / "projection_init.pt",
    )

    mean_curve = var_dir.mean(dim=(0, 1)).cpu()
    std_curve = var_dir.std(dim=(0, 1)).cpu()
    xs = torch.arange(1, head_dim + 1)

    values_path = output_dir / "scaled_key_eigenvalues.csv"
    with values_path.open("w") as f:
        f.write("rank,mean,std\n")
        for idx in range(head_dim):
            f.write(f"{idx + 1},{mean_curve[idx].item():.10e},{std_curve[idx].item():.10e}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(xs.numpy(), mean_curve.numpy())
    plt.yscale("log")
    plt.xlabel("Rank (i)")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of Σ_q^(1/2) Cov(k) Σ_q^(1/2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    plt.close()

    zoom_path = output_dir / "var_dir_curve_zoom.png"
    plt.figure(figsize=(8, 5))
    max_rank = min(32, head_dim)
    plt.plot(xs[:max_rank].numpy(), mean_curve[:max_rank].numpy())
    plt.yscale("log")
    plt.xlabel("Rank (i)")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of Σ_q^(1/2) Cov(k) Σ_q^(1/2) (top 32)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(zoom_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
