#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate cosine and L2 heatmaps for centered keys.")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/key_heatmaps"))
    parser.add_argument("--batch-index", type=int, default=0, help="Batch file index to visualize.")
    parser.add_argument("--example-index", type=int, default=0, help="Batch example to visualize.")
    return parser.parse_args()


def load_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path)
    return {"k_mean": stats["k_mean"].to(torch.float32)}


def load_keys(layer_dir: Path, batch_idx: int) -> torch.Tensor:
    batch_path = layer_dir / f"batch_{batch_idx:04d}.pt"
    payload = torch.load(batch_path, map_location="cpu")
    return payload["k"].to(torch.float32)  # (batch, seq, kv_heads, dim)


def compute_cosine(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(normalized, normalized.transpose(0, 1))


def compute_l2(matrix: torch.Tensor) -> torch.Tensor:
    diff = matrix.unsqueeze(1) - matrix.unsqueeze(0)
    return diff.norm(dim=-1)


def save_heatmap(data: torch.Tensor, path: Path, title: str, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    stats = load_stats(args.stats_path)
    k_mean = stats["k_mean"]  # (layers, kv_heads, dim)

    layers = sorted(
        path for path in args.dump_root.iterdir() if path.is_dir() and path.name.startswith("layer_")
    )

    for layer_dir in layers:
        layer_idx = int(layer_dir.name.split("_")[-1])
        keys = load_keys(layer_dir, args.batch_index)  # (batch, seq, kv_heads, dim)
        sample = keys[args.example_index]  # (seq, kv_heads, dim)
        for kv_head in range(sample.shape[1]):
            head_vecs = sample[:, kv_head, :] - k_mean[layer_idx, kv_head]
            cosine = compute_cosine(head_vecs)
            l2 = compute_l2(head_vecs)

            base_name = f"layer_{layer_idx:02d}_head_{kv_head:02d}"
            base_dir = args.output_dir
            save_heatmap(
                cosine.cpu(),
                base_dir / f"{base_name}_cos.png",
                title=f"Layer {layer_idx} Head {kv_head} Cosine",
            )
            save_heatmap(
                l2.cpu(),
                base_dir / f"{base_name}_l2.png",
                title=f"Layer {layer_idx} Head {kv_head} L2",
                cmap="magma",
            )


if __name__ == "__main__":
    main()
