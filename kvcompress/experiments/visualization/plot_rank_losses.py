#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot projection vs MLP eval losses across ranks.")
    parser.add_argument("--projection-dir", type=Path, default=Path("analysis/projections"))
    parser.add_argument("--mlp-dir", type=Path, default=Path("analysis/mlp"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis/rank_eval_losses.pt"))
    parser.add_argument("--output-plot", type=Path, default=Path("analysis/rank_eval_loss.png"))
    return parser.parse_args()


def load_metrics(root: Path, prefix: str) -> Dict[int, float]:
    metrics: Dict[int, float] = {}
    for path in sorted(root.glob(f"{prefix}_rank_*.pt")):
        payload = torch.load(path)
        rank = int(path.stem.split("_")[-1])
        losses = payload["eval_loss"]
        metrics[rank] = losses.float().mean().item()
    return metrics


def ensure_ranks_consistent(*dicts: Dict[int, float]) -> List[int]:
    keys = [set(d.keys()) for d in dicts if d]
    common = set.intersection(*keys) if keys else set()
    return sorted(common)


def plot_curves(ranks: List[int], projection: Dict[int, float], mlp: Dict[int, float], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    proj_vals = [projection[r] for r in ranks]
    mlp_vals = [mlp[r] for r in ranks]
    plt.plot(ranks, proj_vals, marker="o", label="Projection")
    plt.plot(ranks, mlp_vals, marker="s", label="MLP")
    
    # Add horizontal reference line at variance of output values
    # This represents the mean squared magnitude of the output
    output_variance = 0.000115
    plt.axhline(y=output_variance, color='red', linestyle='--', linewidth=1.5, 
                label=f'Output Variance ({output_variance:.6f})', alpha=0.7)
    
    plt.xlabel("Rank / Output Dim")
    plt.ylabel("Eval MSE")
    plt.title("Eval Loss vs Rank")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    projection_metrics = load_metrics(args.projection_dir, "projection")
    mlp_metrics = load_metrics(args.mlp_dir, "mlp")

    ranks = ensure_ranks_consistent(projection_metrics, mlp_metrics)

    payload = {
        "ranks": ranks,
        "projection": {r: projection_metrics[r] for r in ranks},
        "mlp": {r: mlp_metrics[r] for r in ranks},
    }
    torch.save(payload, args.output_json)

    if ranks:
        plot_curves(ranks, projection_metrics, mlp_metrics, args.output_plot)


if __name__ == "__main__":
    main()
