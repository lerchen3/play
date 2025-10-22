#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot chunk-merge and QFilter metrics")
    parser.add_argument("--chunk-summary", type=Path, default=Path("analysis/chunk_merge_summary.json"))
    parser.add_argument("--qfilter-summary", type=Path, default=Path("analysis/qfilter_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/chunk_vs_qfilter.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunk_rows: List[Dict[str, float]] = json.loads(args.chunk_summary.read_text())
    chunk_rows = [row for row in chunk_rows if row.get("num_generations", 0) >= 30]
    chunk_rows = sorted(chunk_rows, key=lambda r: r["epsilon"])

    qfilter_row = json.loads(args.qfilter_summary.read_text())

    epsilons = [row["epsilon"] for row in chunk_rows]
    compression = [row["compression_ratio"] for row in chunk_rows]
    perplexities = [row["perplexity"] for row in chunk_rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xscale("log")
    ax1.plot(epsilons, compression, marker="o", label="Chunk Merge Compression", color="tab:blue")
    ax1.axhline(qfilter_row["compression_ratio"], color="tab:blue", linestyle="--", label="QFilter Compression")
    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("Compression Ratio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epsilons, perplexities, marker="s", color="tab:red", label="Chunk Merge Perplexity")
    ax2.axhline(qfilter_row["perplexity"], color="tab:red", linestyle=":", label="QFilter Perplexity")
    ax2.set_ylabel("Perplexity", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.1, 0.95))

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
