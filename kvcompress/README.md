# KV-Cache Compression via Whitened Query Space

Novel KV-cache compression methods using whitened query distributions and least-squares chunk merging for efficient long-context inference.

## Overview

This repository presents two novel approaches to KV-cache compression that leverage the geometric structure of query and key distributions:

1. **LSE-correct method**: Least-squares fitted chunk merging with certified error bounds in log-domain
2. **Whitened-chunk method**: Mean L2 distance merging in whitened key space

Unlike prior work that uses heuristics or single-direction projections, we directly fit the empirical query distribution and only merge when we can certify low distortion.

## Key Contributions

- **Whitened geometry**: We recognize that only relative key displacements matter and that queries are anisotropic. By whitening queries and reweighting keys, we work in the correct geometric space.
- **Certified merging**: We merge chunks only when the reconstruction error under the empirical query distribution is below a user-specified threshold ε.
- **Full pipeline**: Complete implementation from data dumping → statistics computation → projection training → decoding with novel cache methods.

## Installation

```bash
# Clone repository with submodules
git clone --recurse-submodules <repo-url>
cd tildetakehome

# Install dependencies
pip install uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Repository Structure

```
tildetakehome/
├── src/                           # Novel cache implementations (YOUR methods)
│   ├── lse_cache_correct.py       # LSE-correct: least-squares merge with error certification
│   ├── whitened_noquery_cache.py  # Whitened-chunk: L2 merging in whitened space
│   └── whitened_chunk_cache.py    # Variant implementation
├── baselines/                     # Baseline implementations for comparison
│   └── kvmerger_cache.py          # KVMerger reimplementation
├── experiments/                   # Runner scripts and utilities
│   ├── data_prep/                 # Data dumping and statistics
│   │   ├── dump_qkv.py            # Step 1: Dump Q/K/V/O tensors
│   │   ├── compute_qk_stats.py    # Step 2: Compute Σ_q, Cov_k statistics
│   │   ├── prepare_projection_init.py  # Initialize projection matrices
│   │   └── key_heatmaps.py        # Visualize key similarity
│   ├── training/                  # Projection and model training
│   │   ├── train_projections.py   # Optimize low-rank projections
│   │   └── train_mlp_keys.py      # Alternative MLP-based approach
│   ├── evaluation/                # Testing and benchmarking
│   │   ├── test_lse_epsilon.py    # Epsilon sensitivity analysis
│   │   ├── final_comparison.py    # Compare all methods
│   │   └── aggregate_*.py         # Aggregate results
│   ├── visualization/             # Plotting utilities
│   │   ├── plot_compression_vs_ppl.py
│   │   ├── plot_final_comparison.py
│   │   └── plot_*.py
│   ├── run_lse_correct.py         # Run LSE-correct method (YOUR METHOD)
│   ├── run_whitened_chunk.py      # Run whitened-chunk method (YOUR METHOD)
│   ├── run_kvmerger.py            # Run KVMerger baseline
│   ├── run_qfilter.py             # Run Q-Filters baseline
│   └── run_baseline.py            # Standard cache baseline
├── modeling/                      # Custom model implementations
│   └── modeling_llama.py          # Modified Llama with cache hooks
├── qfilters/                      # Q-Filters baseline (git submodule)
├── analysis/                      # Experimental results (tracked in git)
├── logs/                          # Run logs (tracked in git)
├── dumps/                         # Cached tensor dumps
└── docs/                          # Documentation
    └── paper.tex                  # Full technical writeup
```

## Full Pipeline: From Data to Results

### Step 1: Dump Q/K/V/O Tensors

First, dump attention tensors from a pretrained model on a dataset:

```bash
python experiments/data_prep/dump_qkv.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset PatrickHaller/fineweb-1B \
  --output-dir dumps/deepseek_r1_qkv \
  --batch-size 96 \
  --total-batches 40 \
  --seq-len 1024
```

This generates per-layer, per-head Q/K/V/O dumps stored as PyTorch tensors.

### Step 2: Compute Statistics

Compute query second moments (Σ_q), key covariance (Cov_k), and key means:

```bash
python experiments/data_prep/compute_qk_stats.py \
  --dump-root dumps/deepseek_r1_qkv \
  --output analysis/qk_stats.pt
```

Initialize projection matrices from the eigenspectrum of Σ_q^{1/2} Cov_k Σ_q^{1/2}:

```bash
python experiments/data_prep/prepare_projection_init.py \
  --stats-path analysis/qk_stats.pt \
  --output analysis/projection_init.pt
```

### Step 3: Train Projection Matrices (Optional)

Optimize low-rank projections to minimize attention reconstruction loss:

```bash
python experiments/training/train_projections.py \
  --dump-root dumps/deepseek_r1_qkv \
  --stats-path analysis/qk_stats.pt \
  --init-path analysis/projection_init.pt \
  --output-dir analysis/projections \
  --ranks 1 2 4 8 16 32 64 128 \
  --max-steps 200
```

This learns per-rank projection matrices P_r that minimize ||Attn(Q, P_r K, V) - O||².

### Step 4: Run

#### LSE-Correct Method

Merges chunks via least-squares fitting with certified error bounds:

```bash
python experiments/run_lse_correct.py \
  --epsilons 0.01 0.1 1 10 100 \
  --projection-rank 8 \
  --projection-cache-dir analysis/projections \
  --stats-path analysis/qk_stats.pt \
  --output-dir analysis/lse_correct \
  --num-prompts 30 \
  --max-new-tokens 256
```

**Key parameters:**
- `--epsilons`: Error tolerance thresholds to sweep
- `--projection-rank`: Dimensionality of key subspace (1-128)
- `--solver-query-count`: Number of queries for least-squares solve
- `--ridge`: Ridge regularization for pseudoinverse

#### Whitened-Chunk Method

Merges based on L2 distance in whitened key space:

```bash
python experiments/run_whitened_chunk.py \
  --epsilons 0.01 0.1 1 10 100 \
  --stats-path analysis/qk_stats.pt \
  --output-dir analysis/whitened_chunk \
  --num-prompts 30 \
  --max-new-tokens 256
```

**Key parameters:**
- `--epsilons`: L2 distance threshold in whitened space
- `--stats-path`: Path to precomputed Σ_q, Cov_k statistics

### Step 5: Run Baselines for Comparison

#### KVMerger Baseline

Gaussian-weighted merging with cosine or L2 thresholds:

```bash
python experiments/run_kvmerger.py \
  --merge-interval 4 \
  --merge-window 8 \
  --l2-threshold 14.85 \
  --use-whitening \
  --stats-path analysis/qk_stats.pt \
  --output-dir analysis/kvmerger
```

#### Q-Filters Baseline

Uses the Q-Filters submodule for single-direction projection:

```bash
python experiments/run_qfilter.py \
  --output-dir analysis/qfilter
```

#### Standard Baseline

No compression, standard KV cache:

```bash
python experiments/run_baseline.py \
  --output-dir analysis/baseline
```

### Step 6: Evaluate and Visualize

Compare all methods:

```bash
python experiments/evaluation/final_comparison.py \
  --output-dir analysis/final_comparison

python experiments/visualization/plot_compression_vs_ppl.py \
  --output analysis/compression_vs_perplexity.png
```

## Results

Results are stored in:
- `analysis/`: JSON metrics, plots, aggregated statistics
- `logs/`: Full run logs with per-step outputs
- `dumps/`: Generated text and chunk structures

Key findings (see `docs/paper.tex` for full details):
- Whitened geometry captures 10× more variance in top eigenvector
- LSE-correct achieves (TODO - runs not finished yet :c)
- Simple chunking in the whitened key space has comparable tentative results to QFilters and KVMerger, though the variance in perplexity across epsilons is quite a bit and we wait for time to evaluate on more data to draw conclusions.
- Whitened-chunk offers simpler alternative with comparable performance

## Approach

For full technical details, motivation, and related work, see the LaTeX writeup in `docs/paper.tex`.

## License

(later)

## Acknowledgments

This codebase uses:
- [Q-Filters](https://arxiv.org/abs/2503.02812) as a baseline (git submodule in `qfilters/`)
- [KVMerger](https://arxiv.org/abs/2407.08454) reimplemented in `baselines/kvmerger_cache.py`