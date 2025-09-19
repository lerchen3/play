Native Sparse Attention (NSA) Overview

This document describes the NSA modules and kernels, their inputs/outputs, grouped selection behavior, and decode-time caching.

Key files
- `train/attention/nsa.py`: Orchestrates NSA. Provides caching and dispatches to kernels.
- `train/attention/casual_fwd.py` + `casual_bwd.py`: Causal (FlashAttention-style) kernel with sliding-window and optional top-k support.
- `train/attention/select_fwd.py` + `select_bwd.py`: Selected-attention kernels driven by per-row block indices (no masks).
- `train/attention/select.py`: Autograd wrapper for selected attention (indices-based API).

Terminology
- B: batch size
- T: sequence length
- G: number of KV heads (GQA groups)
- Hg: heads per group; n_q = G * Hg
- d: head dimension
- nb: number of compressed blocks

nsa.py (NativeSparseAttention)
- Purpose: Replace attention with a 3-branch mixture for GQA:
  - Compressed: attention over overlapping block summaries (global context).
  - Selected: full attention over a sparse set of original tokens addressed by block indices (shared across heads in group).
  - Sliding window: attention over the most recent window_size tokens (local context).
- Constructor args:
  - d_model, n_q_heads, n_kv_heads, d_head, seq_len
  - cmp_blk_size, cmp_stride: compression block size and stride
  - slc_top_n: number of top blocks to select
  - window_size: sliding-window size
- Train/prefill forward: forward(x, q, is_decoding=False)
  - Inputs: x (B,T,d_model); q (B,n_q,T,d)
  - Steps:
    1) Project per-branch K/V for cmp/slc/win; expand KV heads to Q heads for GQA.
    2) Build compressed block summaries (nb blocks) using learned compressors (two-layer MLP with GELU) over flattened blocks with additive intra-block position embeddings.
    3) Compressed branch: causal kernel over block summaries with block-causal `row_max` and optional top-k ranking. Output is per-group (B,G,T,d), broadcast to heads in group.
    4) Selection: compute per-group block scores (sum over heads in group), form per-row top-n block indices; always include sink (block 0) and the last two blocks. Build `block_idx` (B,G,T,Kmax) and `block_count` (B,G,T) and call indices-based select kernel; the kernel loads only referenced blocks directly from full K/V.
    5) Sliding window: run causal attention with WINDOW_SIZE via causal wrapper.
    6) Combine with gating: gate(x) -> (B,n_q,T,3). Output shape: (B,T,n_q*d).
  - Side-effect: prefill() caches K/V and block summaries for decode if not decoding.
- Decode forward: forward(x, q, is_decoding=True) with T=1
  - Updates caches: appends new K/V for cmp/slc/win and updates compressed block summaries incrementally with the learned compressors (only affected/new blocks).
  - Computes per-group compressed scores for the last token, sums scores across heads in group.
  - Selects top-n blocks per group (plus sink + two most recent), builds per-row indices and calls the indices-based select kernel for the last token.
  - Computes sliding-window attention for the last token via the causal kernel.
  - Combines three components with gates for the last token and returns (B,1,n_q*d).
- Caches stored:
  - `_cache_cmp_k/v`, `_cache_slc_k/v`, `_cache_win_k/v`: (B,T,G,d)
  - `_cmp_block_summary_k/v`: (B,nb,n_kv,d)
  - _block_starts: list of starting indices per block; _cached_len: total cached length
- Helper methods:
  - prefill(x,q): builds all caches from a full prompt.
  - reset_cache(): clears caches.

casual_fwd.py (Causal Kernel)
- `attention_forward(..., WINDOW_SIZE)`: Adds strict lower-triangular masking with sliding window band (excludes diagonal). Assumes BLOCK_M and BLOCK_N divide WINDOW_SIZE.
- Streaming top-k indices per query row: forward signature includes (TopVal, TopIdx, TOP_K). Maintains per-row running top-k across K tiles.

casual.py (Wrapper)
- _attention.forward(q,k,v,sm_scale, window_size=0, top_k=0) -> returns o or (o, TopIdx) if top_k>0.
- Backward passes gradient only for output tensor.

select_fwd.py/select_bwd.py (Selected Attention Kernels)
- `select_attention_forward`: indices-driven processing per (timestep, batch-group). Inputs: Q (B,G,T,d), full K/V (B,G,SeqLen,d), `block_idx` (B,G,T,Kmax), `block_count` (B,G,T), `cmp_blk_size`. The kernel loads only referenced CMP blocks, enforces causality per row, and returns O (B,G,T,d) with per-row maxima M for backward.
- `select_attention_backward`: indices-driven backward; accumulates gradients into full K/V via atomic adds.

select.py (Wrapper)
- `select_attention(q, k_full, v_full, sm_scale, block_idx, block_count, cmp_blk_size) -> (B,G,T,d)`
- Autograd wired to indices-based backward.

Grouped selection
- Selection scores are computed per group by summing attention scores across heads within the group (Hg), both in prefill and decode. The top-n set is shared by all heads in the group.

Outputs
- NSA forward returns (B,T,n_q*d) embedding.
- For decode (T=1), returns (B,1,n_q*d).

Backward support & benchmarks
- Triton backward kernels exist for every attention branch that operates on Q/K/V tensors:
  - **Causal / sliding**: `_attn_bwd` in `casual_bwd.py` propagates gradients to Q, K and V (windowed sliding attention is the same kernel with a band mask).
  - **Selected attention**: `select_attention_backward` accumulates gradients into the referenced K/V tiles via block indices.
  - **Top-k indices** are inference-only; no backward pass is needed.
- The helper script `scripts/benchmark_nsa_components.py` measures forward/backward throughput versus PyTorch references. On an A100 (fp16) we observe:

  | Component | Torch FWD (ms) | Triton FWD (ms) | FWD ratio | Torch BWD (ms) | Triton BWD (ms) | BWD ratio |
  |-----------|----------------|-----------------|-----------|----------------|-----------------|-----------|
  | causal (window=0) | 0.693 | 0.216 | 0.31× | 1.768 | 0.647 | 0.36× |
  | sliding (window=128) | 0.724 | 0.201 | 0.28× | 1.804 | 0.655 | 0.36× |
  | selected attention | 1.001 | 0.306 | 0.31× | 2.845 | 0.793 | 0.28× |
  | top-k indices¹ | 0.287 | 0.671 | 2.34× | – | – | – |

  ¹Top-k currently lags PyTorch `topk`; only the forward path is required for NSA.

Notes & assumptions
- Selected-attention is strictly indices-based; no N×N masks are materialized. Memory use scales with the number of referenced blocks.
- Compressed summaries use learned MLP compressors both in training and decoding; decoding updates are incremental (only affected/new blocks).
- Sliding window uses the causal kernel; decode uses it for the last position as well (falling back to the PyTorch implementation if the window length is not a multiple of 32).

Indices format
- `block_idx`: int32, shape (B,G,T,Kmax). Each row (b,g,t) contains up to `block_count[b,g,t]` valid block indices; the rest are ignored.
- `block_count`: int32, shape (B,G,T). Number of valid indices per row.
- `cmp_blk_size`: int, the size of each compressed/selected block. Token indices covered by block `bi` are `[bi*cmp_blk_size, min((bi+1)*cmp_blk_size, SeqLen))`.
