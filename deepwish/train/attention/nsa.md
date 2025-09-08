Native Sparse Attention (NSA) Overview

This document describes the NSA modules and kernels, their inputs/outputs, grouped selection behavior, and decode-time caching.

Key files
- train/attention/nsa.py: Orchestrates NSA. Provides caching and dispatches to kernels.
- train/attention/casual_fwd.py + casual_bwd.py: Causal (FlashAttention-style) kernel with sliding-window and optional top-k indices.
- train/attention/select_fwd.py + select_bwd.py: Selected-attention kernels with vertical-slice processing per GQA group.
- train/attention/select.py: Autograd wrapper for selected attention.

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
  - Selected: full attention over a sparse set of original tokens selected using compressed scores (shared across heads in group).
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
    3) Compressed branch: select_attention over (B,G,T,d) queries and (B,G,nb,d) K/V (shared within group by averaging across group heads). Output -> (B,G,T,d) then broadcast to heads in group.
    4) Selection: compute scores = sum over heads in each group, take top-n per group; always include sink (block 0) and last two blocks. Gather per-group tokens and run select_attention.
    5) Sliding window: run causal attention with WINDOW_SIZE via causal wrapper.
    6) Combine with gating: gate(x) -> (B,n_q,T,3). Output shape: (B,T,n_q*d).
  - Side-effect: prefill() caches K/V and block summaries for decode if not decoding.
- Decode forward: forward(x, q, is_decoding=True) with T=1
  - Updates caches: appends new K/V for cmp/slc/win and rebuilds compressed block summaries.
  - Computes per-group compressed scores for the last token, sums scores across heads in group.
  - Selects top-n blocks per group (plus sink + two most recent), gathers selected tokens, and computes attention scores for the new token using torch (single-row attention only).
  - Computes sliding-window attention for last token using torch.
  - Combines three components with gates for the last token and returns (B,1,n_q*d).
- Caches stored:
  - _cache_cmp_k/v, _cache_slc_k/v, _cache_win_k/v: (B,T,G,d)
  - _cmp_block_summary_k/v: (B,nb,G,d)
  - _block_starts: list of starting indices per block; _cached_len: total cached length
- Helper methods:
  - prefill(x,q): builds all caches from a full prompt.
  - reset_cache(): clears caches.

casual_fwd.py (Causal Kernel)
- attention_forward(..., WINDOW_SIZE): Adds strict lower-triangular masking with sliding window band (excludes diagonal). Assumes BLOCK_M and BLOCK_N divide WINDOW_SIZE.
- Streaming top-k indices per query row: forward signature includes (TopVal, TopIdx, TOP_K). Maintains per-row running top-k across all K tiles.

casual.py (Wrapper)
- _attention.forward(q,k,v,sm_scale, window_size=0, top_k=0) -> returns o or (o, TopIdx) if top_k>0.
- Backward passes gradient only for output tensor.

select_fwd.py/select_bwd.py (Selected Attention Kernels)
- select_attention_forward: vertical-slice processing over tokens per (batch, group). Inputs are (B,G,T,d) queries and (B,G,N_sel,d) selected K/V. Produces (B,G,T,d) outputs.
- _select_bwd: simplified backward that mirrors FA2 backward structure for selected tokens.

select.py (Wrapper)
- select_attention(q,k,v,sm_scale) -> (B,G,T,d)

Grouped selection
- Selection scores are computed per group by summing attention scores across heads within the group (Hg), both in prefill and decode. The top-n set is shared by all heads in the group.

Outputs
- NSA forward returns (B,T,n_q*d) embedding.
- For decode (T=1), returns (B,1,n_q*d).

Notes & assumptions
- Overlapping compression via average pooling is a reference; can be replaced by learned compressors.
- Streaming top-k indices in the causal kernel is partially implemented (API + storage). Extending to full per-row top-k across all tiles is straightforward by keeping running top-k across the K loop.
- Sliding window assumes WINDOW_SIZE is a multiple of BLOCK_M and BLOCK_N.
