import torch

try:
    import triton
    from .topk_fwd import topk_forward
    _HAS_TRITON_TOPK = True
except Exception:  # pragma: no cover - best-effort Triton import
    _HAS_TRITON_TOPK = False

# Disable Triton path for now; torch.topk is faster and numerically exact
_HAS_TRITON_TOPK = False


def topk_indices(q_group: torch.Tensor, k_full: torch.Tensor, sm_scale: float, top_k: int, row_max: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row top-k column indices using Triton.

    Args:
      q_group: (B, G, T, d) queries summed over heads within group.
      k_full:  (B, G, N, d) full keys at group granularity (not duplicated across Hg).
      sm_scale: float scaling factor.
      top_k: number of indices to return per row.
      row_max: (B, G, T) int32 max-allowed column for causality (inclusive) or None for no causal cap.

    Returns:
      TopIdx: (B, G, T, top_k) int32 indices per row.
    """
    assert q_group.ndim == 4 and k_full.ndim == 4
    B, G, T, D = q_group.shape
    N = k_full.shape[2]
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    def _next_power_of_two(n: int) -> int:
        return 1 << (n - 1).bit_length()

    kernel_top_k = _next_power_of_two(top_k)
    TopIdx_full = torch.full((B, G, T, kernel_top_k), -1, device=q_group.device, dtype=torch.int32)
    if _HAS_TRITON_TOPK and N >= 32:
        if row_max is None:
            row_max = torch.full((B, G, T), N - 1, device=q_group.device, dtype=torch.int32)
        else:
            row_max = row_max.to(device=q_group.device, dtype=torch.int32).contiguous()
        grid = (T, B * G)
        max_tiles = (N + 32 - 1) // 32
        topk_forward[grid](
            q_group, k_full, sm_scale, TopIdx_full,
            q_group.stride(0), q_group.stride(1), q_group.stride(2), q_group.stride(3),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            TopIdx_full.stride(0), TopIdx_full.stride(1), TopIdx_full.stride(2), TopIdx_full.stride(3),
            row_max, row_max.stride(0), row_max.stride(1), row_max.stride(2),
            B, G, T, N,
            TOP_K=kernel_top_k,
            HEAD_DIM=D,
            BLOCK_N=32,
            MAX_TILES=max_tiles,
        )
        return TopIdx_full[..., :top_k]

    # Fallback: dense torch.topk with causal mask
    if row_max is None:
        row_max = torch.full((B, G, T), N - 1, device=q_group.device, dtype=torch.int32)
    else:
        row_max = row_max.to(device=q_group.device, dtype=torch.int32)

    scores = torch.matmul(q_group.float(), k_full.float().transpose(-2, -1)) * sm_scale
    col_indices = torch.arange(N, device=q_group.device)
    causal_mask = col_indices.view(1, 1, 1, N) <= row_max.view(B, G, T, 1)
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    topk_vals, topk_idx = torch.topk(scores, top_k, dim=-1)
    return topk_idx.to(torch.int32)
