import torch
import triton
from .topk_fwd import topk_forward


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
    TopIdx = torch.full((B, G, T, top_k), -1, device=q_group.device, dtype=torch.int32)
    if row_max is None:
        row_max = torch.full((B, G, T), N - 1, device=q_group.device, dtype=torch.int32)
    grid = (T, B * G)
    topk_forward[grid](
        q_group, k_full, sm_scale, TopIdx,
        q_group.stride(0), q_group.stride(1), q_group.stride(2), q_group.stride(3),
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        TopIdx.stride(0), TopIdx.stride(1), TopIdx.stride(2), TopIdx.stride(3),
        row_max, row_max.stride(0), row_max.stride(1), row_max.stride(2),
        B, G, T, N,
        TOP_K=top_k,
        HEAD_DIM=D,
        BLOCK_N=32,
    )
    return TopIdx


