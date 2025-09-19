"""Shared helpers for Triton-based cross entropy losses."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from train.kernels.cce import TritonCCE


def triton_cce_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: Optional[int] = None,
    block_size: int = 64,
) -> torch.Tensor:
    """Compute cross entropy using the Triton CCE kernel with PyTorch fallback.

    Args:
        hidden: Tensor of shape (batch, seq, dim) or (N, dim).
        weight: Classification weight matrix of shape (vocab, dim).
        targets: Integer tensor with target token ids.
        ignore_index: Optional index to ignore in the loss.
        block_size: Block size used by the Triton kernel.

    Returns:
        Scalar tensor containing the mean loss over non-ignored targets.
    """

    if hidden.dim() == 3:
        features = hidden.view(-1, hidden.size(-1))
    elif hidden.dim() == 2:
        features = hidden
    else:
        raise ValueError(f"hidden must be rank 2 or 3, got shape {hidden.shape}")

    targets_flat = targets.reshape(-1).to(torch.long)

    if ignore_index is not None:
        mask = targets_flat != ignore_index
    else:
        mask = torch.ones_like(targets_flat, dtype=torch.bool)

    if mask.sum() == 0:
        return features.new_zeros(())

    features = features[mask]
    targets_flat = targets_flat[mask]

    if hidden.is_cuda and weight.is_cuda and targets.is_cuda:
        loss = TritonCCE.apply(features, weight, targets_flat, block_size)
        return loss

    logits = features @ weight.t().to(features.dtype)
    loss = F.cross_entropy(logits, targets_flat, reduction="mean")
    return loss


__all__ = ["triton_cce_loss"]
