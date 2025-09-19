"""Training wrapper for the Qwen3 architecture."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .model import Qwen3Model


class Qwen3TrainModel(nn.Module):
    """Adapter that exposes the Qwen3 model with a loss tuple for train.py."""

    def __init__(self, args) -> None:
        super().__init__()
        self.model = Qwen3Model(args)
        self.pad_token_id = args.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        target_main: Optional[torch.Tensor] = None,
        tgt_matrix: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> torch.Tensor:
        del tgt_matrix  # Qwen3 does not use the multi-token prediction matrix
        if is_training and target_main is None:
            raise ValueError("target_main must be provided during training")

        outputs = self.model(input_ids, target_main=target_main, is_training=is_training)
        if is_training:
            loss = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs, device=input_ids.device)
            zeros = torch.zeros_like(loss)
            return torch.stack([loss, zeros])
        return outputs


__all__ = ["Qwen3TrainModel"]
