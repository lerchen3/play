"""Training wrapper around the Qwen3-Next causal language model."""

from __future__ import annotations

import torch
from torch import nn

from .config import Qwen3NextConfig
from .model import Qwen3NextForCausalLM


class Qwen3NextTrainModel(nn.Module):
    """Minimal training interface compatible with ``train/train.py``."""

    def __init__(self, args) -> None:
        super().__init__()
        base = Qwen3NextConfig()
        self.config = Qwen3NextConfig(
            vocab_size=args.vocab_size,
            hidden_size=getattr(args, "d_model", base.hidden_size),
            intermediate_size=getattr(args, "d_ff", base.intermediate_size),
            num_hidden_layers=getattr(args, "num_layers", base.num_hidden_layers),
            num_attention_heads=getattr(args, "n_q_heads", base.num_attention_heads),
            num_key_value_heads=getattr(args, "n_kv_heads", base.num_key_value_heads),
            num_experts=getattr(args, "num_experts", 0),
            num_experts_per_tok=getattr(args, "num_experts_per_tok", base.num_experts_per_tok),
            decoder_sparse_step=getattr(args, "decoder_sparse_step", base.decoder_sparse_step),
            linear_num_key_heads=getattr(args, "linear_num_key_heads", base.linear_num_key_heads),
            linear_num_value_heads=getattr(args, "linear_num_value_heads", base.linear_num_value_heads),
            linear_key_head_dim=getattr(args, "linear_key_head_dim", base.linear_key_head_dim),
            linear_value_head_dim=getattr(args, "linear_value_head_dim", base.linear_value_head_dim),
            linear_conv_kernel_dim=getattr(args, "linear_conv_kernel_dim", base.linear_conv_kernel_dim),
            mtp_depth=getattr(args, "mtp_depth", base.mtp_depth),
        )
        self.pad_token_id = args.pad_token_id
        self.model = Qwen3NextForCausalLM(self.config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        target_main: Optional[torch.LongTensor] = None,
        tgt_matrix: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_main,
            mtp_targets=tgt_matrix if self.config.mtp_depth > 0 else None,
            mtp_ignore_index=self.pad_token_id,
            output_mtp_logits=is_training,
        )
        zero = self.model.lm_head.weight.sum() * 0.0
        loss_main = outputs.loss if outputs.loss is not None else zero
        loss_mtp = outputs.mtp_loss if outputs.mtp_loss is not None else zero
        return torch.stack([loss_main, loss_mtp])


__all__ = ["Qwen3NextTrainModel"]
