"""High level Qwen3-Next network definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..cce import triton_cce_loss
from .config import Qwen3NextConfig
from .kernels import ZeroCenteredRMSNorm
from .layers import (
    Qwen3NextDecoderLayer,
    Qwen3NextRotaryEmbedding,
)


@dataclass
class Qwen3NextModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class Qwen3NextCausalLMOutput:
    loss: Optional[torch.Tensor]
    aux_loss: Optional[torch.Tensor]
    logits: torch.Tensor
    mtp_logits: Optional[torch.Tensor]
    mtp_loss: Optional[torch.Tensor]
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    attentions: Optional[Tuple[torch.Tensor, ...]]
    router_logits: Optional[Tuple[torch.Tensor, ...]]


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def _load_balancing_loss(
    router_logits: Tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    if not router_logits:
        return torch.tensor(0.0)
    device = router_logits[0].device
    stacked = torch.cat([logits.reshape(-1, logits.shape[-1]) for logits in router_logits], dim=0)
    routing = torch.softmax(stacked, dim=-1, dtype=torch.float32)
    _, selected = torch.topk(routing, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected, num_classes=num_experts).float()
    tokens_per_expert = expert_mask.mean(dim=0)
    prob_per_expert = routing.mean(dim=0)
    return ((tokens_per_expert * prob_per_expert).sum() * num_experts).to(device)


class Qwen3NextModel(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3NextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = ZeroCenteredRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )
        self.rotary_emb = Qwen3NextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> Qwen3NextModelOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        cos, sin = self.rotary_emb(seq_len, device, dtype)
        causal_bias = _causal_mask(seq_len, device, dtype).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            padding_bias = (1.0 - attention_mask.float())[:, None, None, :]
            padding_bias = padding_bias.to(dtype) * -1e9
            attention_bias = causal_bias + padding_bias
        else:
            attention_bias = causal_bias

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_router_logits = [] if output_router_logits else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, attn_probs, router_logits = layer(
                hidden_states,
                cos,
                sin,
                attention_bias,
                attention_mask,
            )
            if all_attentions is not None:
                all_attentions.append(attn_probs)
            if all_router_logits is not None and router_logits is not None:
                all_router_logits.append(router_logits)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3NextModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
            router_logits=tuple(all_router_logits) if all_router_logits is not None else None,
        )

    def reset_inference_state(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_state"):
                layer.reset_state()


class Qwen3NextForCausalLM(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3NextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_heads = (
            nn.ModuleList(
                [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.mtp_depth)]
            )
            if config.mtp_depth > 0
            else None
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mtp_targets: Optional[torch.LongTensor] = None,
        mtp_ignore_index: int = -100,
        output_mtp_logits: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> Qwen3NextCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = triton_cce_loss(
                hidden_states,
                self.lm_head.weight,
                labels,
                ignore_index=-100,
            )

        stacked_mtp = None
        mtp_logits = None
        if self.mtp_heads is not None and (mtp_targets is not None or output_mtp_logits):
            stacked_mtp = torch.stack(
                [head(hidden_states) for head in self.mtp_heads],
                dim=2,
            )
            if output_mtp_logits:
                mtp_logits = stacked_mtp

        mtp_loss = None
        if stacked_mtp is not None and mtp_targets is not None:
            depth = len(self.mtp_heads) if self.mtp_heads is not None else 0
            if depth == 0:
                mtp_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                losses = []
                for head_idx, head in enumerate(self.mtp_heads):
                    targets_depth = mtp_targets[:, :, head_idx]
                    loss_head = triton_cce_loss(
                        hidden_states,
                        head.weight,
                        targets_depth,
                        ignore_index=mtp_ignore_index,
                    )
                    losses.append(loss_head)
                mtp_loss = torch.stack(losses).mean()

        aux_loss = None
        if output_router_logits and outputs.router_logits:
            aux_loss = _load_balancing_loss(outputs.router_logits, self.config.num_experts, self.config.num_experts_per_tok)
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3NextCausalLMOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            mtp_logits=mtp_logits,
            mtp_loss=mtp_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def generate(self, *args, **kwargs):  # pragma: no cover - simple delegation helper
        raise NotImplementedError("Generation helpers are not implemented in this lightweight reference")
