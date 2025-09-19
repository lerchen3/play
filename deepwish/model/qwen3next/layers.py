"""Building blocks composing the Qwen3-Next network."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import Qwen3NextConfig
from .kernels import (
    GatedRMSNorm,
    TritonLinear,
    ZeroCenteredRMSNorm,
    gated_delta_rule,
    gated_delta_step,
    scaled_dot_product_attention,
)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    if attention_mask.dim() == 2:
        return hidden_states * attention_mask[:, :, None]
    return hidden_states


class Qwen3NextRotaryEmbedding(nn.Module):
    """Rotary embeddings restricted to a subset of dimensions."""

    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) + offset
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin

    query = torch.cat([q_rot, q_pass], dim=-1)
    key = torch.cat([k_rot, k_pass], dim=-1)
    return query, key


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_heads * n_rep, seq_len, head_dim)


class Qwen3NextAttention(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        hidden_size = config.hidden_size
        self.q_proj = TritonLinear(hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = TritonLinear(hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = TritonLinear(hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.gate_proj = TritonLinear(hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = TritonLinear(self.num_heads * self.head_dim, hidden_size, bias=config.attention_bias)

        self.q_norm = ZeroCenteredRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )
        self.k_norm = ZeroCenteredRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        gate = self.gate_proj(hidden_states).view(batch, seq_len, -1)

        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        query = self.q_norm(query).transpose(1, 2).contiguous()
        key = self.k_norm(key).transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        query = query.transpose(1, 2)  # (batch, seq, heads, dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        query = query.transpose(1, 2)

        attn_output, attn_probs = scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            self.scaling,
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_probs


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.in_proj_qkvz = TritonLinear(self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False)
        self.in_proj_ba = TritonLinear(self.hidden_size, self.num_v_heads * 2, bias=False)
        self.conv1d = nn.Conv1d(self.key_dim * 2 + self.value_dim, self.key_dim * 2 + self.value_dim, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size - 1, groups=self.key_dim * 2 + self.value_dim, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = GatedRMSNorm(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )
        self.out_proj = TritonLinear(self.value_dim, self.hidden_size, bias=False)

        self._inference_state: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self._inference_state = None

    def _rearrange(self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
        new_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)
        mixed_qkvz = mixed_qkvz.view(*new_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_shape_ba)

        query, key, value, z = torch.split(
            mixed_qkvz,
            [
                self.head_k_dim,
                self.head_k_dim,
                (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
                (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
            ],
            dim=3,
        )
        b, a = torch.split(
            mixed_ba,
            [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads],
            dim=3,
        )
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, padding_mask)
        batch, seq_len, _ = hidden_states.shape

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self._rearrange(projected_qkvz, projected_ba)

        query_flat, key_flat, value_flat = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        mixed = torch.cat((query_flat, key_flat, value_flat), dim=-1)
        mixed = mixed.transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed))[:, :, :seq_len]
        mixed = mixed.transpose(1, 2)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        decay = torch.exp(decay)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not self.training and seq_len == 1:
            if (
                self._inference_state is None
                or self._inference_state.size(0) != batch
            ):
                self._inference_state = None  # ensure reinit below

            state = self._inference_state
            core_step, new_state = gated_delta_step(
                query[:, 0],
                key[:, 0],
                value[:, 0],
                decay[:, 0],
                beta[:, 0],
                state,
            )
            core_out = core_step.unsqueeze(1)
            self._inference_state = new_state
        else:
            state = self._inference_state if (not self.training and self._inference_state is not None) else None
            core_out, final_state = gated_delta_rule(query, key, value, decay=decay, beta=beta, state=state)
            if not self.training:
                self._inference_state = final_state
            else:
                self._inference_state = None

        core_shape = core_out.shape
        normed = self.norm(core_out.reshape(-1, core_shape[-1]), z.reshape(-1, z.shape[-1]))
        normed = normed.reshape(core_shape[0], core_shape[1], -1)
        output = self.out_proj(normed)
        return output


class Qwen3NextMLP(nn.Module):
    def __init__(self, config: Qwen3NextConfig, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = TritonLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = TritonLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = TritonLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gated = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated), None


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        self.gate = TritonLinear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = TritonLinear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(flat_states.dtype)

        final = torch.zeros_like(flat_states)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0, as_tuple=False).flatten()
        for expert_idx in active_experts.tolist():
            expert_layer = self.experts[expert_idx]
            slot_indices, token_indices = torch.where(expert_mask[expert_idx].squeeze(0))
            if token_indices.numel() == 0:
                continue
            selected_states = flat_states[token_indices]
            expert_out, _ = expert_layer(selected_states)
            expert_out = expert_out * topk_weights[token_indices, slot_indices, None]
            final.index_add_(0, token_indices, expert_out)

        shared_out, _ = self.shared_expert(flat_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat_states))
        final = final + shared_out * shared_gate

        final = final.view(batch, seq_len, hidden_dim)
        return final, router_logits.view(batch, seq_len, -1)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            input_weight_decay=config.rms_norm_input_weight_decay,
        )

        if self.layer_type == "full_attention":
            self.token_mixer = Qwen3NextAttention(config, layer_idx)
        else:
            self.token_mixer = Qwen3NextGatedDeltaNet(config, layer_idx)

        if config.num_experts > 0 and (layer_idx % config.decoder_sparse_step == config.decoder_sparse_step - 1):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if isinstance(self.token_mixer, Qwen3NextAttention):
            attn_output, attn_probs = self.token_mixer(hidden_states, cos, sin, attention_bias)
        else:
            attn_output = self.token_mixer(hidden_states, padding_mask)
            attn_probs = None

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_out, router_logits = self.mlp(hidden_states)
        hidden_states = residual + mlp_out
        return hidden_states, attn_probs, router_logits

    def reset_state(self) -> None:
        if hasattr(self.token_mixer, "reset_state"):
            self.token_mixer.reset_state()
