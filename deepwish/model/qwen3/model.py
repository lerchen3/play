"""Qwen3 transformer stack with optional Native Sparse Attention."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cce import triton_cce_loss
from ..gqa import GroupedQueryAttention
from ..rmsnorm import TorchRMSNorm


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block used in the Qwen3 architecture."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.W_gate(x)
        up = self.W_up(x)
        return self.W_down(F.silu(gate) * up)


class Qwen3TransformerBlock(nn.Module):
    """Single transformer block with grouped-query attention."""

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        seq_len: int,
        eps: float = 1e-6,
        use_nsa: bool = False,
        window_size: Optional[int] = None,
        nsa_cmp_blk_size: int = 32,
        nsa_cmp_stride: int = 16,
        nsa_slc_top_n: int = 16,
    ) -> None:
        super().__init__()
        self.use_nsa = use_nsa
        self.attn = GroupedQueryAttention(
            d_model,
            n_q_heads,
            n_kv_heads,
            seq_len,
            use_nsa=use_nsa,
            window_size=window_size,
            cmp_blk_size=nsa_cmp_blk_size,
            cmp_stride=nsa_cmp_stride,
            slc_top_n=nsa_slc_top_n,
        )
        self.ffn = SwiGLU(d_model, d_ff)
        self.attn_norm = TorchRMSNorm(d_model, eps=eps)
        self.ffn_norm = TorchRMSNorm(d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
        cached_len: int = 0,
        return_cache: bool = False,
    ):
        attn_in = self.attn_norm(x)
        attn_result = self.attn(
            attn_in,
            None if self.use_nsa else cached_k,
            None if self.use_nsa else cached_v,
            cached_len,
            return_cache=return_cache,
        )

        if return_cache:
            if isinstance(attn_result, tuple):
                attn_out, new_k, new_v = attn_result
            else:
                attn_out = attn_result
                new_k = None
                new_v = None
        else:
            attn_out = attn_result
            new_k = None
            new_v = None

        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out

        if return_cache:
            return x, new_k, new_v
        return x


class Qwen3Model(nn.Module):
    """Minimal Qwen3 transformer with optional Native Sparse Attention."""

    def __init__(self, args) -> None:
        super().__init__()
        self.d_model = getattr(args, 'd_model', 1024)
        self.num_layers = getattr(args, 'num_layers', 28)
        self.n_q_heads = getattr(args, 'n_q_heads', 16)
        self.n_kv_heads = getattr(args, 'n_kv_heads', 8)
        self.d_ff = getattr(args, 'd_ff', 3072)
        self.seq_len = getattr(args, 'seq_len', 32768)
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        self.window_size = getattr(args, 'window_size', None)
        self.use_nsa = getattr(args, 'use_nsa', False)

        nsa_cmp_blk_size = getattr(args, 'nsa_cmp_blk_size', 32)
        nsa_cmp_stride = getattr(args, 'nsa_cmp_stride', 16)
        nsa_slc_top_n = getattr(args, 'nsa_slc_top_n', 16)

        # Token embeddings
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        with torch.no_grad():
            self.token_embed.weight[self.pad_token_id].zero_()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                Qwen3TransformerBlock(
                    self.d_model,
                    self.n_q_heads,
                    self.n_kv_heads,
                    self.d_ff,
                    self.seq_len,
                    use_nsa=self.use_nsa,
                    window_size=self.window_size,
                    nsa_cmp_blk_size=nsa_cmp_blk_size,
                    nsa_cmp_stride=nsa_cmp_stride,
                    nsa_slc_top_n=nsa_slc_top_n,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm = TorchRMSNorm(self.d_model, eps=getattr(args, 'rmsnorm_eps', 1e-6))
        self.head = nn.Parameter(torch.randn(self.vocab_size, self.d_model) / math.sqrt(self.d_model))

        self._cached_k: Optional[list] = None
        self._cached_v: Optional[list] = None
        self._cached_len = 0

    def reset_kv_cache(self):
        if self.use_nsa:
            self._cached_k = [None] * self.num_layers
            self._cached_v = [None] * self.num_layers
            for layer in self.layers:
                if isinstance(layer.attn, GroupedQueryAttention) and hasattr(layer.attn, "_nsa"):
                    layer.attn._nsa.reset_cache()
        else:
            self._cached_k = None
            self._cached_v = None
        self._cached_len = 0

    @torch.no_grad()
    def prefill_with_cache(self, input_ids: torch.Tensor):
        self.eval()
        self.reset_kv_cache()

        h = self.token_embed(input_ids)
        mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask

        cached_k_list = []
        cached_v_list = []
        for layer in self.layers:
            h, k_cache, v_cache = layer(h, return_cache=True)
            cached_k_list.append(k_cache)
            cached_v_list.append(v_cache)

        self._cached_k = cached_k_list
        self._cached_v = cached_v_list
        self._cached_len = input_ids.size(1)
        return h

    @torch.no_grad()
    def step_with_cache(self, input_ids_step: torch.Tensor, return_all_logits: bool = False):
        assert self._cached_k is not None and self._cached_v is not None, "Call prefill_with_cache() before step_with_cache()"

        h = self.token_embed(input_ids_step)
        mask = (input_ids_step != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask

        new_k_list = []
        new_v_list = []
        for idx, layer in enumerate(self.layers):
            cached_k = self._cached_k[idx]
            cached_v = self._cached_v[idx]
            h, new_k, new_v = layer(
                h,
                cached_k=cached_k,
                cached_v=cached_v,
                cached_len=self._cached_len,
                return_cache=True,
            )

            if new_k is None:
                new_k_list.append(None)
            else:
                if cached_k is None:
                    new_k_list.append(new_k)
                else:
                    new_k_list.append(torch.cat([cached_k, new_k], dim=1))

            if new_v is None:
                new_v_list.append(None)
            else:
                if cached_v is None:
                    new_v_list.append(new_v)
                else:
                    new_v_list.append(torch.cat([cached_v, new_v], dim=1))

        self._cached_k = new_k_list
        self._cached_v = new_v_list
        self._cached_len += input_ids_step.size(1)

        h = self.norm(h)
        if return_all_logits:
            logits = torch.matmul(h, self.head.t())
        else:
            logits = torch.matmul(h[:, -1, :], self.head.t())
        return logits

    def forward(self, input_ids: torch.Tensor, target_main: Optional[torch.Tensor] = None, is_training: bool = True):
        h = self.token_embed(input_ids)
        mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        if target_main is not None and is_training:
            return triton_cce_loss(h, self.head, target_main, ignore_index=self.pad_token_id)

        return torch.matmul(h, self.head.t())
