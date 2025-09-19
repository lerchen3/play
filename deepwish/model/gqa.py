import math

import torch
import torch.nn as nn

from .base_attention import BaseAttention
from .rope import RoPE

try:
    from .decode_triton import single_query_attention
except Exception:  # pragma: no cover - Triton runtime not available
    single_query_attention = None


class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention as used in models like Qwen3.
    Inherits core attention computation from BaseAttention.
    """
    def __init__(
        self,
        d_model,
        n_q_heads,
        n_kv_heads,
        seq_len,
        d_head=None,
        use_nsa=False,
        window_size=None,
        cmp_blk_size=32,
        cmp_stride=16,
        slc_top_n=16,
    ):
        # Use d_head based on query heads for consistency
        d_head = d_head or (d_model // n_q_heads)
        super().__init__(d_model, n_q_heads, d_head, seq_len, use_nsa=use_nsa, window_size=window_size)
        
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        
        # Query projection (full number of heads)
        self.W_q = nn.Linear(d_model, n_q_heads * self.d_head, bias=False)
        # Key and Value projections (fewer heads for GQA)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        # Output projection
        self.W_o = nn.Linear(n_q_heads * self.d_head, d_model, bias=False)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.d_head, seq_len)
        self.nsa_cmp_blk_size = cmp_blk_size
        self.nsa_cmp_stride = cmp_stride
        self.nsa_slc_top_n = slc_top_n
        
    def forward(self, x, cached_k=None, cached_v=None, cached_len=0, return_cache=False):
        B, S, D = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x)  # (B, S, n_q_heads * d_head)
        k = self.W_k(x)  # (B, S, n_kv_heads * d_head)
        v = self.W_v(x)  # (B, S, n_kv_heads * d_head)
        
        # Reshape to head dimensions
        q = q.view(B, S, self.n_q_heads, self.d_head)
        k = k.view(B, S, self.n_kv_heads, self.d_head)
        v = v.view(B, S, self.n_kv_heads, self.d_head)
        
        # Apply RoPE
        q = self.rope(q, offset=cached_len)
        k = self.rope(k, offset=cached_len)
        
        # Handle KV caching for inference
        if cached_k is not None and cached_v is not None and cached_len > 0:
            # Concatenate with cached K, V
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        T = k.shape[1]  # Total sequence length including cache

        # Expand K, V for grouped query attention
        # Repeat each KV head to match the number of Q heads
        k_kv = k
        v_kv = v
        n_rep = self.n_q_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(B, T, n_rep, self.n_kv_heads, self.d_head).reshape(B, T, self.n_q_heads, self.d_head)
            v = v.unsqueeze(2).expand(B, T, n_rep, self.n_kv_heads, self.d_head).reshape(B, T, self.n_q_heads, self.d_head)
        
        # Transpose for attention: (B, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Determine if this is single-token decoding
        is_decoding = (cached_len > 0 and S == 1)
        
        # NSA path (currently supported only for GQA): dispatch to NSA when enabled
        use_nsa = self.use_nsa and x.is_cuda and torch.cuda.is_available()
        if use_nsa:
            from train.attention.nsa import NativeSparseAttention
            if not hasattr(self, "_nsa"):
                # Lazy init to avoid circular import at module import time
                self._nsa = NativeSparseAttention(
                    d_model=self.d_model,
                    n_q_heads=self.n_q_heads,
                    n_kv_heads=self.n_kv_heads,
                    d_head=self.d_head,
                    seq_len=self.seq_len,
                    cmp_blk_size=self.nsa_cmp_blk_size,
                    cmp_stride=self.nsa_cmp_stride,
                    slc_top_n=self.nsa_slc_top_n,
                    window_size=self.window_size or self.seq_len,
                ).to(x.device)
            # NSA uses its own K/V projections per branch; pass x and q
            out = self._nsa(x, q, is_decoding=is_decoding)  # (B, S, n_q*d)
            if return_cache and not is_decoding:
                with torch.no_grad():
                    self._nsa.prefill(x, q)
        else:
            if (
                is_decoding
                and S == 1
                and single_query_attention is not None
                and x.device.type == "cuda"
            ):
                attn = single_query_attention(q.contiguous(), k.contiguous(), v.contiguous(), self.scale)
            else:
                attn = self.compute_attention(q, k, v, is_decoding)
            out = attn.transpose(1, 2).reshape(B, S, self.n_q_heads * self.d_head)

        # Reshape and project output
        out = self.W_o(out)

        if return_cache:
            if use_nsa:
                return out, None, None
            # Return the new K, V for caching (only new tokens) in KV-head layout
            new_k = k_kv[:, -S:, :, :].contiguous()
            new_v = v_kv[:, -S:, :, :].contiguous()
            return out, new_k, new_v

        return out
