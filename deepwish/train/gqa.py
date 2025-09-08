import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.attention import BaseAttention
from train.rope import RoPE


class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention as used in models like Qwen3.
    Inherits core attention computation from BaseAttention.
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads, seq_len, d_head=None, use_nsa=False, window_size=None):
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
        n_rep = self.n_q_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(B, T, n_rep, self.n_kv_heads, self.d_head).reshape(B, T, self.n_q_heads, self.d_head)
            v = v.unsqueeze(2).expand(B, T, n_rep, self.n_kv_heads, self.d_head).reshape(B, T, self.n_q_heads, self.d_head)
        
        # Transpose for attention: (B, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Determine if this is single-token decoding
        is_decoding = (cached_len > 0 and S == 1)
        
        # NSA path (currently supported only for GQA): dispatch to NSA when enabled
        if self.use_nsa:
            from train.attention.nsa import NativeSparseAttention
            if not hasattr(self, "_nsa"):
                # Lazy init to avoid circular import at module import time
                self._nsa = NativeSparseAttention(
                    d_model=self.d_model,
                    n_q_heads=self.n_q_heads,
                    n_kv_heads=self.n_kv_heads,
                    d_head=self.d_head,
                    seq_len=self.seq_len,
                ).to(x.device)
            # NSA uses its own K/V projections per branch; pass x and q
            out = self._nsa(x, q, is_decoding=is_decoding)  # (B, S, n_q*d)
            if is_decoding and S > 1:
                # During prefill, prime caches
                with torch.no_grad():
                    self._nsa.prefill(x, q)
        else:
            # Use base attention computation
            out = self.compute_attention(q, k, v, is_decoding)
            out = out.transpose(1, 2).reshape(B, S, self.n_q_heads * self.d_head)
        
        # Reshape and project output
        out = self.W_o(out)
        
        if return_cache:
            # Return the new K, V for caching (only new tokens)
            new_k = k.transpose(1, 2)[:, -S:, :, :]  
            new_v = v.transpose(1, 2)[:, -S:, :, :]  
            return out, new_k, new_v
        
        return out
