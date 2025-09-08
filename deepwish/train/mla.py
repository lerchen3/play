import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.attention import BaseAttention
from train.rope import RoPE


class MLA(BaseAttention):
    def __init__(self, d_model, n_heads, dc_kv, dc_q, seq_len, device, d_head_override=None, window_size=None):
        d_head = d_model//n_heads if d_head_override is None else d_head_override
        super().__init__(d_model, n_heads, d_head, seq_len, use_nsa=False, window_size=window_size)
        self.dc_kv = dc_kv
        self.dc_q = dc_q
        self.W_dkv = nn.Parameter(torch.randn(d_model, dc_kv) / math.sqrt(d_model))
        self.W_uk  = nn.Parameter(torch.randn(dc_kv, n_heads * self.d_head) / math.sqrt(dc_kv))
        self.W_uv  = nn.Parameter(torch.randn(dc_kv, n_heads * 2 * self.d_head) / math.sqrt(dc_kv))
        self.W_kr  = nn.Parameter(torch.randn(d_model, n_heads * self.d_head) / math.sqrt(d_model))
        self.rope_k = RoPE(self.d_head, seq_len)
        self.W_dq  = nn.Parameter(torch.randn(d_model, dc_q) / math.sqrt(d_model))
        self.W_uq  = nn.Parameter(torch.randn(dc_q, n_heads * self.d_head) / math.sqrt(dc_q))
        self.W_qr  = nn.Parameter(torch.randn(dc_q, n_heads * self.d_head) / math.sqrt(dc_q))
        self.rope_q = RoPE(self.d_head, seq_len)
        self.W_o   = nn.Parameter(torch.randn(n_heads * 2 * self.d_head, d_model) / math.sqrt(n_heads * 2 * self.d_head))
        # Override scale for MLA (uses 2 * d_head)
        self.scale = (self.d_head * 2) ** -0.5

    def forward(self, h, cached_c_kv=None, cached_kR=None, cached_len=0, return_latent=False):
        # Attention with optional latent caching: cache c_kv and kR for prefix; recompute K,V from latents each step
        B, S, _ = h.size()
        flat_suffix = h.view(-1, self.d_model)

        # Build c_kv for full context (prefix from cache, suffix newly computed)
        if cached_c_kv is not None and cached_len > 0:
            # Validate cached tensors before use
            if not torch.isfinite(cached_c_kv).all():
                print("Warning: NaN/Inf detected in cached_c_kv, reinitializing...")
                cached_c_kv = None
                cached_len = 0
            
        if cached_c_kv is not None and cached_len > 0:
            dc_kv = cached_c_kv.size(2)
            c_kv_suffix = (flat_suffix @ self.W_dkv).view(B, S, dc_kv)
            
            # Validate computation before concatenation
            if not torch.isfinite(c_kv_suffix).all():
                print("Warning: NaN/Inf in c_kv_suffix computation")
                c_kv_suffix = torch.zeros_like(c_kv_suffix)
            
            # Ensure device and dtype compatibility
            if cached_c_kv.device != c_kv_suffix.device:
                cached_c_kv = cached_c_kv.to(c_kv_suffix.device)
            if cached_c_kv.dtype != c_kv_suffix.dtype:
                cached_c_kv = cached_c_kv.to(c_kv_suffix.dtype)
                
            c_kv_full = torch.cat([cached_c_kv, c_kv_suffix], dim=1)
            T = cached_len + S
        else:
            # compute for all positions (prefill)
            dc_kv = self.W_uk.size(0)
            c_kv_full = (flat_suffix @ self.W_dkv).view(B, S, dc_kv)
            T = S

        # Build ku and vu from c_kv_full
        flat_c = c_kv_full.view(-1, c_kv_full.size(-1))
        ku = (flat_c @ self.W_uk).view(B, T, self.n_heads, self.d_head)
        vu = (flat_c @ self.W_uv).view(B, T, self.n_heads, 2 * self.d_head)

        # Build kR for full context
        if cached_kR is not None and cached_len > 0:
            # Validate cached kR tensors
            if not torch.isfinite(cached_kR).all():
                print("Warning: NaN/Inf detected in cached_kR, reinitializing...")
                cached_kR = None
                cached_len = 0
                
        if cached_kR is not None and cached_len > 0:
            kR_suffix = self.rope_k((flat_suffix @ self.W_kr).view(B, S, self.n_heads, self.d_head), offset=cached_len)
            
            # Validate kR computation
            if not torch.isfinite(kR_suffix).all():
                print("Warning: NaN/Inf in kR_suffix computation")
                kR_suffix = torch.zeros_like(kR_suffix)
            
            # Ensure compatibility
            if cached_kR.device != kR_suffix.device:
                cached_kR = cached_kR.to(kR_suffix.device)
            if cached_kR.dtype != kR_suffix.dtype:
                cached_kR = cached_kR.to(kR_suffix.dtype)
                
            kR_full = torch.cat([cached_kR, kR_suffix], dim=1)
        else:
            kR_full = self.rope_k((flat_suffix @ self.W_kr).view(B, S, self.n_heads, self.d_head), offset=0)

        # Compose K and V
        k = torch.cat((ku, kR_full), dim=-1).permute(0, 2, 1, 3)  # (B, H, T, 2*dh)
        v = vu.permute(0, 2, 1, 3)  # (B, H, T, 2*dh)

        # Build Q: prefix zeros, suffix real queries
        if cached_len > 0:
            # suffix queries
            c_q_suffix = flat_suffix @ self.W_dq
            qU_suffix = (c_q_suffix @ self.W_uq).view(B, S, self.n_heads, self.d_head)
            qR_suffix = self.rope_q((c_q_suffix @ self.W_qr).view(B, S, self.n_heads, self.d_head), offset=cached_len)
            q_suffix = torch.cat((qU_suffix, qR_suffix), dim=-1)
            # prefix zeros
            q_prefix = torch.zeros((B, cached_len, self.n_heads, 2 * self.d_head), device=h.device, dtype=q_suffix.dtype)
            q_full = torch.cat([q_prefix, q_suffix], dim=1).permute(0, 2, 1, 3)  # (B, H, T, 2*dh)
        else:
            c_q = flat_suffix @ self.W_dq
            qU = (c_q @ self.W_uq).view(B, S, self.n_heads, self.d_head)
            qR = self.rope_q((c_q @ self.W_qr).view(B, S, self.n_heads, self.d_head), offset=0)
            q_full = torch.cat((qU, qR), dim=-1).permute(0, 2, 1, 3)

        # Determine if this is single-token decoding
        is_decoding = (cached_len > 0 and S == 1)
        
        # Use inherited attention computation
        out = self.compute_attention(q_full, k, v, is_decoding)
        
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        out = out @ self.W_o
        
        # Final validation of output
        if not torch.isfinite(out).all():
            print("Warning: NaN/Inf in final MLA output after W_o projection")
            out = torch.zeros_like(out)

        if cached_len > 0:
            # Return only suffix output aligned with input h
            out_suffix = out[:, -S:, :]
        else:
            out_suffix = out

        if return_latent:
            return out_suffix, c_kv_full, kR_full
        return out_suffix