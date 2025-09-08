import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.rmsnorm import TorchRMSNorm
from train.mla import MLA
from train.gqa import GroupedQueryAttention
from train.moe import DeepSeekMoE


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = TorchRMSNorm(args.d_model, args.rmsnorm_eps)
        # Choose attention: NSA only compatible with GQA; otherwise MLA.
        if getattr(args, 'use_nsa', False):
            # NSA requires GQA; ensure head grouping is valid
            assert args.n_heads % args.num_kv_heads == 0, "n_heads must be divisible by num_kv_heads for GQA/NSA"
            self.attn = GroupedQueryAttention(
                d_model=args.d_model,
                n_q_heads=args.n_heads,
                n_kv_heads=args.num_kv_heads,
                seq_len=args.seq_len,
                d_head=args.d_head,
                use_nsa=True,
                window_size=args.window_size,
            )
        else:
            self.attn = MLA(args.d_model, args.n_heads, args.dc_kv, args.dc_q, args.seq_len, args.device, args.d_head, window_size=args.window_size)
        self.ln2 = TorchRMSNorm(args.d_model, args.rmsnorm_eps)
        self.moe = DeepSeekMoE(
            args.d_model,
            args.n_shared_experts,
            args.n_routed_experts,
            args.k_routed_experts,
            args.bias_update_speed,
            args.moe_balance_factor,
            args.d_ff_expert,
            args.device
        )
        # To track expert usage from both MoE calls (if using double MoE like in train.py)
        self.total_expert_usage = None

    def forward(self, x, cached_c_kv=None, cached_kR=None, cached_len=0, return_latent=False):
        # 1. Attention Branch
        if isinstance(self.attn, MLA):
            if return_latent or cached_c_kv is not None or cached_kR is not None or cached_len > 0:
                attn_out, c_kv, kR = self.attn(self.ln1(x), cached_c_kv=cached_c_kv, cached_kR=cached_kR, cached_len=cached_len, return_latent=True)
            else:
                attn_out = self.attn(self.ln1(x))
                c_kv, kR = None, None
        else:
            # GQA/NSA path: ignore MLA-specific latent cache arguments
            if return_latent:
                attn_out = self.attn(self.ln1(x))
                c_kv, kR = None, None
            else:
                attn_out = self.attn(self.ln1(x))
                c_kv, kR = None, None
        h = x + attn_out
        
        # 2. MoE (FFN) Branch
        moe_out = self.moe(self.ln2(h), training=self.training)
        out = h + moe_out
        
        # Track expert usage
        if self.training:
            self.total_expert_usage = self.moe.last_expert_usage.clone()
        
        if return_latent:
            return out, c_kv, kR
        return out