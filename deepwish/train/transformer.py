import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.rmsnorm import TorchRMSNorm
from train.mla import MLA
from train.moe import DeepSeekMoE


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = TorchRMSNorm(args.d_model, args.rmsnorm_eps)
        self.attn = MLA(args.d_model, args.n_heads, args.dc_kv, args.dc_q, args.seq_len, args.device, args.d_head)
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
        if return_latent or cached_c_kv is not None or cached_kR is not None or cached_len > 0:
            attn_out, c_kv, kR = self.attn(self.ln1(x), cached_c_kv=cached_c_kv, cached_kR=cached_kR, cached_len=cached_len, return_latent=True)
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