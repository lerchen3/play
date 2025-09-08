import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.rmsnorm import TorchRMSNorm
from train.gqa import GroupedQueryAttention
from train.kernels.cce import TritonCCE


def triton_cce_loss(hidden, weight, targets, ignore_index=None, block_size=64):
    """Cross-entropy loss using Triton kernel"""
    # hidden: (B, S, D) -> E: (N, D), targets: (N,)
    E = hidden.view(-1, hidden.size(-1))
    t = targets.reshape(-1)
    if ignore_index is not None:
        mask = t != ignore_index
        E = E[mask]
        t = t[mask]
        if E.numel() == 0:
            raise ValueError("Everything is a PAD token; empty string was example...")
    # weight is [V, D], directly use as class matrix C
    C = weight
    return TritonCCE.apply(E, C, t, block_size)


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in Qwen3"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        gate = self.W_gate(x)
        up = self.W_up(x)
        return self.W_down(F.silu(gate) * up)


class Qwen3TransformerBlock(nn.Module):
    """Single transformer block for Qwen3"""
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, seq_len, eps=1e-6):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads, seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
        self.attn_norm = TorchRMSNorm(d_model, eps=eps)
        self.ffn_norm = TorchRMSNorm(d_model, eps=eps)
        
    def forward(self, x, cached_k=None, cached_v=None, cached_len=0, return_cache=False):
        # Pre-norm architecture as used in Qwen3
        
        # Attention with residual connection
        if return_cache:
            attn_out, new_k, new_v = self.attn(self.attn_norm(x), cached_k, cached_v, cached_len, return_cache=True)
            x = x + attn_out
        else:
            attn_out = self.attn(self.attn_norm(x), cached_k, cached_v, cached_len)
            x = x + attn_out
        
        # FFN with residual connection
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        
        if return_cache:
            return x, new_k, new_v
        return x


class Qwen3Model(nn.Module):
    """
    Qwen3-0.6B model implementation
    
    Architecture specs:
    - 28 layers
    - 1024 hidden dimension
    - 3072 FFN dimension  
    - 16 query heads, 8 key-value heads (GQA)
    - RMSNorm with pre-normalization
    - SwiGLU activation
    - RoPE positional embeddings
    - 32K context length
    """
    def __init__(self, args):
        super().__init__()
        # Use Qwen3 defaults if not provided in args
        self.d_model = getattr(args, 'd_model', 1024)
        self.num_layers = getattr(args, 'num_layers', 28)
        self.n_q_heads = getattr(args, 'n_q_heads', 16)  
        self.n_kv_heads = getattr(args, 'n_kv_heads', 8)
        self.d_ff = getattr(args, 'd_ff', 3072)
        self.seq_len = getattr(args, 'seq_len', 32768)
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        
        # Token embeddings
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        with torch.no_grad():
            self.token_embed.weight[self.pad_token_id].zero_()
            
        # Transformer layers
        self.layers = nn.ModuleList([
            Qwen3TransformerBlock(
                self.d_model, 
                self.n_q_heads, 
                self.n_kv_heads, 
                self.d_ff, 
                self.seq_len
            ) 
            for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.norm = TorchRMSNorm(self.d_model, eps=1e-6)
        
        # Output head (saved as [Vocab, D] to avoid transpose in loss)
        self.head = nn.Parameter(torch.randn(self.vocab_size, self.d_model) / math.sqrt(self.d_model))
        
        # KV cache for inference
        self._cached_k = None  # list of tensors per layer
        self._cached_v = None  # list of tensors per layer
        self._cached_len = 0
        
    def reset_kv_cache(self):
        """Reset KV cache for inference"""
        self._cached_k = None
        self._cached_v = None
        self._cached_len = 0
        
    @torch.no_grad()
    def prefill_with_cache(self, input_ids: torch.Tensor):
        """Build KV cache from a full prompt sequence for inference"""
        self.eval()
        B, S = input_ids.shape
        h = self.token_embed(input_ids)
        mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask
        
        cached_k_list, cached_v_list = [], []
        
        for layer in self.layers:
            h, k_cache, v_cache = layer(h, return_cache=True)
            cached_k_list.append(k_cache)
            cached_v_list.append(v_cache)
            
        self._cached_k = cached_k_list
        self._cached_v = cached_v_list
        self._cached_len = S
        return h
        
    @torch.no_grad()
    def step_with_cache(self, input_ids_step: torch.Tensor, return_all_logits: bool = False):
        """Generate next token(s) using KV cache"""
        assert self._cached_k is not None and self._cached_v is not None, "Call prefill_with_cache() first"
        
        B, S = input_ids_step.shape
        h = self.token_embed(input_ids_step)
        mask = (input_ids_step != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask
        
        new_k_list, new_v_list = [], []
        
        for idx, layer in enumerate(self.layers):
            cached_k = self._cached_k[idx] if self._cached_k else None
            cached_v = self._cached_v[idx] if self._cached_v else None
            h, new_k, new_v = layer(h, cached_k, cached_v, self._cached_len, return_cache=True)
            new_k_list.append(torch.cat([cached_k, new_k], dim=1) if cached_k is not None else new_k)
            new_v_list.append(torch.cat([cached_v, new_v], dim=1) if cached_v is not None else new_v)
            
        # Update cache
        self._cached_k = new_k_list
        self._cached_v = new_v_list
        self._cached_len += S
        
        # Final norm and output projection
        h = self.norm(h)
        if return_all_logits:
            logits = torch.matmul(h, self.head.t())  # (B, S, V)
        else:
            last_hidden = h[:, -1, :]  # (B, D)
            logits = torch.matmul(last_hidden, self.head.t())  # (B, V)
            
        return logits

    def forward(self, input_ids, target_main=None, is_training=True):
        """Forward pass for training or inference"""
        B, S = input_ids.shape
        h = self.token_embed(input_ids)
        mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()
        h = h * mask
        
        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h)
            
        # Final normalization
        h = self.norm(h)
        
        # Compute loss if targets provided
        if target_main is not None and is_training:
            loss = triton_cce_loss(h, self.head, target_main, ignore_index=self.pad_token_id)
            return loss
        else:
            # Return logits for inference
            logits = torch.matmul(h, self.head.t())
            return logits
