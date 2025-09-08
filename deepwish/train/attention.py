import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.kernels.fa2 import attention


class BaseAttention(nn.Module):
    """
    Base attention class that handles the core attention computation.
    Can use FlashAttention for training or PyTorch for inference.
    Subclasses implement the specific Q, K, V projections and transformations.
    """
    def __init__(self, d_model, n_heads, d_head=None, seq_len=2048, use_nsa=False, window_size=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.seq_len = seq_len
        self.scale = (self.d_head) ** -0.5
        self.use_nsa = use_nsa
        self.window_size = window_size  # None or int

    def compute_attention(self, q, k, v, is_decoding=False):
        """
        Core attention computation given Q, K, V tensors.
        
        Args:
            q: Query tensor (B, H, T, D_head)
            k: Key tensor (B, H, T, D_head) 
            v: Value tensor (B, H, T, D_head)
            is_decoding: True for single-token generation, False for training/prefill
            
        Returns:
            out: Attention output (B, H, T, D_head)
        """
        if is_decoding and q.shape[2] == 1:
            # Single token decoding - use PyTorch for efficiency (only need last row)
            out = self._single_token_attention(q, k, v)
        else:
            # Sliding window (training/prefill) uses a masked PyTorch path for correctness
            if self.window_size is not None and self.window_size > 0:
                out = self._sliding_window_attention(q, k, v, self.window_size)
            else:
                # Training or prefilling - use FlashAttention for full matrix
                out = attention(q, k, v, self.scale)
            
        return out

    def _single_token_attention(self, q, k, v):
        """Efficient single-token attention for decoding (only compute last row)"""
        # q: (B, H, 1, D_head), k,v: (B, H, T, D_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 1, T)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 1, T)
        out = torch.matmul(attn_weights, v)  # (B, H, 1, D_head)
        return out

    def _sliding_window_attention(self, q, k, v, window_size: int):
        """Causal sliding-window attention using a PyTorch masked path.
        q, k, v: (B, H, T, D)
        window_size: attend to last window_size tokens for each position.
        """
        B, H, T, D = q.shape
        # Compute full scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        # Build causal + sliding window mask once on device
        device = scores.device
        arange = torch.arange(T, device=device)
        causal = arange[None, :] <= arange[:, None]  # (T,T)
        if window_size is not None and window_size > 0:
            lower = arange[:, None] - arange[None, :]
            within = lower < window_size
            mask = causal & within
        else:
            mask = causal
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x, **kwargs):
        """
        Forward pass - to be implemented by subclasses.
        Subclasses should:
        1. Project input to Q, K, V
        2. Apply any transformations (RoPE, etc.)
        3. Call self.compute_attention(q, k, v)
        4. Project output and return
        """
        raise NotImplementedError("Subclasses must implement forward()")
