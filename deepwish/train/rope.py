import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, seq_len, base=10000.0):
        super().__init__()
        # Compute rotary frequencies for half the dimension
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # shape: (dim/2,)
        t = torch.arange(seq_len, dtype=inv_freq.dtype)  # shape: (seq_len,)
        freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)  # (seq_len, dim/2); *all* the rotations.
        # Store cos and sin for half-dimension
        self.register_buffer('cos', freqs.cos())  # (seq_len, dim/2)
        self.register_buffer('sin', freqs.sin())  # (seq_len, dim/2)

    def forward(self, x, offset=0):
        # x: (B, S, ..., D) where D is even
        S = x.shape[1]
        d = x.shape[-1]
        d2 = d // 2
        # cos/sin: (seq_len, d2) -> take slice for current sequence length
        cos = self.cos[offset:offset+S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, d2)
        sin = self.sin[offset:offset+S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, d2)
        x1, x2 = x[..., :d2], x[..., d2:]  # both: (B, S, ..., d2)
        # Apply rotary: (B, S, ..., d2) * (1, S, 1, d2)
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        return torch.cat((xr1, xr2), dim=-1)  # (B, S, ..., D)