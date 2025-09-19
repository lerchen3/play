import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, seq_len, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(seq_len)

    def _build_cache(self, seq_len):
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def _ensure_length(self, total_len, device, dtype):
        current = self.cos.size(0)
        cos = self.cos.to(device=device, dtype=dtype)
        sin = self.sin.to(device=device, dtype=dtype)
        if total_len > current:
            inv_freq = self.inv_freq.to(device=device, dtype=dtype)
            t = torch.arange(current, total_len, device=device, dtype=dtype)
            if t.numel() > 0:
                freqs = torch.outer(t, inv_freq)
                cos = torch.cat([cos, freqs.cos()], dim=0)
                sin = torch.cat([sin, freqs.sin()], dim=0)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x, offset=0):
        S = x.shape[1]
        d = x.shape[-1]
        d2 = d // 2
        device = x.device
        dtype = x.dtype
        self._ensure_length(offset + S, device, dtype)
        cos = self.cos[offset:offset + S].unsqueeze(0).unsqueeze(2)
        sin = self.sin[offset:offset + S].unsqueeze(0).unsqueeze(2)
        x1, x2 = x[..., :d2], x[..., d2:]
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        return torch.cat((xr1, xr2), dim=-1)