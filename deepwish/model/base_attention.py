import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional Triton dependency for fast attention
    from train.kernels.fa2 import attention as _triton_attention
except ImportError:  # pragma: no cover - executed when Triton is unavailable
    _triton_attention = None

_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128, 256}


def _torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Fallback attention implementation that works on any device."""
    q_float = q.to(torch.float32)
    k_float = k.to(torch.float32)
    v_float = v.to(torch.float32)
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v_float)
    return context.to(v.dtype)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Dispatch attention to Triton when available, otherwise use torch fallback."""
    head_dim = q.shape[-1]
    same_device = q.device == k.device == v.device
    if not same_device:
        raise ValueError("Q, K, V must live on the same device for attention computation")
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError("Q, K, V must share the same head dimension")

    use_triton = (
        _triton_attention is not None
        and q.device.type == "cuda"
        and k.device.type == "cuda"
        and v.device.type == "cuda"
        and head_dim in _SUPPORTED_HEAD_DIMS
    )

    if use_triton:
        return _triton_attention(q, k, v, scale)

    return _torch_attention(q, k, v, scale)


class BaseAttention(nn.Module):
    """Core attention wrapper shared by MLA and GQA blocks."""

    def __init__(self, d_model, n_heads, d_head=None, seq_len=2048, use_nsa=False, window_size=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.seq_len = seq_len
        self.scale = (self.d_head) ** -0.5
        self.use_nsa = use_nsa
        self.window_size = window_size

    def compute_attention(self, q, k, v, is_decoding=False):
        if is_decoding and q.shape[2] == 1:
            return self._single_token_attention(q, k, v)

        if self.window_size is not None and self.window_size > 0:
            return self._sliding_window_attention(q, k, v, self.window_size)

        return attention(q, k, v, self.scale)

    def _single_token_attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def _sliding_window_attention(self, q, k, v, window_size: int):
        B, H, T, _ = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        device = scores.device
        arange = torch.arange(T, device=device)
        causal = arange[None, :] <= arange[:, None]
        if window_size is not None and window_size > 0:
            lower = arange[:, None] - arange[None, :]
            mask = causal & (lower < window_size)
        else:
            mask = causal
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        return torch.matmul(attn, v)

    def forward(self, x, **kwargs):  # pragma: no cover - abstract API
        raise NotImplementedError


__all__ = ["BaseAttention", "attention"]
