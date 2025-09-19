import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicFFN(nn.Module):
    """
    Basic Feed-Forward Network used as MoE experts.
    Note: When used in MoE, input sequence lengths vary because each expert 
    only processes tokens routed to it (not the full sequence).
    This is expected MoE behavior, not a bug.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        B, S, D = x.shape  # x: (B, S, D)
        flat = x.view(-1, D)  # flat: (B*S, D)
        h = F.gelu(flat @ self.W1 + self.b1)  # W1: (D, d_ff) -> h: (B*S, d_ff)
        out = h @ self.W2 + self.b2  # W2: (d_ff, D) -> out: (B*S, D)
        return out.view(B, S, D)
