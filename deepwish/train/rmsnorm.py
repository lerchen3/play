import torch
import torch.nn as nn


class TorchRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x / torch.sqrt(variance + self.eps)
        return x_norm * self.weight