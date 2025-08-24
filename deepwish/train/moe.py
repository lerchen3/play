import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.ffn import BasicFFN


class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, ns, nr, kr, bias_sp, bal, d_ff, device):
        super().__init__()
        self.shared = nn.ModuleList([BasicFFN(d_model, d_ff) for _ in range(ns)])
        self.routed = nn.ModuleList([BasicFFN(d_model, d_ff) for _ in range(nr)])
        self.centroids = nn.Parameter(torch.randn(nr, d_model) / math.sqrt(d_model))
        # register bias as a non-trainable buffer so it moves with .to(device)
        self.register_buffer('bias', torch.zeros(nr, device=device))
        self.ns, self.nr, self.kr = ns, nr, kr
        self.bias_sp, self.bal = bias_sp, bal
        self.dev = device

    def forward(self, u, training=True):
        B, S, D = u.shape  # u: (B, S, D)
        flat = u.view(-1, D)  # flat: (B*S, D)
        out = u.clone()       # out: (B, S, D)
        for e in self.shared:
            out = out + e(u)
        # Compute gating scores and routing indices
        # scores: (B*S, nr)
        scores = torch.sigmoid(flat @ self.centroids.t()) + (self.bias if training else 0)
        
        # Validate scores before routing
        if not torch.isfinite(scores).all():
            print("Warning: NaN/Inf in MoE scores - using uniform routing")
            scores = torch.ones_like(scores) / self.nr  # uniform distribution
        
        values, idx = scores.topk(self.kr, dim=1)  # values: (B*S, kr), idx: (B*S, kr)
        
        # Validate routing indices to prevent CUDA illegal memory access
        valid_mask = (idx >= 0) & (idx < self.nr)
        if not valid_mask.all():
            print(f"Warning: Invalid expert indices detected. Clamping to [0, {self.nr-1}]")
            idx = torch.clamp(idx, 0, self.nr - 1)
        
        gating = torch.zeros_like(scores)  # (B*S, nr)
        gating.scatter_(1, idx, values)    # place top-k scores
        
        # Avoid division by zero
        gating_sum = gating.sum(-1, keepdim=True)
        gating_sum = torch.clamp(gating_sum, min=1e-8)  # prevent division by zero
        gating = gating / gating_sum  # normalize along experts
        
        # record expert usage stats with validation
        assignments = idx.flatten()
        try:
            self.last_expert_usage = torch.bincount(assignments, minlength=self.nr).detach().cpu()
        except RuntimeError as e:
            print(f"Warning: bincount failed: {e}. Using zero usage stats.")
            self.last_expert_usage = torch.zeros(self.nr, dtype=torch.long)
        rout = torch.zeros_like(flat)  # (B*S, D)
        # MoE Routing: Each expert processes only a subset of tokens based on routing decisions
        # This is why BasicFFN sees variable sequence lengths - it's the correct MoE behavior
        # Dispatch tokens through each expert
        for j in range(self.nr):
            mask_j = (idx == j).any(dim=1)  # mask for tokens assigned to expert j: (B*S,)
            if mask_j.any():
                toks = flat[mask_j]  # selected token representations: (n_j, D)
                fo = self.routed[j](toks.view(1, -1, D)).view(-1, D)  # FFN outputs: (n_j, D)
                gate_j = gating[mask_j, j]  # gating weights for expert j: (n_j,)
                rout[mask_j] += fo * gate_j.unsqueeze(1)  # accumulate gating contributions
        # Reshape and combine routed output
        return out + rout.view(B, S, D)

    def update_biases(self, speed):
        """
        Adjust expert biases for load balancing: experts in bottom half get increased bias, top half decreased.
        """
        if not hasattr(self, 'last_expert_usage') or self.last_expert_usage is None:
            return
        # last_expert_usage is on CPU
        counts = self.last_expert_usage.to(self.bias.device)
        nr = self.nr
        # sort experts by usage ascending
        sorted_idx = torch.argsort(counts)
        half = nr // 2
        bottom = sorted_idx[:half]
        top = sorted_idx[half:]
        with torch.no_grad():
            self.bias[bottom] += speed
            self.bias[top] -= speed