import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.transformer import TransformerBlock


class MTP(nn.Module):
    def __init__(self, depth, args):
        super().__init__()
        self.depth = depth
        self.block = TransformerBlock(args)
        # Latent cache for inference-time speculative decoding (per block)
        self._cached_c_kv = None  # (B, T, dc_kv)
        self._cached_kR = None    # (B, T, n_heads, d_head)
        self._cached_len = 0

    def forward(self, h):
        B, S, D = h.shape
        h_i = h
        outs = []
        for _ in range(self.depth):
            h_i = self.block(h_i)
            outs.append(h_i)  # collect embeddings
        # stacked embeddings: (B, S, depth, D)
        return torch.stack(outs, 2)

    # -------- Inference-time cache API for speculative decoding --------
    @torch.no_grad()
    def reset_latent_cache(self):
        self._cached_c_kv = None
        self._cached_kR = None
        self._cached_len = 0

    @torch.no_grad()
    def prefill_with_cache(self, hidden_full: torch.Tensor):
        """
        Build latent cache for the MTP block from the full hidden sequence of the base model.
        hidden_full: (B, S, D)
        Returns the block output over the full sequence (not usually needed by caller).
        """
        self.eval()
        h, c_kv, kR = self.block(hidden_full, return_latent=True)
        self._cached_c_kv = c_kv
        self._cached_kR = kR
        self._cached_len = hidden_full.size(1)
        return h

    @torch.no_grad()
    def step_with_cache_hidden(
        self,
        hidden_step: torch.Tensor,
        update_cache: bool = True,
        return_hidden: bool = True,
    ):
        """
        Append hidden_step (B, s_step, D) through the MTP block using latent caches.
        Returns the block output aligned to hidden_step (B, s_step, D).
        """
        assert self._cached_c_kv is not None and self._cached_kR is not None, "Call prefill_with_cache() before step_with_cache_hidden()"
        h, c_kv_full, kR_full = self.block(
            hidden_step,
            cached_c_kv=self._cached_c_kv,
            cached_kR=self._cached_kR,
            cached_len=self._cached_len,
            return_latent=True,
        )
        if update_cache:
            self._cached_c_kv = c_kv_full
            self._cached_kR = kR_full
            self._cached_len = self._cached_len + hidden_step.size(1)
        if return_hidden:
            return h
        return None

    @torch.no_grad()
    def snapshot_cache(self):
        if self._cached_c_kv is None:
            return None
        return (
            self._cached_c_kv.clone(),
            self._cached_kR.clone(),
            int(self._cached_len),
        )

    @torch.no_grad()
    def restore_cache(self, snapshot):
        if snapshot is None:
            self.reset_latent_cache()
            return
        c_kv, kR, cached_len = snapshot
        self._cached_c_kv = c_kv
        self._cached_kR = kR
        self._cached_len = cached_len