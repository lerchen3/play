import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.transformer import TransformerBlock
from train.mtp import MTP
from train.kernels.cce import TritonCCE


def triton_cce_loss(hidden, weight, targets, ignore_index=None, block_size=64):
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


class DeepSeekV3Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embed = nn.Embedding(args.vocab_size, args.d_model, padding_idx=args.pad_token_id)
        with torch.no_grad():
            self.token_embed.weight[args.pad_token_id].zero_()
        # Extend args with NSA/GQA defaults if missing
        if not hasattr(args, 'use_nsa'):
            args.use_nsa = False
        if not hasattr(args, 'num_kv_heads'):
            # default to MHA equivalent (no grouping) unless specified
            args.num_kv_heads = args.n_heads
        if not hasattr(args, 'window_size'):
            args.window_size = None
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_layers)])
        # classifier matrix saved as [Vocab, D] to avoid transpose in loss
        self.head = nn.Parameter(torch.randn(args.vocab_size, args.d_model) / math.sqrt(args.d_model))
        self.mtp = MTP(args.mtp_depth, args) if args.mtp_depth > 0 else None

        # Latent cache storage for inference (list per layer)
        self._cached_c_kv = None  # list of tensors per layer: (B, T, dc_kv)
        self._cached_kR = None    # list of tensors per layer: (B, T, n_heads, d_head)
        self._cached_len = 0

    def reset_latent_cache(self):
        self._cached_c_kv = None
        self._cached_kR = None
        self._cached_len = 0

    @torch.no_grad()
    def prefill_with_cache(self, input_ids: torch.Tensor):
        """
        Build latent cache c_kv and kR for all layers from a full prompt sequence.
        input_ids: (B, S)
        """
        self.eval()
        B, S = input_ids.shape
        h = self.token_embed(input_ids)
        mask = (input_ids != self.token_embed.padding_idx).unsqueeze(-1).float()
        h = h * mask
        cached_c_list, cached_kR_list = [], []
        for layer in self.layers:
            h, c_kv, kR = layer(h, return_latent=True)
            cached_c_list.append(c_kv)
            cached_kR_list.append(kR)
        self._cached_c_kv = cached_c_list
        self._cached_kR = cached_kR_list
        self._cached_len = S
        return h  # final hidden of prompt

    @torch.no_grad()
    def step_with_cache(
        self,
        input_ids_step: torch.Tensor,
        return_all_logits: bool = False,
        return_hidden: bool = False,
        update_cache: bool = True,
    ):
        """
        Append one (or a few) tokens to the sequence using latent cache.
        input_ids_step: (B, s_step)
        - return_all_logits: if True, return logits for all step positions (B, s_step, V)
                             otherwise return only last-position logits (B, V)
        - return_hidden: if True, also return the hidden states for the step (B, s_step, D)
        - update_cache: if False, do not commit updated latent caches (used for speculative verification)
        Returns a tuple depending on flags.
        """
        assert self._cached_c_kv is not None and self._cached_kR is not None, "Call prefill_with_cache() before step_with_cache()"
        B, s_step = input_ids_step.shape
        h = self.token_embed(input_ids_step)
        mask = (input_ids_step != self.token_embed.padding_idx).unsqueeze(-1).float()
        h = h * mask

        new_c_list, new_kR_list = [], []
        cached_len = self._cached_len
        for idx, layer in enumerate(self.layers):
            c_prev = self._cached_c_kv[idx]
            kR_prev = self._cached_kR[idx]
            h, c_kv_full, kR_full = layer(
                h,
                cached_c_kv=c_prev,
                cached_kR=kR_prev,
                cached_len=cached_len,
                return_latent=True,
            )
            # Collect candidate caches for potential commit
            new_c_list.append(c_kv_full)
            new_kR_list.append(kR_full)

        if update_cache:
            self._cached_c_kv = new_c_list
            self._cached_kR = new_kR_list
            self._cached_len = cached_len + s_step

        emb_main = h  # (B, s_step, D)
        if return_all_logits:
            logits = torch.matmul(emb_main, self.head.t())  # (B, s_step, V)
        else:
            last_hidden = emb_main[:, -1, :]
            logits = torch.matmul(last_hidden, self.head.t())  # (B, V)

        if return_hidden:
            return logits, emb_main
        return logits

    def forward(self, input_ids, target_main=None, tgt_matrix=None, is_training=True):
        h = self.token_embed(input_ids)
        mask = (input_ids != self.token_embed.padding_idx).unsqueeze(-1).float()
        h = h * mask
        
        # pass through transformer blocks
        for i, layer in enumerate(self.layers):
            h = layer(h)
                
        # main final embeddings: (B, S, D)
        emb_main = h
        
        # compute main loss if targets provided
        loss_main = None
        if target_main is not None:
            loss_main = triton_cce_loss(emb_main, self.head, target_main, ignore_index=self.token_embed.padding_idx)
                
        # compute MTP embeddings and loss
        emb_mtp = None
        loss_mtp = None
        if self.mtp and is_training:
            emb_mtp = self.mtp(h)
                
            if tgt_matrix is not None:
                # Create zero loss that's connected to computation graph
                loss_mtp = 0.0 * self.head.sum()
                for j in range(self.mtp.depth):
                    hidden_j = emb_mtp[:, :, j, :]
                    target_j = tgt_matrix[:, :, j].reshape(-1)
                    mtp_loss_j = triton_cce_loss(hidden_j, self.head, target_j, ignore_index=self.token_embed.padding_idx)
                    loss_mtp += mtp_loss_j
                loss_mtp /= self.mtp.depth
                    
        # ensure losses are tensors and stack into one 1D tensor for proper grad tracking
        lm = loss_main if loss_main is not None else 0.0 * self.head.sum()
        lmt = loss_mtp if loss_mtp is not None else 0.0 * self.head.sum()
            
        return torch.stack([lm, lmt], dim=0)