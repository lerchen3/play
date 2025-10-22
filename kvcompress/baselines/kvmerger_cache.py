from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


def _repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    """Replicates key/value tensors for grouped query attention."""
    batch, kv_heads, seq_len = hidden_states.shape[:3]
    if repeat == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, kv_heads, repeat, seq_len, *hidden_states.shape[3:])
    return hidden_states.reshape(batch, kv_heads * repeat, seq_len, *hidden_states.shape[4:])


class KVMergerCache(Cache):
    """
    KVMerger: Gaussian-weighted key merging based on attention scores.
    Merges keys every N tokens using attention-score-based pivots.
    """

    is_sliding = False

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        merge_interval: int = 4,
        merge_window: int = 8,
        cosine_threshold: Optional[float] = None,
        l2_threshold: Optional[float] = None,
        use_whitening: bool = False,
        stats_path: Optional[str | Path] = None,
        sigma: float = 1.0,  # Gaussian kernel bandwidth
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.merge_interval = merge_interval
        self.merge_window = merge_window
        self.cosine_threshold = cosine_threshold
        self.l2_threshold = l2_threshold
        self.use_whitening = use_whitening
        self.sigma = sigma

        # Load whitening matrices if using whitened version
        if use_whitening:
            assert stats_path is not None, "stats_path required for whitening"
            stats = torch.load(Path(stats_path))
            sigma_q = stats["sigma_q"].to(torch.float64)
            
            self.whitening_matrices = torch.zeros((num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32)
            for layer in range(num_layers):
                for kv_head in range(num_kv_heads):
                    q_indices = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
                    sigma_group = 0.5 * (sigma_q[layer, q_indices] + sigma_q[layer, q_indices].transpose(-1, -2))
                    sigma_avg = sigma_group.mean(dim=0)
                    
                    eigvals, eigvecs = torch.linalg.eigh(sigma_avg)
                    eigvals = torch.clamp(eigvals, min=1e-8)
                    sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
                    self.whitening_matrices[layer, kv_head] = sigma_inv_sqrt.to(torch.float32)
        else:
            self.whitening_matrices = None

        self._initialized: List[bool] = [False] * num_layers
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Storage: list of keys/values (potentially merged)
        self._keys: List[List[List[torch.Tensor]]] = []
        self._values: List[List[List[torch.Tensor]]] = []
        self._counts: List[List[List[int]]] = []  # Number of original keys merged into each
        self._attention_scores: List[List[List[float]]] = []  # Aggregated attention scores for pivot selection
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        # Metrics
        self.decode_tokens = 0
        self.total_merged = 0

        for _ in range(num_layers):
            layer_keys = [[] for _ in range(num_kv_heads)]
            layer_values = [[] for _ in range(num_kv_heads)]
            layer_counts = [[] for _ in range(num_kv_heads)]
            layer_attention_scores = [[] for _ in range(num_kv_heads)]
            self._keys.append(layer_keys)
            self._values.append(layer_values)
            self._counts.append(layer_counts)
            self._attention_scores.append(layer_attention_scores)

    def _whiten(self, vec: torch.Tensor, layer_idx: int, kv_head: int) -> torch.Tensor:
        """Apply whitening if enabled"""
        if not self.use_whitening:
            return vec
        W = self.whitening_matrices[layer_idx, kv_head].to(device=vec.device, dtype=vec.dtype)
        if vec.dim() == 1:
            return torch.matmul(W, vec)
        return torch.matmul(vec, W.T)

    def update_attention_scores(self, layer_idx: int, kv_head: int, attention_matrix: torch.Tensor) -> None:
        """
        Update aggregated attention scores for all cached keys.
        
        Args:
            layer_idx: Layer index
            kv_head: KV head index
            attention_matrix: Attention matrix A[i,j] = attention from query i to key j
                             Shape: [num_queries, num_keys]
        """
        # Sum attention scores across all queries for each key
        aggregated_scores = attention_matrix.sum(dim=0)  # Shape: [num_keys]
        
        # Update scores for existing keys
        num_existing_keys = len(self._attention_scores[layer_idx][kv_head])
        num_new_scores = aggregated_scores.shape[0]
        
        # Extend if we have more scores than existing keys
        while len(self._attention_scores[layer_idx][kv_head]) < num_new_scores:
            self._attention_scores[layer_idx][kv_head].append(0.0)
        
        # Update scores for the relevant keys
        for i in range(min(num_existing_keys, num_new_scores)):
            self._attention_scores[layer_idx][kv_head][i] += aggregated_scores[i].item()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._initialize_metadata(key_states)

        if key_states.shape[2] > 1:
            return self._update_prefill(key_states, value_states, layer_idx)
        return self._update_decode(key_states, value_states, layer_idx, cache_kwargs)

    def _initialize_metadata(self, ref_tensor: torch.Tensor) -> None:
        if self._device is None:
            self._device = ref_tensor.device
            self._dtype = ref_tensor.dtype

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store all keys as singletons during prefill
        batch, num_heads, seq_len, head_dim = key_states.shape
        
        # Count prefill tokens (only once, on first layer)
        if layer_idx == 0:
            self.decode_tokens += seq_len
        
        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                for t in range(seq_len):
                    key = key_states[b, kv_head, t, :]
                    value = value_states[b, kv_head, t, :]
                    self._keys[layer_idx][kv_head].append(key)
                    self._values[layer_idx][kv_head].append(value)
                    self._counts[layer_idx][kv_head].append(1)
                    # Initialize attention score to 0 for prefill tokens
                    self._attention_scores[layer_idx][kv_head].append(0.0)

        self._initialized[layer_idx] = True
        self._invalidate_cache(layer_idx)
        return self._build_cache_tensors(layer_idx)

    def _update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads, seq_len, head_dim = key_states.shape
        assert seq_len == 1
        
        if layer_idx == 0:
            self.decode_tokens += 1

        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                key = key_states[b, kv_head, 0, :]
                value = value_states[b, kv_head, 0, :]
                
                # Add new key
                self._keys[layer_idx][kv_head].append(key)
                self._values[layer_idx][kv_head].append(value)
                self._counts[layer_idx][kv_head].append(1)
                # Initialize attention score to 0 for new token
                self._attention_scores[layer_idx][kv_head].append(0.0)
                
                # Try merging every merge_interval tokens
                if len(self._keys[layer_idx][kv_head]) % self.merge_interval == 0:
                    self._try_merge(layer_idx, kv_head)

        self._invalidate_cache(layer_idx)
        return self._build_cache_tensors(layer_idx)

    def _try_merge(self, layer_idx: int, kv_head: int) -> None:
        """
        KVMerger strategy with attention-based pivot selection:
        1. Find last merge_window singleton entries
        2. Check similarity threshold (L2 or cosine)
        3. If merge: select pivot as most-attended token
        4. Apply Gaussian weighting around pivot
        
        Only merge the last merge_window SINGLETON entries (count=1).
        This prevents re-merging already merged entries.
        """
        keys = self._keys[layer_idx][kv_head]
        values = self._values[layer_idx][kv_head]
        counts = self._counts[layer_idx][kv_head]
        attention_scores = self._attention_scores[layer_idx][kv_head]
        
        # Find singleton entries (count=1) from the end
        singleton_indices = []
        for i in range(len(keys) - 1, -1, -1):
            if counts[i] == 1:
                singleton_indices.append(i)
                if len(singleton_indices) >= self.merge_window:
                    break
        
        if len(singleton_indices) < self.merge_window:
            return
        
        # Work on last merge_window SINGLETON keys only
        window_indices = singleton_indices[:self.merge_window]
        window_keys = torch.stack([keys[i] for i in window_indices], dim=0).to(torch.float32)
        window_values = torch.stack([values[i] for i in window_indices], dim=0)
        window_counts = [counts[i] for i in window_indices]
        window_attention_scores = [attention_scores[i] for i in window_indices]
        
        # Whiten if enabled
        if self.use_whitening:
            window_keys_proc = self._whiten(window_keys, layer_idx, kv_head)
        else:
            window_keys_proc = window_keys
        
        # Check similarity threshold to decide if we should merge
        should_merge = False
        if self.l2_threshold is not None:
            # L2-based threshold
            center = window_keys_proc.mean(dim=0)
            l2_distances = torch.norm(window_keys_proc - center, dim=1)
            mean_l2 = l2_distances.mean().item()
            should_merge = mean_l2 < self.l2_threshold
        elif self.cosine_threshold is not None:
            # Cosine similarity threshold
            window_keys_norm = F.normalize(window_keys_proc, p=2, dim=1)
            cos_sim = torch.matmul(window_keys_norm, window_keys_norm.T)
            mask = ~torch.eye(self.merge_window, dtype=torch.bool, device=cos_sim.device)
            mean_cosine = cos_sim[mask].mean().item()
            should_merge = mean_cosine > self.cosine_threshold
        
        if not should_merge:
            return
        
        # KVMerger Algorithm 2: Attention-based pivot selection
        # Select pivot as token with highest aggregated attention score
        pivot_idx = max(range(len(window_attention_scores)), key=lambda i: window_attention_scores[i])
        pivot_key = window_keys_proc[pivot_idx]  # Use processed (whitened) key for distance computation
        
        # Compute Gaussian weights around pivot
        weights = torch.zeros(self.merge_window, dtype=torch.float32, device=window_keys.device)
        for i in range(self.merge_window):
            if i == pivot_idx:
                # Pivot gets maximum weight
                weights[i] = 1.0
            else:
                # Gaussian weight based on distance to pivot
                key_i = window_keys_proc[i]
                distance_sq = torch.norm(key_i - pivot_key) ** 2
                gaussian_weight = torch.exp(-distance_sq / (2 * self.sigma ** 2))
                weights[i] = gaussian_weight
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Perform merge (ensure proper device and dtype)
        weights = weights.to(device=window_keys.device, dtype=window_keys.dtype)
        merged_key = (window_keys.T @ weights).to(self._dtype)
        merged_value = (window_values.to(window_keys.dtype).T @ weights).to(self._dtype)
        merged_count = sum(window_counts)
        
        # Replace the specific singleton entries with merged entry
        # Remove in reverse order to maintain indices
        for idx in sorted(window_indices, reverse=True):
            self._keys[layer_idx][kv_head].pop(idx)
            self._values[layer_idx][kv_head].pop(idx)
            self._counts[layer_idx][kv_head].pop(idx)
            self._attention_scores[layer_idx][kv_head].pop(idx)
        
        self._keys[layer_idx][kv_head].append(merged_key)
        self._values[layer_idx][kv_head].append(merged_value)
        self._counts[layer_idx][kv_head].append(merged_count)
        # Merged entry gets the sum of attention scores
        self._attention_scores[layer_idx][kv_head].append(sum(window_attention_scores))
        
        if layer_idx == 0 and kv_head == 0:
            self.total_merged += self.merge_window - 1

    def _invalidate_cache(self, layer_idx: int) -> None:
        self._key_cache[layer_idx] = None
        self._value_cache[layer_idx] = None

    def _build_cache_tensors(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._key_cache[layer_idx] is not None:
            return self._key_cache[layer_idx], self._value_cache[layer_idx]

        max_len = max(len(self._keys[layer_idx][h]) for h in range(self.num_kv_heads))
        if max_len == 0:
            return (
                torch.empty(1, self.num_kv_heads, 0, self.head_dim, device=self._device, dtype=self._dtype),
                torch.empty(1, self.num_kv_heads, 0, self.head_dim, device=self._device, dtype=self._dtype),
            )

        key_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=self._device, dtype=self._dtype)
        value_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=self._device, dtype=self._dtype)

        for kv_head in range(self.num_kv_heads):
            if self._keys[layer_idx][kv_head]:
                keys_stacked = torch.stack(self._keys[layer_idx][kv_head], dim=0)
                values_stacked = torch.stack(self._values[layer_idx][kv_head], dim=0)
                key_tensor[0, kv_head, :len(self._keys[layer_idx][kv_head])] = keys_stacked
                value_tensor[0, kv_head, :len(self._values[layer_idx][kv_head])] = values_stacked

        self._key_cache[layer_idx] = key_tensor
        self._value_cache[layer_idx] = value_tensor
        return key_tensor, value_tensor

    def __call__(self, query_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key_tensor, value_tensor = self._build_cache_tensors(layer_idx)
        key_tensor_exp = _repeat_kv(key_tensor, self.group_size)
        value_tensor_exp = _repeat_kv(value_tensor, self.group_size)
        return key_tensor_exp, value_tensor_exp

    def get_usable_length(self, layer_idx: int, batch_idx: int = 0) -> int:
        if not self._keys[layer_idx]:
            return 0
        return max(len(self._keys[layer_idx][h]) for h in range(self.num_kv_heads))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.get_usable_length(layer_idx, 0)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_stats(self) -> Dict:
        total_entries = sum(
            len(self._keys[layer][head])
            for layer in range(self.num_layers)
            for head in range(self.num_kv_heads)
        )
        return {
            "decode_tokens": self.decode_tokens,
            "total_entries": total_entries,
            "total_merged": self.total_merged,
            "compression_ratio": total_entries / (self.num_layers * self.num_kv_heads * max(1, self.decode_tokens)),
        }

