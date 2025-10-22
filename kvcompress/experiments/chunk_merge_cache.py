from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


def _repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    """
    Replicates key/value tensors for grouped query attention.

    Args:
        hidden_states: Tensor of shape (batch, kv_heads, seq_len, dim)
        repeat: Number of query heads per key/value head.

    Returns:
        Tensor of shape (batch, kv_heads * repeat, seq_len, dim)
    """
    batch, kv_heads, seq_len = hidden_states.shape[:3]
    if repeat == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, kv_heads, repeat, seq_len, *hidden_states.shape[3:])
    return hidden_states.reshape(batch, kv_heads * repeat, seq_len, *hidden_states.shape[4:])


class ChunkMergeCache(Cache):
    """
    Cache that merges temporally contiguous keys during decoding when the merge error stays below a threshold.
    Current chunks remain as raw key/value entries until closed.
    """

    is_sliding = False

    def __init__(
        self,
        *,
        epsilon: float,
        num_layers: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        max_queries: int = 512,
        ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.max_queries = max_queries
        self.eval_query_count = max(1, min(32, max_queries // 4 if max_queries >= 4 else 1))
        self.fit_query_count = max(1, max_queries - self.eval_query_count)
        self.ridge = ridge

        self._initialized: List[bool] = [False] * num_layers
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Per-layer storage
        self._completed: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._current: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._log_counts_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._query_history: List[List[torch.Tensor]] = []
        self._chunk_sizes: List[List[List[int]]] = []
        self._chunk_query_lengths: List[List[List[int]]] = []
        self._error_sums: List[List[float]] = []
        self._error_counts: List[List[int]] = []
        self._debug_errors: List[List[List[float]]] = []

        # Metrics
        self.decode_tokens: int = 0
        self.decode_chunks: int = 0

        for _ in range(num_layers):
            layer_completed: List[Dict[str, List[torch.Tensor]]] = []
            layer_current: List[Dict[str, List[torch.Tensor]]] = []
            layer_queries: List[torch.Tensor] = []
            layer_chunk_sizes: List[List[int]] = []
            layer_query_lengths: List[List[int]] = []
            layer_error_sums: List[float] = []
            layer_error_counts: List[int] = []
            for _head in range(num_kv_heads):
                layer_completed.append({"keys": [], "values": [], "log_counts": []})
                layer_current.append({"keys": [], "values": []})
                layer_queries.append(torch.empty(0, head_dim, dtype=torch.float32))
                layer_chunk_sizes.append([])
                layer_query_lengths.append([])
                layer_error_sums.append(0.0)
                layer_error_counts.append(0)
            self._completed.append(layer_completed)
            self._current.append(layer_current)
            self._query_history.append(layer_queries)
            self._chunk_sizes.append(layer_chunk_sizes)
            self._chunk_query_lengths.append(layer_query_lengths)
            self._error_sums.append(layer_error_sums)
            self._error_counts.append(layer_error_counts)
        self._debug_errors = [[[] for _ in range(num_kv_heads)] for _ in range(num_layers)]

        self.supports_chunk_merge = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_setup(self, key_states: torch.Tensor) -> None:
        if self._device is None:
            self._device = key_states.device
            self._dtype = key_states.dtype

    def _append_queries(self, layer_idx: int, query_states: Optional[torch.Tensor]) -> None:
        if query_states is None:
            return
        # query_states: (batch, num_heads, seq_len, head_dim)
        queries = query_states.detach().to("cpu", torch.float32)
        # query_states shape: (batch, num_heads, seq_len, dim)
        queries = queries.permute(0, 1, 2, 3)  # explicit for clarity
        batch, num_heads, seq_len, _ = queries.shape
        assert batch == 1, "ChunkMergeCache currently supports batch size 1."
        queries = queries[0]  # (num_heads, seq_len, dim)
        for kv_head in range(self.num_kv_heads):
            head_slice = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
            head_queries = queries[head_slice].reshape(-1, self.head_dim)
            if head_queries.numel() == 0:
                continue
            hist = self._query_history[layer_idx][kv_head]
            combined = torch.cat([hist, head_queries], dim=0) if hist.numel() else head_queries
            if combined.shape[0] > self.max_queries:
                combined = combined[-self.max_queries :]
            self._query_history[layer_idx][kv_head] = combined

    def _ingest_prefill(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # key_states: (1, kv_heads, seq_len, head_dim)
        seq_len = key_states.shape[2]
        for pos in range(seq_len):
            keys_slice = key_states[0, :, pos, :]
            values_slice = value_states[0, :, pos, :]
            for kv_head in range(self.num_kv_heads):
                key = keys_slice[kv_head].detach().clone()
                value = values_slice[kv_head].detach().clone()
                self._completed[layer_idx][kv_head]["keys"].append(key)
                self._completed[layer_idx][kv_head]["values"].append(value)
                zero_log = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                self._completed[layer_idx][kv_head]["log_counts"].append(zero_log)
        self._initialized[layer_idx] = True

    def _stack_or_empty(
        self, tensors: List[torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if not tensors:
            return torch.empty(0, self.head_dim, device=device, dtype=dtype)
        stacked = torch.stack(tensors, dim=0)
        return stacked.to(device=device, dtype=dtype, non_blocking=True)

    def _build_dense_cache(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._device is not None and self._dtype is not None
        device = self._device
        dtype = self._dtype

        head_entries: List[torch.Tensor] = []
        head_values: List[torch.Tensor] = []
        head_log_counts: List[torch.Tensor] = []
        max_len = 0

        for kv_head in range(self.num_kv_heads):
            completed = self._completed[layer_idx][kv_head]
            current = self._current[layer_idx][kv_head]

            comp_keys = self._stack_or_empty(completed["keys"], device, dtype)
            comp_vals = self._stack_or_empty(completed["values"], device, dtype)
            if completed["log_counts"]:
                comp_log = torch.stack(completed["log_counts"], dim=0).to(device=device, dtype=torch.float32)
            else:
                comp_log = torch.empty(0, device=device, dtype=torch.float32)

            curr_keys = self._stack_or_empty(current["keys"], device, dtype)
            curr_vals = self._stack_or_empty(current["values"], device, dtype)
            curr_log = torch.zeros(curr_keys.shape[0], device=device, dtype=torch.float32)

            all_keys = torch.cat([comp_keys, curr_keys], dim=0)
            all_vals = torch.cat([comp_vals, curr_vals], dim=0)
            all_logs = torch.cat([comp_log, curr_log], dim=0)

            head_entries.append(all_keys)
            head_values.append(all_vals)
            head_log_counts.append(all_logs)
            max_len = max(max_len, all_keys.shape[0])

        if max_len == 0:
            max_len = 1

        key_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=device, dtype=dtype)
        value_tensor = torch.zeros_like(key_tensor)
        log_count_tensor = torch.full(
            (1, self.num_kv_heads, max_len), float("-inf"), device=device, dtype=torch.float32
        )

        for kv_head in range(self.num_kv_heads):
            keys = head_entries[kv_head]
            vals = head_values[kv_head]
            logs = head_log_counts[kv_head]
            if keys.shape[0] == 0:
                continue
            key_tensor[0, kv_head, : keys.shape[0]] = keys
            value_tensor[0, kv_head, : vals.shape[0]] = vals
            log_count_tensor[0, kv_head, : logs.shape[0]] = logs

        self._key_cache[layer_idx] = key_tensor
        self._value_cache[layer_idx] = value_tensor
        self._log_counts_cache[layer_idx] = log_count_tensor
        return key_tensor, value_tensor, log_count_tensor

    def _chunk_summary(
        self,
        layer_idx: int,
        kv_head: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # keys: (chunk_len, head_dim), values: (chunk_len, head_dim)
        if keys.shape[0] == 1:
            mean_v = values[0]
            k_center = keys[0].to(torch.float32)
            return k_center, mean_v

        queries = self._query_history[layer_idx][kv_head]
        if queries.numel() == 0:
            mean_v = values.mean(dim=0)
            k_center = keys.mean(dim=0).to(torch.float32)
            return k_center, mean_v

        scale = 1.0 / math.sqrt(self.head_dim)
        fit_queries = queries[-self.fit_query_count :]
        q = fit_queries.to(torch.float32)
        k = keys.to(torch.float32)
        logits = torch.matmul(q, k.transpose(0, 1)) * scale  # (num_queries, chunk_len)
        targets = torch.logsumexp(logits, dim=1) - math.log(keys.shape[0])  # (num_queries,)
        q_t = q.transpose(0, 1)  # (head_dim, num_queries)
        cov = torch.matmul(q_t, q)
        cov = cov + self.ridge * torch.eye(self.head_dim, dtype=torch.float32)
        rhs = torch.matmul(q_t, targets)
        k_center = torch.linalg.solve(cov, rhs)
        mean_v = values.mean(dim=0)
        return k_center, mean_v

    def _chunk_error(
        self,
        layer_idx: int,
        kv_head: int,
        keys: torch.Tensor,
        k_center: torch.Tensor,
    ) -> float:
        queries = self._query_history[layer_idx][kv_head]
        k = keys.to(torch.float32)
        l2_mean = torch.norm(k - k_center.unsqueeze(0), dim=1).mean().item()
        if queries.numel() == 0:
            return l2_mean
        scale = 1.0 / math.sqrt(self.head_dim)
        if queries.shape[0] > self.fit_query_count:
            eval_queries = queries[:-self.fit_query_count]
        else:
            eval_queries = queries
        q = eval_queries.to(torch.float32)
        logits = torch.matmul(q, k.transpose(0, 1)) * scale
        original = torch.exp(logits).sum(dim=1)
        merged = torch.exp(torch.matmul(q, k_center) * scale + math.log(keys.shape[0]))
        err = torch.mean((original - merged) ** 2).item()
        return err + l2_mean

    def _finalize_chunk(
        self,
        layer_idx: int,
        kv_head: int,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
    ) -> None:
        if not keys:
            return
        keys_tensor = torch.stack(keys, dim=0)
        values_tensor = torch.stack(values, dim=0)
        k_center, mean_v = self._chunk_summary(layer_idx, kv_head, keys_tensor, values_tensor)
        error = self._chunk_error(layer_idx, kv_head, keys_tensor, k_center)

        key_tensor = k_center.to(device=self._device, dtype=self._dtype)
        value_tensor = mean_v.to(device=self._device, dtype=self._dtype)
        log_count = torch.tensor(math.log(len(keys)), dtype=torch.float32, device=self._device)

        self._completed[layer_idx][kv_head]["keys"].append(key_tensor)
        self._completed[layer_idx][kv_head]["values"].append(value_tensor)
        self._completed[layer_idx][kv_head]["log_counts"].append(log_count)
        self._chunk_sizes[layer_idx][kv_head].append(len(keys))
        self._chunk_query_lengths[layer_idx][kv_head].append(
            self._query_history[layer_idx][kv_head].shape[0]
        )
        self.decode_tokens += len(keys)
        self.decode_chunks += 1

    def _process_decode(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        # key_states/value_states: (kv_heads, head_dim)
        for kv_head in range(self.num_kv_heads):
            key_vec = key_states[kv_head, 0].detach().clone()
            value_vec = value_states[kv_head, 0].detach().clone()
            current = self._current[layer_idx][kv_head]
            current["keys"].append(key_vec)
            current["values"].append(value_vec)

            # Need at least 2 tokens to consider merging
            if len(current["keys"]) <= 1:
                continue

            queries = self._query_history[layer_idx][kv_head]
            if queries.numel() == 0:
                continue

            keys_tensor = torch.stack(current["keys"], dim=0)
            values_tensor = torch.stack(current["values"], dim=0)
            k_center, mean_v = self._chunk_summary(layer_idx, kv_head, keys_tensor, values_tensor)
            error = self._chunk_error(layer_idx, kv_head, keys_tensor, k_center)
            self._error_sums[layer_idx][kv_head] += error
            self._error_counts[layer_idx][kv_head] += 1
            if len(self._debug_errors[layer_idx][kv_head]) < 16:
                self._debug_errors[layer_idx][kv_head].append(error)
            if error <= self.epsilon:
                continue

            # Finalize previous tokens (exclude latest)
            prev_keys = current["keys"][:-1]
            prev_vals = current["values"][:-1]
            self._finalize_chunk(layer_idx, kv_head, prev_keys, prev_vals)

            # Reset current chunk to last token
            latest_key = current["keys"][-1]
            latest_val = current["values"][-1]
            current["keys"] = [latest_key]
            current["values"] = [latest_val]

    # ------------------------------------------------------------------
    # Cache API
    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_setup(key_states)
        assert key_states.shape[0] == 1, "ChunkMergeCache assumes batch size 1 during generation."

        query_states = None
        if cache_kwargs is not None:
            query_states = cache_kwargs.get("query_states")

        self._append_queries(layer_idx, query_states)

        if key_states.shape[2] > 1 or not self._initialized[layer_idx]:
            self._ingest_prefill(layer_idx, key_states, value_states)
            keys, values, _ = self._build_dense_cache(layer_idx)
            return keys, values

        # Build cache snapshot using current completed chunks and open chunk (before adding new token)
        keys_snapshot, values_snapshot, _ = self._build_dense_cache(layer_idx)

        # Process new token (seq_len == 1)
        key_vecs = key_states[0]
        value_vecs = value_states[0]
        self._process_decode(layer_idx, key_vecs, value_vecs)

        return keys_snapshot, values_snapshot

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        if layer_idx is None:
            layer_idx = 0
        cache = self._key_cache[layer_idx]
        if cache is None:
            return 0
        return cache.shape[2]

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------
    def chunk_merge_forward(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Custom attention computation leveraging merged chunks.

        Args:
            layer_idx: Layer index.
            query_states: Tensor of shape (batch, num_heads, q_len, head_dim).

        Returns:
            Attention output tensor of shape (batch, num_heads, q_len, head_dim).
        """
        assert self._log_counts_cache[layer_idx] is not None
        assert self._key_cache[layer_idx] is not None
        assert self._value_cache[layer_idx] is not None

        keys = self._key_cache[layer_idx]
        values = self._value_cache[layer_idx]
        log_counts = self._log_counts_cache[layer_idx]

        k_exp = _repeat_kv(keys, self.group_size)
        v_exp = _repeat_kv(values, self.group_size)
        log_exp = _repeat_kv(log_counts.unsqueeze(-1), self.group_size).squeeze(-1)

        logits = torch.matmul(
            query_states.to(dtype=torch.float32),
            k_exp.transpose(-1, -2).to(dtype=torch.float32),
        )
        logits = logits + log_exp.unsqueeze(-2)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(v_exp.dtype)
        output = torch.matmul(attn, v_exp)
        return output

    # ------------------------------------------------------------------
    # Metrics / utilities
    # ------------------------------------------------------------------
    def flush(self) -> None:
        for layer_idx in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                current = self._current[layer_idx][kv_head]
                if current["keys"]:
                    self._finalize_chunk(layer_idx, kv_head, current["keys"], current["values"])
                    current["keys"] = []
                    current["values"] = []
            # rebuild cache tensors after flushing
            self._build_dense_cache(layer_idx)

    def summary(self) -> Dict[str, object]:
        chunk_sizes = {
            f"layer_{layer:02d}": {
                f"head_{head:02d}": sizes
                for head, sizes in enumerate(self._chunk_sizes[layer])
            }
            for layer in range(self.num_layers)
        }
        query_lengths = {
            f"layer_{layer:02d}": {
                f"head_{head:02d}": lengths
                for head, lengths in enumerate(self._chunk_query_lengths[layer])
            }
            for layer in range(self.num_layers)
        }
        avg_errors = {
            f"layer_{layer:02d}": {
                f"head_{head:02d}": (
                    self._error_sums[layer][head] / max(1, self._error_counts[layer][head])
                )
                for head in range(self.num_kv_heads)
            }
            for layer in range(self.num_layers)
        }
        debug_errors = {
            f"layer_{layer:02d}": {
                f"head_{head:02d}": self._debug_errors[layer][head]
                for head in range(self.num_kv_heads)
            }
            for layer in range(self.num_layers)
        }
        return {
            "epsilon": self.epsilon,
            "decode_tokens": self.decode_tokens,
            "decode_chunks": self.decode_chunks,
            "chunk_sizes": chunk_sizes,
            "chunk_query_lengths": query_lengths,
            "avg_errors": avg_errors,
            "debug_errors": debug_errors,
        }
