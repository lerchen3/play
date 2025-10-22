from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


def _repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    """Replicates key/value tensors for grouped-query attention."""
    batch, kv_heads, seq_len = hidden_states.shape[:3]
    if repeat == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, kv_heads, repeat, seq_len, *hidden_states.shape[3:])
    return hidden_states.reshape(batch, kv_heads * repeat, seq_len, *hidden_states.shape[4:])


class WhitenedNoQueryCache(Cache):
    """
    Minimal cache that merges keys using whitened L2 distance only.

    This corresponds to the historic "fallback" behaviour: no query sampling,
    distortion is mean L2 in whitened space, values are averaged.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        num_layers: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        stats_path: str | Path,
        projection_rank: Optional[int] = None,
    ) -> None:
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self.epsilon = float(epsilon)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        self.log_kv_lengths = True
        self.supports_chunk_merge = True
        self.rank = projection_rank

        stats = torch.load(Path(stats_path))
        sigma_q = stats["sigma_q"].to(torch.float64)

        self.whitening_matrices = torch.zeros(
            (num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32
        )
        for layer in range(num_layers):
            for kv_head in range(num_kv_heads):
                group = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
                sigma_group = 0.5 * (sigma_q[layer, group] + sigma_q[layer, group].transpose(-1, -2))
                sigma_avg = sigma_group.mean(dim=0)
                eigvals, eigvecs = torch.linalg.eigh(sigma_avg)
                eigvals = torch.clamp(eigvals, min=1e-8)
                sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
                self.whitening_matrices[layer, kv_head] = sigma_inv_sqrt.to(torch.float32)

        self._initialized = [False] * num_layers
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        self._completed: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._current: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._log_counts_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._chunk_sizes: List[List[List[int]]] = []
        self._expect_chunk_merge: List[bool] = [False] * num_layers
        self._basis: Optional[torch.Tensor] = None

        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.decode_chunks = 0

        for _ in range(num_layers):
            completed_layer: List[Dict[str, List[torch.Tensor]]] = []
            current_layer: List[Dict[str, List[torch.Tensor]]] = []
            chunk_sizes: List[List[int]] = []
            for _ in range(num_kv_heads):
                completed_layer.append({"keys": [], "values": [], "log_counts": []})
                current_layer.append({"keys": [], "values": []})
                chunk_sizes.append([])
            self._completed.append(completed_layer)
            self._current.append(current_layer)
            self._chunk_sizes.append(chunk_sizes)

        if self.rank is not None:
            basis_path = Path(f"analysis/projections/projection_rank_{self.rank}.pt")
            if not basis_path.exists():
                raise FileNotFoundError(f"Projection basis not found at {basis_path}")
            proj_data = torch.load(basis_path, map_location="cpu")
            basis = proj_data.get("basis")
            if basis is None:
                raise ValueError(f"projection file {basis_path} missing 'basis'")
            if basis.shape != (num_layers, num_kv_heads, head_dim, self.rank):
                raise ValueError(
                    f"projection basis shape {basis.shape} does not match "
                    f"({num_layers}, {num_kv_heads}, {head_dim}, {self.rank})"
                )
            self._basis = basis.to(torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_sliding(self) -> List[bool]:
        return [False] * self.num_layers

    def _whiten(self, vec: torch.Tensor, layer_idx: int, kv_head: int) -> torch.Tensor:
        W = self.whitening_matrices[layer_idx, kv_head].to(device=vec.device, dtype=vec.dtype)
        if vec.dim() == 1:
            return torch.matmul(W, vec)
        return torch.matmul(vec, W.T)

    def _initialize_metadata(self, ref_tensor: torch.Tensor) -> None:
        if self._device is None:
            self._device = ref_tensor.device
            self._dtype = ref_tensor.dtype

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
        if layer_idx == 0:
            self._initialize_metadata(key_states)

        if key_states.shape[2] > 1:
            return self._update_prefill(key_states, value_states, layer_idx)
        return self._update_decode(key_states, value_states, layer_idx)

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, seq_len, _ = key_states.shape
        if layer_idx == 0:
            self.prefill_tokens += seq_len

        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                for t in range(seq_len):
                    key = key_states[b, kv_head, t, :]
                    value = value_states[b, kv_head, t, :]
                    self._completed[layer_idx][kv_head]["keys"].append(key)
                    self._completed[layer_idx][kv_head]["values"].append(value)
                    self._completed[layer_idx][kv_head]["log_counts"].append(
                        torch.tensor(0.0, dtype=torch.float32, device=self._device)
                    )
                    self._chunk_sizes[layer_idx][kv_head].append(1)

        self._initialized[layer_idx] = True
        self._invalidate(layer_idx)
        return self._build_cache_tensors(layer_idx)[:2]

    def _update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, _, _ = key_states.shape
        if layer_idx == 0:
            self.decode_tokens += 1
        self._expect_chunk_merge[layer_idx] = True

        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                key = key_states[b, kv_head, 0, :]
                value = value_states[b, kv_head, 0, :]
                current = self._current[layer_idx][kv_head]

                candidate_keys = current["keys"] + [key]
                candidate_vals = current["values"] + [value]

                if len(candidate_keys) > 1:
                    error = self._chunk_error(layer_idx, kv_head, candidate_keys)
                    print(
                        f"[CHUNK_TEST] layer={layer_idx} head={kv_head} "
                        f"chunk_len={len(candidate_keys)} error={error:.4f} epsilon={self.epsilon}",
                        flush=True,
                    )
                    if error > self.epsilon:
                        print(
                            f"[CHUNK_FINALIZE] layer={layer_idx} head={kv_head} "
                            f"final_len={len(current['keys'])} error={error:.4f} > epsilon",
                            flush=True,
                        )
                        self._finalize_chunk(layer_idx, kv_head, current["keys"], current["values"])
                        current["keys"] = [key]
                        current["values"] = [value]
                        self.decode_chunks += 1
                    else:
                        current["keys"] = candidate_keys
                        current["values"] = candidate_vals
                else:
                    current["keys"] = candidate_keys
                    current["values"] = candidate_vals

        self._invalidate(layer_idx)
        return self._build_cache_tensors(layer_idx)[:2]

    # ------------------------------------------------------------------
    # Chunk utilities
    # ------------------------------------------------------------------
    def _chunk_error(self, layer_idx: int, kv_head: int, keys: List[torch.Tensor]) -> float:
        keys_tensor = torch.stack(keys, dim=0).to(torch.float32)
        whitened = self._whiten(keys_tensor, layer_idx, kv_head)
        center = whitened.mean(dim=0)
        return torch.norm(whitened - center.unsqueeze(0), dim=1).mean().item()

    def _finalize_chunk(
        self,
        layer_idx: int,
        kv_head: int,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
    ) -> None:
        if not keys:
            return

        keys_tensor = torch.stack(keys, dim=0).to(torch.float32)
        values_tensor = torch.stack(values, dim=0)

        whitened = self._whiten(keys_tensor, layer_idx, kv_head)
        center_whitened = whitened.mean(dim=0)

        if self.rank is not None and self._basis is not None:
            basis = self._basis[layer_idx, kv_head].to(device=center_whitened.device, dtype=center_whitened.dtype)
            coeffs = torch.matmul(basis.T, center_whitened)
            center_whitened = torch.matmul(basis, coeffs)

        W = self.whitening_matrices[layer_idx, kv_head].to(device=center_whitened.device)
        W_inv = torch.linalg.inv(W.to(torch.float64)).to(torch.float32)
        center = torch.matmul(W_inv, center_whitened)
        mean_value = values_tensor.mean(dim=0)

        print(
            f"[CHUNK_STORE] layer={layer_idx} head={kv_head} stored_len={len(keys)}",
            flush=True,
        )

        key_tensor = center.to(device=self._device, dtype=self._dtype)
        value_tensor = mean_value.to(device=self._device, dtype=self._dtype)
        log_count = torch.tensor(math.log(len(keys)), dtype=torch.float32, device=self._device)

        self._completed[layer_idx][kv_head]["keys"].append(key_tensor)
        self._completed[layer_idx][kv_head]["values"].append(value_tensor)
        self._completed[layer_idx][kv_head]["log_counts"].append(log_count)
        self._chunk_sizes[layer_idx][kv_head].append(len(keys))

    def _invalidate(self, layer_idx: int) -> None:
        self._key_cache[layer_idx] = None
        self._value_cache[layer_idx] = None
        self._log_counts_cache[layer_idx] = None

    def _build_cache_tensors(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached_keys = self._key_cache[layer_idx]
        cached_vals = self._value_cache[layer_idx]
        cached_logs = self._log_counts_cache[layer_idx]
        if cached_keys is not None and cached_vals is not None and cached_logs is not None:
            return cached_keys, cached_vals, cached_logs

        device = self._device
        dtype = self._dtype
        head_entries: List[torch.Tensor] = []
        head_values: List[torch.Tensor] = []
        head_logs: List[torch.Tensor] = []
        max_len = 0

        for kv_head in range(self.num_kv_heads):
            comp = self._completed[layer_idx][kv_head]
            curr = self._current[layer_idx][kv_head]

            keys = comp["keys"] + curr["keys"]
            vals = comp["values"] + curr["values"]
            logs = comp["log_counts"] + [
                torch.tensor(0.0, dtype=torch.float32, device=device) for _ in curr["keys"]
            ]

            if keys:
                keys_tensor = torch.stack(keys, dim=0).to(device=device, dtype=dtype)
                vals_tensor = torch.stack(vals, dim=0).to(device=device, dtype=dtype)
                logs_tensor = torch.stack(logs, dim=0).to(device=device, dtype=torch.float32)
            else:
                keys_tensor = torch.empty(0, self.head_dim, device=device, dtype=dtype)
                vals_tensor = torch.empty(0, self.head_dim, device=device, dtype=dtype)
                logs_tensor = torch.empty(0, device=device, dtype=torch.float32)

            head_entries.append(keys_tensor)
            head_values.append(vals_tensor)
            head_logs.append(logs_tensor)
            max_len = max(max_len, keys_tensor.shape[0])

        key_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=device, dtype=dtype)
        value_tensor = torch.zeros_like(key_tensor)
        log_tensor = torch.full(
            (1, self.num_kv_heads, max_len), float("-inf"), device=device, dtype=torch.float32
        )

        for kv_head in range(self.num_kv_heads):
            keys = head_entries[kv_head]
            vals = head_values[kv_head]
            logs = head_logs[kv_head]
            if keys.shape[0] > 0:
                key_tensor[0, kv_head, : keys.shape[0]] = keys
                value_tensor[0, kv_head, : vals.shape[0]] = vals
                log_tensor[0, kv_head, : logs.shape[0]] = logs

        self._key_cache[layer_idx] = key_tensor
        self._value_cache[layer_idx] = value_tensor
        self._log_counts_cache[layer_idx] = log_tensor
        return key_tensor, value_tensor, log_tensor

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def __call__(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_tensor, value_tensor, _ = self._build_cache_tensors(layer_idx)
        if query_states.shape[2] == 1 and self._expect_chunk_merge[layer_idx]:
            raise RuntimeError(
                "Chunk-merge fast path not available (attention fell back to dense mode). "
                "Ensure FlashAttention2 is installed so chunk_merge_forward can be used."
            )
        key_tensor_exp = _repeat_kv(key_tensor, self.group_size)
        value_tensor_exp = _repeat_kv(value_tensor, self.group_size)
        return key_tensor_exp, value_tensor_exp

    def chunk_merge_forward(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        key_tensor, value_tensor, log_count_tensor = self._build_cache_tensors(layer_idx)

        key_tensor_exp = _repeat_kv(key_tensor, self.group_size)
        value_tensor_exp = _repeat_kv(value_tensor, self.group_size)
        log_exp = _repeat_kv(log_count_tensor.unsqueeze(-1), self.group_size).squeeze(-1)

        logits = torch.matmul(
            query_states.to(dtype=torch.float32),
            key_tensor_exp.transpose(-1, -2).to(dtype=torch.float32),
        )
        logits = logits + log_exp.unsqueeze(-2)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(value_tensor_exp.dtype)
        output = torch.matmul(attn, value_tensor_exp)

        if self.log_kv_lengths:
            cache_len = key_tensor_exp.shape[-2]
            print(
                f"[KV_LEN] layer={layer_idx} q_len={query_states.shape[2]} cache_len={cache_len}",
                flush=True,
            )
            print(
                f"[KV_HEADS] layer={layer_idx} raw_len={key_tensor.shape[-2]}",
                flush=True,
            )
        return output

    def finalize_all_chunks(self) -> None:
        for layer_idx in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                current = self._current[layer_idx][kv_head]
                if current["keys"]:
                    self._finalize_chunk(layer_idx, kv_head, current["keys"], current["values"])
                    current["keys"] = []
                    current["values"] = []
            self._invalidate(layer_idx)

    def get_seq_length(self, layer_idx: int = 0, batch_idx: int = 0) -> int:
        total = sum(len(self._completed[layer_idx][h]["keys"]) for h in range(self.num_kv_heads))
        total += sum(len(self._current[layer_idx][h]["keys"]) for h in range(self.num_kv_heads))
        return total // max(1, self.num_kv_heads)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_chunk_stats(self) -> Dict[str, object]:
        stats: Dict[str, object] = {"chunks_per_layer": {}, "total_chunks": 0}
        for layer_idx in range(self.num_layers):
            layer_chunks = sum(len(self._chunk_sizes[layer_idx][h]) for h in range(self.num_kv_heads))
            stats["chunks_per_layer"][layer_idx] = layer_chunks
            stats["total_chunks"] += layer_chunks
        stats["total_tokens"] = self.decode_tokens
        stats["compression_ratio"] = (
            stats["total_chunks"] / max(1, self.decode_tokens) if self.decode_tokens > 0 else 1.0
        )
        return stats
