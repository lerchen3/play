"""
Correct LSE chunk cache implementation with:
1. Low-rank coefficient storage (actual r-dimensional vectors)
2. Correct LSE scaling (consistent 1/√d_k factor)
3. Cardinality weighting in attention (log_counts in logits)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import torch
from transformers.cache_utils import Cache


def _repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    """Replicates key/value tensors for grouped query attention."""
    batch, kv_heads, seq_len = hidden_states.shape[:3]
    if repeat == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, kv_heads, repeat, seq_len, *hidden_states.shape[3:])
    return hidden_states.reshape(batch, kv_heads * repeat, seq_len, *hidden_states.shape[4:])


class LSECacheCorrect(Cache):
    """
    LSE chunk merging with correct implementation:
    - Stores r-dimensional coefficients
    - Proper scaling in LSE computation
    - Uses log_counts in attention
    """
    
    is_sliding = False
    supports_chunk_merge = True  # Enable cardinality weighting

    def __init__(
        self,
        *,
        epsilon: float,
        num_layers: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        stats_path: str | Path,
        projection_cache_path: str | Path | None = Path("analysis/projections"),
        query_cache_path: str | Path = Path("dumps/deepseek_r1_qkv"),
        query_bank_size: int = 16384,
        solver_query_count: int = 1024,
        max_prefill_queries: int = 1024,
        max_decode_queries: int = 128,
        ridge: float = 1e-4,
        rank: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.rank = head_dim if rank is None or rank <= 0 else min(rank, head_dim)
        self.group_size = num_query_heads // num_kv_heads
        self.max_prefill_queries = max_prefill_queries
        self.max_decode_queries = max_decode_queries
        self.ridge = ridge
        self.scale = 1.0 / math.sqrt(head_dim)  # Attention scale factor
        self._projection_cache_path = (
            Path(projection_cache_path) if projection_cache_path is not None else None
        )
        self._query_cache_path = Path(query_cache_path)
        self.query_bank_size = int(query_bank_size)
        self.solver_query_count = int(solver_query_count)
        self.seed = int(seed)
        self._rng = torch.Generator()
        self._rng.manual_seed(self.seed)

        # Load stats
        stats = torch.load(Path(stats_path))
        sigma_q = stats["sigma_q"].to(torch.float64)

        # Compute whitening matrices
        self.whitening_matrices = torch.zeros((num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32)
        for layer in range(num_layers):
            for kv_head in range(num_kv_heads):
                q_indices = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
                sigma_group = 0.5 * (sigma_q[layer, q_indices] + sigma_q[layer, q_indices].transpose(-1, -2))
                sigma_avg = sigma_group.mean(dim=0)
                
                eigvals, eigvecs_sigma = torch.linalg.eigh(sigma_avg)
                eigvals = torch.clamp(eigvals, min=1e-8)
                sigma_inv_sqrt = eigvecs_sigma @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs_sigma.T
                self.whitening_matrices[layer, kv_head] = sigma_inv_sqrt.to(torch.float32)

        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self.P_r = self._load_projection_basis()

        # Storage: r-dimensional coefficients in whitened subspace
        self._coeffs: List[List[List[torch.Tensor]]] = []  # (layers, kv_heads, chunks) each (rank,)
        self._values: List[List[List[torch.Tensor]]] = []
        self._log_counts: List[List[List[float]]] = []
        self._reconstructed_keys: List[List[List[torch.Tensor]]] = []  # Cache reconstructed keys
        self._current_keys: List[List[List[torch.Tensor]]] = []  # Temp storage for active chunk
        self._current_values: List[List[List[torch.Tensor]]] = []
        
        # Cache invalidation
        self._cache_valid: List[bool] = [False] * num_layers
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._head_lengths: List[List[int]] = [[0 for _ in range(num_kv_heads)] for _ in range(num_layers)]
        self._query_bank_device: Optional[torch.device] = None

        # Metrics
        self.prefill_tokens = 0
        self.decode_tokens = 0

        for _ in range(num_layers):
            self._coeffs.append([[] for _ in range(num_kv_heads)])
            self._values.append([[] for _ in range(num_kv_heads)])
            self._log_counts.append([[] for _ in range(num_kv_heads)])
            self._reconstructed_keys.append([[] for _ in range(num_kv_heads)])
            self._current_keys.append([[] for _ in range(num_kv_heads)])
            self._current_values.append([[] for _ in range(num_kv_heads)])
        self._query_bank = self._load_query_bank()

    def _load_projection_basis(self) -> torch.Tensor:
        if self.rank <= 0:
            eye = torch.eye(self.head_dim, dtype=torch.float32)
            return eye.unsqueeze(0).unsqueeze(0).repeat(self.num_layers, self.num_kv_heads, 1, 1)

        basis_tensor: Optional[torch.Tensor] = None
        if self._projection_cache_path is not None:
            proj_file = self._projection_cache_path / f"projection_rank_{self.rank}.pt"
            if proj_file.exists():
                payload = torch.load(proj_file, map_location="cpu")
                basis_tensor = payload.get("basis")
                if basis_tensor is not None and basis_tensor.shape[3] >= self.rank:
                    basis_tensor = basis_tensor[:, :, :, : self.rank].to(torch.float32)
        if basis_tensor is None:
            init_path = Path("analysis/projection_init.pt")
            init_payload = torch.load(init_path, map_location="cpu")
            eigenvectors = init_payload["eigenvectors"].to(torch.float32)
            if eigenvectors.shape[3] < self.rank:
                raise ValueError(
                    f"Projection init rank {eigenvectors.shape[3]} smaller than requested rank {self.rank}"
                )
            basis_tensor = eigenvectors[:, :, :, : self.rank].contiguous()

        whitened_basis = torch.zeros(
            (self.num_layers, self.num_kv_heads, self.head_dim, self.rank),
            dtype=torch.float32,
        )
        for layer in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                raw = basis_tensor[layer, kv_head]
                W = self.whitening_matrices[layer, kv_head].to(dtype=torch.float32)
                transformed = W @ raw
                q, _ = torch.linalg.qr(transformed, mode="reduced")
                whitened_basis[layer, kv_head] = q[:, : self.rank]
        return whitened_basis

    def _load_query_bank(self) -> List[List[torch.Tensor]]:
        query_bank: List[List[torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            layer_dir = self._query_cache_path / f"layer_{layer_idx:02d}"
            if not layer_dir.exists():
                raise FileNotFoundError(f"Missing query cache directory: {layer_dir}")
            files = sorted(layer_dir.glob("batch_*.pt"))
            if not files:
                raise FileNotFoundError(f"No query cache batches found in {layer_dir}")

            per_head_batches: List[List[torch.Tensor]] = [[] for _ in range(self.num_kv_heads)]
            total_counts = [0] * self.num_kv_heads

            for batch_path in files:
                batch = torch.load(batch_path, map_location="cpu")
                if "q" not in batch:
                    continue
                queries = batch["q"].to(torch.float32).reshape(-1, self.num_query_heads, self.head_dim)
                for kv_head in range(self.num_kv_heads):
                    head_slice = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
                    head_queries = queries[:, head_slice, :].reshape(-1, self.head_dim)
                    per_head_batches[kv_head].append(head_queries)
                    total_counts[kv_head] += head_queries.shape[0]
                if all(count >= self.query_bank_size for count in total_counts):
                    break

            layer_bank: List[torch.Tensor] = []
            for kv_head in range(self.num_kv_heads):
                if not per_head_batches[kv_head]:
                    raise ValueError(f"No queries collected for layer {layer_idx}, head {kv_head}")
                combined = torch.cat(per_head_batches[kv_head], dim=0)
                if combined.shape[0] < self.query_bank_size:
                    raise ValueError(
                        f"Insufficient queries for layer {layer_idx}, head {kv_head}: "
                        f"got {combined.shape[0]}, need {self.query_bank_size}"
                    )
                head_generator = torch.Generator()
                head_generator.manual_seed(self.seed + layer_idx * self.num_kv_heads + kv_head)
                perm = torch.randperm(combined.shape[0], generator=head_generator)[: self.query_bank_size]
                sampled = combined[perm]
                layer_bank.append(sampled.contiguous())
            query_bank.append(layer_bank)
        return query_bank

    def _sample_queries(self, layer_idx: int, kv_head: int) -> torch.Tensor:
        bank = self._query_bank[layer_idx][kv_head]
        if bank.shape[0] <= self.solver_query_count:
            return bank.to(device=self._device, dtype=torch.float32)
        idx = torch.randperm(bank.shape[0], generator=self._rng)[: self.solver_query_count]
        return bank[idx].to(device=self._device, dtype=torch.float32)

    def _move_query_bank_to_device(self) -> None:
        if self._device is None or self._query_bank_device == self._device:
            return
        moved_bank: List[List[torch.Tensor]] = []
        for layer_bank in self._query_bank:
            moved_layer = [tensor.to(device=self._device, dtype=torch.float32) for tensor in layer_bank]
            moved_bank.append(moved_layer)
        self._query_bank = moved_bank
        self._query_bank_device = self._device

    def _to_subspace(self, vec: torch.Tensor, layer_idx: int, kv_head: int) -> torch.Tensor:
        """Transform to whitened r-dimensional subspace: P_r^T @ W @ vec"""
        W = self.whitening_matrices[layer_idx, kv_head].to(device=vec.device, dtype=vec.dtype)
        P_r = self.P_r[layer_idx, kv_head].to(device=vec.device, dtype=vec.dtype)

        if vec.dim() == 1:
            vec_whitened = torch.matmul(W, vec)
            return torch.matmul(P_r.T, vec_whitened)  # (rank,)
        else:
            vec_whitened = torch.matmul(vec, W.T)
            return torch.matmul(vec_whitened, P_r)  # (..., rank)
    
    def _from_subspace(self, coeffs: torch.Tensor, layer_idx: int, kv_head: int) -> torch.Tensor:
        """Reconstruct from r-dim coefficients: W^{-1} @ P_r @ coeffs"""
        P_r = self.P_r[layer_idx, kv_head].to(device=coeffs.device, dtype=coeffs.dtype)
        W = self.whitening_matrices[layer_idx, kv_head].to(device=coeffs.device, dtype=coeffs.dtype)
        W_inv = torch.linalg.inv(W.to(torch.float64)).to(torch.float32)
        
        if coeffs.dim() == 1:
            vec_whitened = torch.matmul(P_r, coeffs)
            return torch.matmul(W_inv, vec_whitened)
        else:
            vec_whitened = torch.matmul(coeffs, P_r.T)
            return torch.matmul(vec_whitened, W_inv.T)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = key_states.device
            self._dtype = key_states.dtype
            self._move_query_bank_to_device()
        
        if key_states.shape[2] > 1:
            return self._update_prefill(key_states, value_states, layer_idx, cache_kwargs)
        return self._update_decode(key_states, value_states, layer_idx, cache_kwargs)

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads, seq_len, head_dim = key_states.shape
        
        if layer_idx == 0:
            self.prefill_tokens += seq_len
        
        # Store prefill keys as singleton chunks
        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                for t in range(seq_len):
                    key = key_states[b, kv_head, t, :]
                    value = value_states[b, kv_head, t, :]
                    
                    # Store as r-dimensional coefficient
                    key_coeffs = self._to_subspace(key.float(), layer_idx, kv_head)
                    self._coeffs[layer_idx][kv_head].append(key_coeffs.to(self._dtype))
                    self._values[layer_idx][kv_head].append(value)
                    self._log_counts[layer_idx][kv_head].append(0.0)  # log(1) = 0
                    
                    # Cache reconstructed key (for prefill singletons, same as original)
                    self._reconstructed_keys[layer_idx][kv_head].append(key)
        
        self._cache_valid[layer_idx] = False
        return self._build_cache(layer_idx)

    def _update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads, seq_len, head_dim = key_states.shape
        
        if layer_idx == 0:
            self.decode_tokens += 1
        
        # Store decode query
        # Try adding to current chunk
        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                key = key_states[b, kv_head, 0, :]
                value = value_states[b, kv_head, 0, :]
                
                current_keys = self._current_keys[layer_idx][kv_head]
                candidate_keys = current_keys + [key]
                
                if len(candidate_keys) > 1:
                    # Check LSE error
                    lse_error = self._compute_lse_error(layer_idx, kv_head, candidate_keys)
                    
                    if lse_error > self.epsilon:
                        # Finalize current chunk
                        self._finalize_chunk(layer_idx, kv_head, 
                                           self._current_keys[layer_idx][kv_head],
                                           self._current_values[layer_idx][kv_head])
                        self._current_keys[layer_idx][kv_head] = [key]
                        self._current_values[layer_idx][kv_head] = [value]
                    else:
                        self._current_keys[layer_idx][kv_head] = candidate_keys
                        self._current_values[layer_idx][kv_head].append(value)
                else:
                    self._current_keys[layer_idx][kv_head] = candidate_keys
                    self._current_values[layer_idx][kv_head] = [value]
        
        self._cache_valid[layer_idx] = False
        return self._build_cache(layer_idx)

    def _compute_lse_error(self, layer_idx: int, kv_head: int, keys: List[torch.Tensor]) -> float:
        """Compute LSE distortion with correct scaling"""
        if len(keys) == 1:
            return 0.0
        
        queries_raw = self._sample_queries(layer_idx, kv_head)
        if queries_raw.shape[0] == 0:
            return 0.0

        # Transform to subspace
        queries = self._to_subspace(queries_raw, layer_idx, kv_head)
        keys_tensor = torch.stack(keys, dim=0).float()
        keys_subspace = self._to_subspace(keys_tensor, layer_idx, kv_head)  # (num_keys, rank)

        logits = torch.matmul(queries.float(), keys_subspace.T) * self.scale  # (m, num_keys)
        
        # Target: log(sum_k exp(scaled logits)) - log(|C|)
        targets = torch.logsumexp(logits, dim=1) - math.log(len(keys))  # (m,)
        
        # Solve for coefficients: min ||Q @ c * scale - targets||^2
        Q = queries.float()  # (m, rank)
        Q_scaled = Q * self.scale
        Q_t = Q_scaled.T
        cov = torch.matmul(Q_t, Q_scaled) + self.ridge * torch.eye(self.rank, device=Q.device)
        rhs = torch.matmul(Q_t, targets)
        coeffs = torch.linalg.solve(cov, rhs)  # (rank,)
        
        # Compute distortion
        logits_merged = torch.matmul(Q, coeffs) * self.scale
        exp_merged_times_C = torch.exp(logits_merged) * len(keys)
        exp_sum_original = torch.exp(logits).sum(dim=1)
        
        squared_errors = (exp_sum_original - exp_merged_times_C) ** 2
        lse_error = squared_errors.sum().item() / (queries.shape[0] * len(keys))
        
        return lse_error

    def _finalize_chunk(self, layer_idx: int, kv_head: int, keys: List[torch.Tensor], values: List[torch.Tensor]) -> None:
        """Finalize chunk by computing optimal r-dimensional coefficients"""
        if not keys:
            return
        
        if len(keys) == 1:
            key_coeffs = self._to_subspace(keys[0].float(), layer_idx, kv_head)
            mean_v = values[0]
        else:
            queries_raw = self._sample_queries(layer_idx, kv_head)
            if queries_raw.shape[0] == 0:
                queries = None
            else:
                queries = self._to_subspace(queries_raw, layer_idx, kv_head)

            keys_tensor = torch.stack(keys, dim=0).float()
            keys_subspace = self._to_subspace(keys_tensor, layer_idx, kv_head)

            if queries is not None and queries.shape[0] > 0:
                logits = torch.matmul(queries.float(), keys_subspace.T) * self.scale
                targets = torch.logsumexp(logits, dim=1) - math.log(len(keys))

                Q_scaled = queries.float() * self.scale
                Q_t = Q_scaled.T
                cov = torch.matmul(Q_t, Q_scaled) + self.ridge * torch.eye(self.rank, device=Q_scaled.device)
                rhs = torch.matmul(Q_t, targets)
                key_coeffs = torch.linalg.solve(cov, rhs)
            else:
                key_coeffs = keys_subspace.mean(dim=0)
            
            mean_v = torch.stack(values, dim=0).mean(dim=0)
        
        # Store coefficient and reconstruct key ONCE
        self._coeffs[layer_idx][kv_head].append(key_coeffs.to(self._dtype))
        self._values[layer_idx][kv_head].append(mean_v)
        self._log_counts[layer_idx][kv_head].append(math.log(len(keys)))
        
        # Cache reconstructed key to avoid repeated reconstruction
        key_reconstructed = self._from_subspace(key_coeffs.float(), layer_idx, kv_head)
        self._reconstructed_keys[layer_idx][kv_head].append(key_reconstructed.to(self._dtype))

    def _build_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build cache tensors by reconstructing keys and REPEATING them |C| times"""
        # Return cached version if valid
        debug_cache = os.environ.get("LSE_CORRECT_DEBUG_CACHE", "0") == "1"
        if self._cache_valid[layer_idx]:
            if debug_cache:
                cached_keys = self._key_cache[layer_idx]
                print(
                    f"[LSECacheCorrect] layer={layer_idx} cached length="
                    f"{0 if cached_keys is None else cached_keys.shape[2]}"
                )
            return self._key_cache[layer_idx], self._value_cache[layer_idx]
        
        # Count total entries (including repetitions for cardinality)
        total_entries = 0
        if debug_cache:
            print(f"[LSECacheCorrect] layer={layer_idx} rebuilding cache")
        for kv_head in range(self.num_kv_heads):
            # Each completed chunk contributes |C| entries
            for log_c in self._log_counts[layer_idx][kv_head]:
                count = int(round(math.exp(log_c)))
                total_entries += count
            # Current keys contribute 1 each
            total_entries += len(self._current_keys[layer_idx][kv_head])
        
        if total_entries == 0:
            empty_k = torch.empty(1, self.num_kv_heads, 0, self.head_dim, device=self._device, dtype=self._dtype)
            empty_v = torch.empty(1, self.num_kv_heads, 0, self.head_dim, device=self._device, dtype=self._dtype)
            return empty_k, empty_v
        
        max_len_per_head = total_entries // self.num_kv_heads + 100  # Buffer
        key_list = [[] for _ in range(self.num_kv_heads)]
        value_list = [[] for _ in range(self.num_kv_heads)]
        
        for kv_head in range(self.num_kv_heads):
            # Completed chunks: use cached reconstructed key, repeat |C| times
            for key_reconstructed, val, log_c in zip(self._reconstructed_keys[layer_idx][kv_head],
                                                     self._values[layer_idx][kv_head],
                                                     self._log_counts[layer_idx][kv_head]):
                count = int(round(math.exp(log_c)))
                
                # Repeat |C| times for correct cardinality weighting
                for _ in range(count):
                    key_list[kv_head].append(key_reconstructed)
                    value_list[kv_head].append(val)
            
            # Current chunk keys (not yet merged)
            for key, val in zip(self._current_keys[layer_idx][kv_head],
                                self._current_values[layer_idx][kv_head]):
                key_list[kv_head].append(key)
                value_list[kv_head].append(val)
            if debug_cache:
                print(
                    f"[LSECacheCorrect] layer={layer_idx} head={kv_head} entries={len(key_list[kv_head])}"
                )
            self._head_lengths[layer_idx][kv_head] = len(key_list[kv_head])

        # Stack into tensors
        max_len = max(len(key_list[h]) for h in range(self.num_kv_heads))
        key_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=self._device, dtype=self._dtype)
        value_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=self._device, dtype=self._dtype)
        
        for kv_head in range(self.num_kv_heads):
            if key_list[kv_head]:
                key_tensor[0, kv_head, :len(key_list[kv_head])] = torch.stack(key_list[kv_head], dim=0)
                value_tensor[0, kv_head, :len(value_list[kv_head])] = torch.stack(value_list[kv_head], dim=0)
        
        # Cache for reuse (store base kv heads)
        self._key_cache[layer_idx] = key_tensor
        self._value_cache[layer_idx] = value_tensor
        self._cache_valid[layer_idx] = True
        
        return key_tensor, value_tensor

    def __call__(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_tensor, value_tensor = self._build_cache(layer_idx)
        lengths = self._head_lengths[layer_idx]
        print(f"[LSECacheCorrect] layer={layer_idx} merged lengths={lengths}")
        key_tensor_exp = _repeat_kv(key_tensor, self.group_size)
        value_tensor_exp = _repeat_kv(value_tensor, self.group_size)
        return key_tensor_exp, value_tensor_exp

    def finalize_all_chunks(self) -> None:
        """Finalize all open chunks"""
        for layer_idx in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                if self._current_keys[layer_idx][kv_head]:
                    self._finalize_chunk(layer_idx, kv_head,
                                       self._current_keys[layer_idx][kv_head],
                                       self._current_values[layer_idx][kv_head])
                    self._current_keys[layer_idx][kv_head] = []
                    self._current_values[layer_idx][kv_head] = []
        self._cache_valid = [False] * self.num_layers

    def get_chunk_stats(self) -> Dict[str, float | int | Dict[int, int]]:
        stats: Dict[str, float | int | Dict[int, int]] = {
            "chunks_per_layer": {},
            "total_chunks": 0,
            "total_tokens": self.decode_tokens,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
        }
        chunk_sizes: List[int] = []
        for layer_idx in range(self.num_layers):
            layer_chunk_count = 0
            for kv_head in range(self.num_kv_heads):
                head_logs = self._log_counts[layer_idx][kv_head]
                layer_chunk_count += len(head_logs)
                for log_c in head_logs:
                    size = int(round(math.exp(log_c)))
                    chunk_sizes.append(size)
            stats["chunks_per_layer"][layer_idx] = layer_chunk_count
            stats["total_chunks"] += layer_chunk_count
        stats["total_entries"] = sum(chunk_sizes)
        stats["avg_chunk_size"] = (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0
        stats["compression_ratio"] = (
            stats["total_chunks"] / max(1, self.decode_tokens)
            if self.decode_tokens > 0
            else 1.0
        )
        stats["head_lengths"] = {
            layer_idx: list(self._head_lengths[layer_idx])
            for layer_idx in range(self.num_layers)
        }
        return stats

    def get_usable_length(self, layer_idx: int, batch_idx: int = 0) -> int:
        total = sum(len(self._coeffs[layer_idx][h]) + len(self._current_keys[layer_idx][h])
                   for h in range(self.num_kv_heads))
        return total // self.num_kv_heads

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.get_usable_length(layer_idx, 0)

    def get_max_length(self) -> Optional[int]:
        return None
