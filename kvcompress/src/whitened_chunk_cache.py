from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


CacheKey = Tuple[str, str, int, int, int, int, int, int, Optional[int], int]
_QUERY_BANK_CACHE: Dict[CacheKey, List[List[torch.Tensor]]] = {}


def _repeat_kv(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
    """Replicates key/value tensors for grouped query attention."""
    batch, kv_heads, seq_len = hidden_states.shape[:3]
    if repeat == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, kv_heads, repeat, seq_len, *hidden_states.shape[3:])
    return hidden_states.reshape(batch, kv_heads * repeat, seq_len, *hidden_states.shape[4:])


class WhitenedChunkCache(Cache):
    """
    Cache that merges keys using whitened space with L2 threshold.
    Uses sigma_q^(-1/2) to whiten queries and keys before computing distances.
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
        stats_path: str | Path,
        query_cache_path: str | Path = Path("dumps/deepseek_r1_qkv"),
        query_bank_size: int = 16384,
        solver_query_count: int = 1024,
        projection_cache_path: str | Path | None = Path("analysis/projections"),
        ridge: float = 1e-4,
        rank: int | None = None,  # Optional rank for projection
        seed: int = 0,
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
        self.query_bank_size = int(query_bank_size)
        self.solver_query_count = int(solver_query_count)
        self.ridge = ridge
        self.rank = rank
        self.seed = int(seed)
        self._rng = torch.Generator()
        self._rng.manual_seed(self.seed)
        self.log_kv_lengths = True

        if self.query_bank_size < self.solver_query_count:
            raise ValueError("query_bank_size must be >= solver_query_count.")

        self._stats_path = Path(stats_path)
        self._query_cache_path = Path(query_cache_path)
        self._projection_cache_path = (
            Path(projection_cache_path) if projection_cache_path is not None else None
        )

        # Load whitening matrices and optionally projection matrices
        stats = torch.load(self._stats_path)
        sigma_q = stats["sigma_q"].to(torch.float64)  # (layers, q_heads, dim, dim)

        # Compute sigma_q^(-1/2) for whitening
        self.whitening_matrices = torch.zeros((num_layers, num_kv_heads, head_dim, head_dim), dtype=torch.float32)
        for layer in range(num_layers):
            for kv_head in range(num_kv_heads):
                q_indices = slice(kv_head * self.group_size, (kv_head + 1) * self.group_size)
                sigma_group = 0.5 * (sigma_q[layer, q_indices] + sigma_q[layer, q_indices].transpose(-1, -2))
                sigma_avg = sigma_group.mean(dim=0)
                
                # Eigendecomposition
                eigvals, eigvecs = torch.linalg.eigh(sigma_avg)
                eigvals = torch.clamp(eigvals, min=1e-8)
                
                # Compute sigma^(-1/2)
                sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
                self.whitening_matrices[layer, kv_head] = sigma_inv_sqrt.to(torch.float32)

        self._precomputed_basis: Optional[torch.Tensor] = self._load_precomputed_basis()

        self._initialized: List[bool] = [False] * num_layers
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Per-layer storage
        self._completed: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._current: List[List[Dict[str, List[torch.Tensor]]]] = []
        self._log_counts_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._chunk_sizes: List[List[List[int]]] = []
        self._query_bank: List[List[torch.Tensor]] = self._load_query_bank()
        self._query_bank_device: Optional[torch.device] = None
        self._basis_device: Optional[torch.device] = None
        self.is_sliding = [False] * num_layers

        # Metrics
        self.prefill_tokens: int = 0
        self.decode_tokens: int = 0
        self.decode_chunks: int = 0

        for _ in range(num_layers):
            layer_completed: List[Dict[str, List[torch.Tensor]]] = []
            layer_current: List[Dict[str, List[torch.Tensor]]] = []
            layer_chunk_sizes: List[List[int]] = []
            for _head in range(num_kv_heads):
                layer_completed.append({"keys": [], "values": [], "log_counts": []})
                layer_current.append({"keys": [], "values": []})
                layer_chunk_sizes.append([])
            self._completed.append(layer_completed)
            self._current.append(layer_current)
            self._chunk_sizes.append(layer_chunk_sizes)

    def _load_precomputed_basis(self) -> Optional[torch.Tensor]:
        if self.rank is None or self.rank <= 0 or self._projection_cache_path is None:
            return None
        file_path = self._projection_cache_path / f"projection_rank_{self.rank}.pt"
        if not file_path.exists():
            print(f"Warning: projection file not found at {file_path}, proceeding without precomputed basis.")
            return None
        proj_data = torch.load(file_path, map_location="cpu")
        basis = proj_data.get("basis")
        if basis is None:
            raise ValueError(f"Projection file {file_path} missing 'basis' tensor.")
        if basis.shape[0] != self.num_layers or basis.shape[1] != self.num_kv_heads or basis.shape[2] != self.head_dim:
            raise ValueError(
                f"Projection basis shape {basis.shape} does not match expected "
                f"({self.num_layers}, {self.num_kv_heads}, {self.head_dim}, {self.rank})."
            )
        if basis.shape[3] < self.rank:
            raise ValueError(
                f"Projection basis rank {basis.shape[3]} smaller than requested rank {self.rank}."
            )
        basis = basis[:, :, :, : self.rank].contiguous().to(torch.float32)

        # Transform to whitened coordinates and re-orthonormalize
        whitened_basis = torch.zeros_like(basis)
        for layer in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                raw = basis[layer, kv_head]
                W = self.whitening_matrices[layer, kv_head].to(dtype=raw.dtype, device=raw.device)
                transformed = W @ raw
                # QR to enforce orthonormality in whitened space
                q, _ = torch.linalg.qr(transformed, mode="reduced")
                whitened_basis[layer, kv_head] = q[:, : self.rank]
        return whitened_basis

    def _load_query_bank(self) -> List[List[torch.Tensor]]:
        cache_key = (
            str(self._query_cache_path.resolve()),
            str(self._stats_path.resolve()),
            self.num_layers,
            self.num_kv_heads,
            self.num_query_heads,
            self.head_dim,
            self.query_bank_size,
            self.solver_query_count,
            self.rank,
            self.seed,
        )
        if cache_key in _QUERY_BANK_CACHE:
            base_bank = _QUERY_BANK_CACHE[cache_key]
            return [[tensor.clone() for tensor in layer] for layer in base_bank]

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
                whitened = self._whiten(sampled, layer_idx, kv_head).to(torch.float32).contiguous()
                layer_bank.append(whitened)

            query_bank.append(layer_bank)

        _QUERY_BANK_CACHE[cache_key] = [[tensor.clone() for tensor in layer] for layer in query_bank]
        return [[tensor.clone() for tensor in layer] for layer in query_bank]

    def _move_query_bank_to_device(self) -> None:
        if not self._query_bank:
            return
        if self._device is None or self._query_bank_device == self._device:
            pass
        else:
            moved_bank: List[List[torch.Tensor]] = []
            for layer in self._query_bank:
                moved_layer = [tensor.to(device=self._device) for tensor in layer]
                moved_bank.append(moved_layer)
            self._query_bank = moved_bank
            self._query_bank_device = self._device
            self._rng = torch.Generator(device=self._device)
            self._rng.manual_seed(self.seed)

        if self._precomputed_basis is not None and self._basis_device != self._device:
            self._precomputed_basis = self._precomputed_basis.to(device=self._device)
            self._basis_device = self._device

    def _sample_queries(self, layer_idx: int, kv_head: int) -> torch.Tensor:
        if not self._query_bank:
            return torch.empty(0, self.head_dim, device=self._device, dtype=torch.float32)
        queries = self._query_bank[layer_idx][kv_head]
        if queries.shape[0] <= self.solver_query_count:
            return queries
        indices = torch.randperm(
            queries.shape[0],
            generator=self._rng,
            device=queries.device,
        )[: self.solver_query_count]
        return queries[indices]

    def _compute_chunk_basis(
        self,
        layer_idx: int,
        kv_head: int,
        keys_whitened: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.rank is None or self.rank <= 0:
            return None
        if self._precomputed_basis is not None:
            return self._precomputed_basis[layer_idx, kv_head].to(
                device=keys_whitened.device, dtype=keys_whitened.dtype
            )
        max_rank = min(self.rank, keys_whitened.shape[0], keys_whitened.shape[1])
        if max_rank <= 0:
            return None
        try:
            _u, _s, vh = torch.linalg.svd(keys_whitened, full_matrices=False)
            basis = vh[:max_rank, :].T  # (dim, max_rank)
        except torch.linalg.LinAlgError:
            cov = torch.matmul(keys_whitened.T, keys_whitened)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            basis = eigvecs[:, -max_rank:]
        return basis

    @staticmethod
    def _project(matrix: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        # matrix: (..., dim), basis: (dim, rank)
        return torch.matmul(matrix, basis)

    def _whiten(self, vec: torch.Tensor, layer_idx: int, kv_head: int) -> torch.Tensor:
        """Apply whitening transformation: sigma^(-1/2) @ vec."""
        W = self.whitening_matrices[layer_idx, kv_head].to(device=vec.device, dtype=vec.dtype)
        if vec.dim() == 1:
            vec_whitened = torch.matmul(W, vec)
        else:
            vec_whitened = torch.matmul(vec, W.T)
        return vec_whitened

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._initialize_or_update_metadata(key_states)

        # Handle prefill vs decode
        if key_states.shape[2] > 1:
            return self._update_prefill(key_states, value_states, layer_idx)
        return self._update_decode(key_states, value_states, layer_idx)

    def _initialize_or_update_metadata(self, ref_tensor: torch.Tensor) -> None:
        if self._device is None:
            self._device = ref_tensor.device
            self._dtype = ref_tensor.dtype
            self._move_query_bank_to_device()

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # During prefill, store everything as singleton chunks
        batch, num_heads, seq_len, head_dim = key_states.shape
        
        # Store queries for later use
        # query_states: (batch, num_heads, seq_len, head_dim)
        # Count prefill tokens (only once, on first layer)
        if layer_idx == 0:
            self.prefill_tokens += seq_len
        
        # Store each prefill key as a singleton chunk (no merging during prefill)
        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                for t in range(seq_len):
                    key = key_states[b, kv_head, t, :]
                    value = value_states[b, kv_head, t, :]
                    log_count = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                    
                    self._completed[layer_idx][kv_head]["keys"].append(key)
                    self._completed[layer_idx][kv_head]["values"].append(value)
                    self._completed[layer_idx][kv_head]["log_counts"].append(log_count)
                    self._chunk_sizes[layer_idx][kv_head].append(1)
        
        # Initialize empty current chunks for decode (don't start merging prefill tokens)
        # The first decode token will start a new chunk

        self._initialized[layer_idx] = True
        self._invalidate_cache(layer_idx)
        key_tensor, value_tensor, _ = self._build_cache_tensors(layer_idx)
        return key_tensor, value_tensor

    def _update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads, seq_len, head_dim = key_states.shape
        assert seq_len == 1
        
        # Count decode tokens only once per step (on layer 0)
        if layer_idx == 0:
            self.decode_tokens += 1

        for b in range(batch):
            for kv_head in range(self.num_kv_heads):
                key = key_states[b, kv_head, 0, :]
                value = value_states[b, kv_head, 0, :]
                
                current_keys = self._current[layer_idx][kv_head]["keys"]
                current_values = self._current[layer_idx][kv_head]["values"]
                
                # Try adding to current chunk
                candidate_keys = current_keys + [key]
                candidate_values = current_values + [value]
                
                # Check LSE error in whitened space
                if len(candidate_keys) > 1:
                    # Compute LSE error for candidate chunk
                    lse_error = self._compute_lse_error(layer_idx, kv_head, candidate_keys)
                    
                    if lse_error > self.epsilon:
                        # Close current chunk (without new key)
                        self._finalize_chunk(layer_idx, kv_head, current_keys, current_values)
                        self.decode_chunks += 1
                        self._current[layer_idx][kv_head]["keys"] = [key]
                        self._current[layer_idx][kv_head]["values"] = [value]
                    else:
                        # Add to current chunk
                        self._current[layer_idx][kv_head]["keys"] = candidate_keys
                        self._current[layer_idx][kv_head]["values"] = candidate_values
                else:
                    # First key in chunk
                    self._current[layer_idx][kv_head]["keys"] = candidate_keys
                    self._current[layer_idx][kv_head]["values"] = candidate_values

        self._invalidate_cache(layer_idx)
        key_tensor, value_tensor, _ = self._build_cache_tensors(layer_idx)
        return key_tensor, value_tensor

    def _compute_lse_error(
        self,
        layer_idx: int,
        kv_head: int,
        keys: List[torch.Tensor],
    ) -> float:
        """
        Compute LSE (log-sum-exp) error for merging keys.
        
        We fit k_c so that: exp(q_i^T k_c) * |C| ≈ sum_{j in C} exp(q_i^T k_j)
        That is: q_i^T k_c ≈ log(sum_j exp(q_i^T k_j)) - log(|C|)
        
        The - log(|C|) term makes k_c smaller/regularized compared to just matching the sum.
        Even though we don't use log(|C|) in attention, this regularization helps:
        - Creates denoised, higher-quality representative keys
        - Achieves better compression (more keys merge successfully)
        - Better perplexity than the "textbook" version
        
        The merged value is the mean: v_c = mean(v_j).
        
        Distortion: epsilon = 1/(m * |C|) * sum_{i=1 to m} [sum_{j in C} exp(q_i^T k_j) - exp(q_i^T k_c) * |C|]^2
        
        where:
        - m = number of queries
        - |C| = number of keys in chunk
        - q_i = whitened query
        - k_c = merged center key (whitened)
        - k_j = individual keys in chunk (whitened)
        """
        if len(keys) == 1:
            return 0.0
        keys_tensor = torch.stack(keys, dim=0).to(torch.float32)
        keys_whitened = self._whiten(keys_tensor, layer_idx, kv_head)
        basis = self._compute_chunk_basis(layer_idx, kv_head, keys_whitened)
        if basis is not None:
            keys_proj = self._project(keys_whitened, basis)
            center_proj = keys_proj.mean(dim=0)
            center_whitened = torch.matmul(basis, center_proj)
        else:
            center_whitened = keys_whitened.mean(dim=0)
        return torch.norm(keys_whitened - center_whitened.unsqueeze(0), dim=1).mean().item()

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
        keys_whitened = self._whiten(keys_tensor.to(torch.float32), layer_idx, kv_head)
        basis = self._compute_chunk_basis(layer_idx, kv_head, keys_whitened)
        keys_projected = self._project(keys_whitened, basis) if basis is not None else None

        if len(keys) == 1:
            k_center = keys_tensor[0].to(torch.float32)
            mean_v = values_tensor[0]
        else:
            queries = self._sample_queries(layer_idx, kv_head).to(device=keys_whitened.device, dtype=torch.float32)
            has_queries = queries.numel() > 0 and queries.shape[0] >= 2

            if basis is not None and basis.numel() > 0:
                if has_queries:
                    Q = self._project(queries, basis)
                    scale = 1.0 / math.sqrt(basis.shape[1])
                    logits = torch.matmul(Q, keys_projected.T) * scale
                    targets = torch.logsumexp(logits, dim=1) - math.log(len(keys))
                    Q_t = Q.T
                    cov = torch.matmul(Q_t, Q)
                    cov = cov + self.ridge * torch.eye(basis.shape[1], dtype=torch.float32, device=cov.device)
                    rhs = torch.matmul(Q_t, targets)
                    k_proj = torch.linalg.solve(cov, rhs)
                    k_center_whitened = torch.matmul(basis, k_proj)
                else:
                    center_proj = keys_projected.mean(dim=0)
                    k_center_whitened = torch.matmul(basis, center_proj)
            else:
                if has_queries:
                    Q = queries
                    scale = 1.0 / math.sqrt(self.head_dim)
                    logits = torch.matmul(Q, keys_whitened.T) * scale
                    targets = torch.logsumexp(logits, dim=1) - math.log(len(keys))
                    Q_t = Q.T
                    cov = torch.matmul(Q_t, Q)
                    cov = cov + self.ridge * torch.eye(self.head_dim, dtype=torch.float32, device=cov.device)
                    rhs = torch.matmul(Q_t, targets)
                    k_center_whitened = torch.linalg.solve(cov, rhs)
                else:
                    k_center_whitened = keys_whitened.mean(dim=0)

            if len(keys) > 1:
                W = self.whitening_matrices[layer_idx, kv_head].to(device=keys_whitened.device)
                W_inv = torch.linalg.inv(W.to(torch.float64)).to(torch.float32)
                k_center = torch.matmul(W_inv, k_center_whitened)
                mean_v = values_tensor.mean(dim=0)

        key_tensor = k_center.to(device=self._device, dtype=self._dtype)
        value_tensor = mean_v.to(device=self._device, dtype=self._dtype)
        log_count = torch.tensor(math.log(len(keys)), dtype=torch.float32, device=self._device)

        self._completed[layer_idx][kv_head]["keys"].append(key_tensor)
        self._completed[layer_idx][kv_head]["values"].append(value_tensor)
        self._completed[layer_idx][kv_head]["log_counts"].append(log_count)
        self._chunk_sizes[layer_idx][kv_head].append(len(keys))

    def _invalidate_cache(self, layer_idx: int) -> None:
        self._key_cache[layer_idx] = None
        self._value_cache[layer_idx] = None
        self._log_counts_cache[layer_idx] = None

    def get_usable_length(self, layer_idx: int, batch_idx: int = 0) -> int:
        total = sum(len(self._completed[layer_idx][h]["keys"]) for h in range(self.num_kv_heads))
        total += sum(len(self._current[layer_idx][h]["keys"]) for h in range(self.num_kv_heads))
        return total // self.num_kv_heads

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.get_usable_length(layer_idx, 0)

    def get_max_length(self) -> Optional[int]:
        return None

    def _build_cache_tensors(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._key_cache[layer_idx] is not None:
            return self._key_cache[layer_idx], self._value_cache[layer_idx], self._log_counts_cache[layer_idx]

        device = self._device
        dtype = self._dtype

        # Collect all entries
        head_entries = []
        head_values = []
        head_log_counts = []
        max_len = 0
        for kv_head in range(self.num_kv_heads):
            completed_keys = self._completed[layer_idx][kv_head]["keys"]
            completed_vals = self._completed[layer_idx][kv_head]["values"]
            completed_logs = self._completed[layer_idx][kv_head]["log_counts"]
            current_keys = self._current[layer_idx][kv_head]["keys"]
            current_vals = self._current[layer_idx][kv_head]["values"]

            all_keys = completed_keys + current_keys
            all_vals = completed_vals + current_vals
            all_logs = completed_logs + [torch.tensor(0.0, device=device, dtype=torch.float32)] * len(current_keys)

            if all_keys:
                head_entries.append(torch.stack(all_keys, dim=0))
                head_values.append(torch.stack(all_vals, dim=0))
                head_log_counts.append(torch.stack(all_logs, dim=0))
                max_len = max(max_len, len(all_keys))
            else:
                head_entries.append(torch.empty(0, self.head_dim, device=device, dtype=dtype))
                head_values.append(torch.empty(0, self.head_dim, device=device, dtype=dtype))
                head_log_counts.append(torch.empty(0, device=device, dtype=torch.float32))

        key_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=device, dtype=dtype)
        value_tensor = torch.zeros(1, self.num_kv_heads, max_len, self.head_dim, device=device, dtype=dtype)
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

    def __call__(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query_states: (batch, num_heads, seq_len, head_dim)
        key_tensor, value_tensor, log_count_tensor = self._build_cache_tensors(layer_idx)
        
        key_tensor_exp = _repeat_kv(key_tensor, self.group_size)
        value_tensor_exp = _repeat_kv(value_tensor, self.group_size)
        log_count_tensor_exp = _repeat_kv(log_count_tensor.unsqueeze(-1), self.group_size).squeeze(-1)

        return key_tensor_exp, value_tensor_exp

    def finalize_all_chunks(self) -> None:
        """Finalize all open chunks at the end of generation"""
        for layer_idx in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                current_keys = self._current[layer_idx][kv_head]["keys"]
                current_values = self._current[layer_idx][kv_head]["values"]
                if current_keys:
                    self._finalize_chunk(layer_idx, kv_head, current_keys, current_values)
                    self._current[layer_idx][kv_head]["keys"] = []
                    self._current[layer_idx][kv_head]["values"] = []
    
    def get_chunk_stats(self) -> Dict:
        stats = {"chunks_per_layer": {}, "total_chunks": 0, "total_tokens": self.decode_tokens}
        for layer_idx in range(self.num_layers):
            layer_chunks = sum(len(self._chunk_sizes[layer_idx][h]) for h in range(self.num_kv_heads))
            stats["chunks_per_layer"][layer_idx] = layer_chunks
            stats["total_chunks"] += layer_chunks
        stats["compression_ratio"] = stats["total_chunks"] / max(1, self.decode_tokens) if self.decode_tokens > 0 else 1.0
        return stats
