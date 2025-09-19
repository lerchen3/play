import torch
import torch.nn as nn
import torch.nn.functional as F

from .casual import _attention as casual_attention
from .select import select_attention
from .topk import topk_indices
from model.rope import RoPE


class NativeSparseAttention(nn.Module):
    """
    NSA for GQA: orchestrates three kernels – compressed (FA2/causal), selected (vertical-slice), and sliding-window (FA2 with WINDOW_SIZE).
    During decoding, caches K/V per branch and compressed block projections; uses torch for single-token scoring.
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_head, seq_len,
                 cmp_blk_size: int = 32, cmp_stride: int = 16,
                 slc_top_n: int = 16, window_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.d_head = d_head
        self.seq_len = seq_len
        self.cmp_blk_size = cmp_blk_size
        self.cmp_stride = cmp_stride
        self.slc_top_n = slc_top_n
        self.window_size = window_size

        # Gates per head
        self.gate = nn.Linear(d_model, 3 * n_q_heads, bias=True)
        # Projections per branch
        self.W_k_cmp = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.W_v_cmp = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.W_k_slc = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.W_v_slc = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.W_k_win = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.W_v_win = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.rope = RoPE(d_head, seq_len)

        # Learned compressor MLPs and intra-block positional embedding (shared across heads)
        self.block_pos = nn.Parameter(torch.zeros(cmp_blk_size, d_head))
        in_dim = cmp_blk_size * d_head
        self.compress_k1 = nn.Linear(in_dim, d_head, bias=True)
        self.compress_k2 = nn.Linear(d_head, d_head, bias=True)
        self.compress_v1 = nn.Linear(in_dim, d_head, bias=True)
        self.compress_v2 = nn.Linear(d_head, d_head, bias=True)

        # Decode-time caches
        self._cache_cmp_k = None
        self._cache_cmp_v = None
        self._cache_slc_k = None
        self._cache_slc_v = None
        self._cache_win_k = None
        self._cache_win_v = None
        self._cmp_block_summary_k = None
        self._cmp_block_summary_v = None
        self._cached_len = 0

    def reset_cache(self):
        self._cache_cmp_k = None
        self._cache_cmp_v = None
        self._cache_slc_k = None
        self._cache_slc_v = None
        self._cache_win_k = None
        self._cache_win_v = None
        self._cmp_block_summary_k = None
        self._cmp_block_summary_v = None
        self._cached_len = 0

    def _sanitize_linear(self, linear: nn.Linear):
        if torch.isnan(linear.weight).any():
            linear.weight.data.nan_to_num_(nan=0.0)
        if linear.bias is not None and torch.isnan(linear.bias).any():
            linear.bias.data.nan_to_num_(nan=0.0)

    def _compress_block_pair(self, kblk: torch.Tensor, vblk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a single block of K/V values down to summaries with NaN safeguards."""
        kproj, vproj = self._compress_blocks(
            kblk.unsqueeze(1), vblk.unsqueeze(1)
        )
        return kproj[:, 0], vproj[:, 0]

    def _compress_blocks(self, kblocks: torch.Tensor, vblocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized block compression for (B, nb, blk, H, D) inputs."""
        B, nb, blk, H, D = kblocks.shape
        out_dtype = kblocks.dtype
        kflat = kblocks.permute(0, 1, 3, 2, 4).contiguous().view(B * nb * H, blk * D)
        vflat = vblocks.permute(0, 1, 3, 2, 4).contiguous().view(B * nb * H, blk * D)
        compute_dtype = self.compress_k1.weight.dtype

        self._sanitize_linear(self.compress_k1)
        self._sanitize_linear(self.compress_k2)
        self._sanitize_linear(self.compress_v1)
        self._sanitize_linear(self.compress_v2)

        kproj = self.compress_k2(F.gelu(self.compress_k1(kflat.to(compute_dtype)))).view(B, nb, H, D)
        vproj = self.compress_v2(F.gelu(self.compress_v1(vflat.to(compute_dtype)))).view(B, nb, H, D)

        if not torch.isfinite(kproj).all():
            kproj = kblocks.mean(dim=2)
        if not torch.isfinite(vproj).all():
            vproj = vblocks.mean(dim=2)
        return kproj.to(out_dtype), vproj.to(out_dtype)

    def _build_cmp_blocks(
        self, k_cmp: torch.Tensor, v_cmp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Extract padded sliding blocks and compress them for summary attention."""
        B, T, H, D = k_cmp.shape
        stride = self.cmp_stride
        blk = self.cmp_blk_size
        remainder = max(0, T - blk)
        total_blocks = 1 + (remainder + stride - 1) // stride if stride > 0 else 1
        total_blocks = max(total_blocks, 1)
        total_span = (total_blocks - 1) * stride + blk
        pad_len = total_span - T

        if pad_len > 0:
            pad_shape = (B, pad_len, H, D)
            k_pad = torch.zeros(pad_shape, device=k_cmp.device, dtype=k_cmp.dtype)
            v_pad = torch.zeros(pad_shape, device=v_cmp.device, dtype=v_cmp.dtype)
            k_cmp = torch.cat([k_cmp, k_pad], dim=1)
            v_cmp = torch.cat([v_cmp, v_pad], dim=1)

        k_cmp = k_cmp.contiguous()
        v_cmp = v_cmp.contiguous()
        k_blocks = k_cmp.unfold(1, blk, stride).permute(0, 1, 3, 2, 4).contiguous()
        v_blocks = v_cmp.unfold(1, blk, stride).permute(0, 1, 3, 2, 4).contiguous()

        block_starts = torch.arange(total_blocks, device=k_cmp.device, dtype=torch.int32) * stride
        max_valid = max(T - 1, 0)
        if max_valid == 0:
            block_starts = block_starts.clamp(max=0)
        else:
            block_starts = block_starts.clamp(max=max_valid)

        block_pos = self.block_pos.to(k_blocks.dtype).view(1, 1, blk, 1, D)
        k_blocks = k_blocks + block_pos
        v_blocks = v_blocks + block_pos

        k_summary, v_summary = self._compress_blocks(k_blocks, v_blocks)
        return k_summary, v_summary, block_starts, total_blocks

    def _fallback_sliding_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    scale: float, window_size: int | None) -> torch.Tensor:
        """Pure PyTorch causal attention used when Triton kernel produces NaNs."""
        B, H, T, D = q.shape
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        idx = torch.arange(T, device=q.device)
        causal = idx[None, None, :, None] >= idx[None, None, None, :]
        if window_size is not None and window_size > 0:
            causal &= (idx[None, None, :, None] - idx[None, None, None, :]) < window_size
        scores = scores.masked_fill(~causal, float('-inf'))
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        return probs @ v

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, q: torch.Tensor):
        """Build NSA caches from full prompt x (B,S,D) and q (B,n_q,S,d)."""
        self.reset_cache()
        B, T, _ = x.shape
        device = x.device
        # Project and cache per-branch K/V
        k_cmp = self.rope(self.W_k_cmp(x).view(B, T, self.n_kv, self.d_head), offset=0)
        v_cmp = self.W_v_cmp(x).view(B, T, self.n_kv, self.d_head)
        k_slc = self.rope(self.W_k_slc(x).view(B, T, self.n_kv, self.d_head), offset=0)
        v_slc = self.W_v_slc(x).view(B, T, self.n_kv, self.d_head)
        k_win = self.rope(self.W_k_win(x).view(B, T, self.n_kv, self.d_head), offset=0)
        v_win = self.W_v_win(x).view(B, T, self.n_kv, self.d_head)
        self._cache_cmp_k, self._cache_cmp_v = k_cmp, v_cmp
        self._cache_slc_k, self._cache_slc_v = k_slc, v_slc
        self._cache_win_k, self._cache_win_v = k_win, v_win
        # Build block summaries (KV heads)
        summaries_k, summaries_v, block_starts, total_blocks = self._build_cmp_blocks(k_cmp.contiguous(), v_cmp.contiguous())
        self._cmp_block_summary_k = summaries_k  # (B, nb, n_kv, d)
        self._cmp_block_summary_v = summaries_v
        self._block_starts = block_starts.tolist()
        self._cached_len = T

    def _expand_kv_to_q(self, t):
        # (B, L, n_kv, d) -> (B, L, n_q, d)
        if self.n_q == self.n_kv:
            return t
        n_rep = self.n_q // self.n_kv
        B, L = t.shape[:2]
        return t.unsqueeze(2).expand(B, L, n_rep, self.n_kv, self.d_head).reshape(B, L, self.n_q, self.d_head)

    def forward(self, x, q, is_decoding: bool = False, return_components: bool = False):
        B, T, D = x.shape
        device = x.device
        scale = (self.d_head) ** -0.5

        gates = torch.sigmoid(self.gate(x)).view(B, T, self.n_q, 3).permute(0, 2, 1, 3)
        g_cmp, g_slc, g_win = gates[..., 0], gates[..., 1], gates[..., 2]  # (B, n_q, T)

        if is_decoding and T == 1:
            # Single-token decode: update caches and compute outputs for last token only using torch
            t_new = self._cached_len
            # Update per-branch K/V caches with new token projections
            k_cmp_new = self.rope(self.W_k_cmp(x).view(B, 1, self.n_kv, self.d_head), offset=t_new)
            v_cmp_new = self.W_v_cmp(x).view(B, 1, self.n_kv, self.d_head)
            k_slc_new = self.rope(self.W_k_slc(x).view(B, 1, self.n_kv, self.d_head), offset=t_new)
            v_slc_new = self.W_v_slc(x).view(B, 1, self.n_kv, self.d_head)
            k_win_new = self.rope(self.W_k_win(x).view(B, 1, self.n_kv, self.d_head), offset=t_new)
            v_win_new = self.W_v_win(x).view(B, 1, self.n_kv, self.d_head)
            if self._cache_cmp_k is None:
                self._cache_cmp_k, self._cache_cmp_v = k_cmp_new, v_cmp_new
                self._cache_slc_k, self._cache_slc_v = k_slc_new, v_slc_new
                self._cache_win_k, self._cache_win_v = k_win_new, v_win_new
                self._block_starts = [0]
            else:
                self._cache_cmp_k = torch.cat([self._cache_cmp_k, k_cmp_new], dim=1)
                self._cache_cmp_v = torch.cat([self._cache_cmp_v, v_cmp_new], dim=1)
                self._cache_slc_k = torch.cat([self._cache_slc_k, k_slc_new], dim=1)
                self._cache_slc_v = torch.cat([self._cache_slc_v, v_slc_new], dim=1)
                self._cache_win_k = torch.cat([self._cache_win_k, k_win_new], dim=1)
                self._cache_win_v = torch.cat([self._cache_win_v, v_win_new], dim=1)
            Ttot = self._cache_cmp_k.shape[1]
            summaries_k, summaries_v, block_starts_tensor, _ = self._build_cmp_blocks(
                self._cache_cmp_k.contiguous(),
                self._cache_cmp_v.contiguous(),
            )
            self._cmp_block_summary_k = summaries_k
            self._cmp_block_summary_v = summaries_v
            self._block_starts = block_starts_tensor.tolist()
            self._cached_len = Ttot

            # Shapes for selection: reshape q last to (B, G, Hg, d)
            G = self.n_kv
            Hg = self.n_q // self.n_kv
            q_last = q[:, :, -1, :].view(B, G, Hg, self.d_head)
            # Compressed summaries per group: expand KV to groups
            Kcmp = self._cmp_block_summary_k  # (B, nb, n_kv, d)
            Vcmp = self._cmp_block_summary_v
            nb = Kcmp.shape[1]
            Kcmp_g = Kcmp.permute(0, 2, 1, 3)  # (B, G, nb, d)
            Vcmp_g = Vcmp.permute(0, 2, 1, 3)
            # Compute per-head scores and sum within group
            attn_cmp = torch.einsum('bghd,bgnd->bghn', q_last, Kcmp_g) * scale  # (B,G,Hg,nb)
            group_scores = attn_cmp.sum(dim=2)  # (B,G,nb)
            # Select top-n blocks per group (with forced sink and last two)
            topn = min(self.slc_top_n, nb)
            top_idx = torch.topk(group_scores, k=topn, dim=-1).indices  # (B,G,topn)
            top_idx = torch.clamp(top_idx, max=nb - 1)
            sink = torch.zeros(B, G, 1, dtype=torch.long, device=device)
            recent_idxs = torch.stack([torch.full((B, G), max(0, nb - 2), device=device, dtype=torch.long),
                                       torch.full((B, G), max(0, nb - 1), device=device, dtype=torch.long)], dim=-1)  # (B,G,2)
            # Build selected block sets per (B,G)
            sel_blocks = []
            for b in range(B):
                sel_blocks_b = []
                for g in range(G):
                    merged = torch.unique(torch.cat([sink[b, g], recent_idxs[b, g], top_idx[b, g]], dim=0), sorted=True)
                    sel_blocks_b.append(merged)
                sel_blocks.append(sel_blocks_b)

            # Compute compressed output for last token
            prob_cmp = torch.softmax(attn_cmp, dim=-1)  # (B,G,Hg,nb)
            out_cmp_heads = torch.einsum('bghn,bgnd->bghd', prob_cmp, Vcmp_g).unsqueeze(3)  # (B,G,Hg,1,d)
            out_cmp_heads = torch.nan_to_num(out_cmp_heads)

            # Selected branch: build indices and use indices-based kernel
            nb = len(self._block_starts)
            sel_blocks_all = torch.cat([
                torch.zeros(B, G, 1, dtype=torch.long, device=device),
                torch.full((B, G, 1), max(0, nb - 2), dtype=torch.long, device=device),
                torch.full((B, G, 1), max(0, nb - 1), dtype=torch.long, device=device),
                top_idx
            ], dim=-1)  # (B,G,3+topn)
            blk_mask_bg = torch.zeros(B, G, nb, dtype=torch.bool, device=device)
            blk_mask_bg.scatter_(dim=2, index=sel_blocks_all, src=torch.ones_like(sel_blocks_all, dtype=torch.bool))
            block_count = blk_mask_bg.sum(dim=-1).to(torch.int32)  # (B,G)
            Kmax = min(nb, sel_blocks_all.shape[-1])
            idx_range = torch.arange(nb, device=device, dtype=torch.int32)
            idx_exp = idx_range.view(1, 1, nb).expand(B, G, nb)
            masked = torch.where(blk_mask_bg, idx_exp, torch.full_like(idx_exp, nb))
            sorted_idx, _ = torch.sort(masked, dim=-1)
            block_idx = sorted_idx[:, :, :Kmax]
            block_idx = torch.where(block_idx == nb, torch.zeros_like(block_idx), block_idx)
            block_count = torch.minimum(block_count, torch.tensor(Kmax, device=device, dtype=torch.int32)).unsqueeze(2)
            block_idx = block_idx.unsqueeze(2)  # (B,G,1,Kmax)
            # Prepare inputs for kernel
            q_slc = q_last.mean(dim=2).unsqueeze(2)  # (B,G,1,d)
            Kslc_full = self._cache_slc_k.permute(0, 2, 1, 3)  # (B,G,Ttot,d)
            Vslc_full = self._cache_slc_v.permute(0, 2, 1, 3)
            out_slc_group = select_attention(q_slc, Kslc_full, Vslc_full, scale, block_idx, block_count, self.cmp_blk_size)  # (B,G,1,d)
            out_slc_heads = out_slc_group.unsqueeze(2).expand(B, G, Hg, 1, self.d_head)
            out_slc_heads = torch.nan_to_num(out_slc_heads)

            # Sliding window branch via kernel for last position
            win = self.window_size or Ttot
            start = max(0, Ttot - win)
            q_win = q_last.view(B, G * Hg, self.d_head)  # (B, n_q, d) collapsed by groups
            q_win = q_win.view(B, self.n_q, self.d_head).unsqueeze(2)  # (B, n_q, 1, d)
            Kwin = self._cache_win_k[:, start:Ttot]  # (B, Lw, n_kv, d)
            Vwin = self._cache_win_v[:, start:Ttot]
            Kwin_q = self._expand_kv_to_q(Kwin).permute(0, 2, 1, 3)  # (B, n_q, Lw, d)
            Vwin_q = self._expand_kv_to_q(Vwin).permute(0, 2, 1, 3)
            window_tokens = Kwin_q.shape[2]
            if window_tokens % 32 != 0:
                out_win_last = self._fallback_sliding_attention(q_win, Kwin_q, Vwin_q, scale, win)
            else:
                q_win_fp32 = q_win.float()
                Kwin_fp32 = Kwin_q.float()
                Vwin_fp32 = Vwin_q.float()
                out_win_last = casual_attention.apply(q_win_fp32, Kwin_fp32, Vwin_fp32, scale, win)
                out_win_last = out_win_last.to(q_win.dtype)
                if not torch.isfinite(out_win_last).all():
                    out_win_last = self._fallback_sliding_attention(
                        q_win,
                        Kwin_q,
                        Vwin_q,
                        scale,
                        win,
                    )
            out_win_heads = out_win_last.view(B, self.n_q, 1, self.d_head).view(B, G, Hg, 1, self.d_head)
            out_win_heads = torch.nan_to_num(out_win_heads)

            # Combine per head, apply gates of last position
            g_cmp_last = g_cmp[:, :, -1].view(B, G, Hg)
            g_slc_last = g_slc[:, :, -1].view(B, G, Hg)
            g_win_last = g_win[:, :, -1].view(B, G, Hg)
            out_heads = g_cmp_last[..., None, None] * out_cmp_heads + g_slc_last[..., None, None] * out_slc_heads + g_win_last[..., None, None] * out_win_heads
            out_heads = out_heads.reshape(B, self.n_q, self.d_head)  # merge groups
            out = out_heads.view(B, 1, self.n_q * self.d_head)
            if return_components:
                return out, (out_cmp_heads, out_slc_heads, out_win_heads)
            return out

        # Training/prefill path: implement compressed-attention TopK + per-row selected attention + sliding window
        # 1) Project K/V for branches (keep KV heads for cmp/slc; expand only for sliding window)
        k_cmp_kv = self.W_k_cmp(x).view(B, T, self.n_kv, self.d_head)
        v_cmp_kv = self.W_v_cmp(x).view(B, T, self.n_kv, self.d_head)
        k_slc_kv = self.W_k_slc(x).view(B, T, self.n_kv, self.d_head)
        v_slc_kv = self.W_v_slc(x).view(B, T, self.n_kv, self.d_head)
        k_win_kv = self.W_k_win(x).view(B, T, self.n_kv, self.d_head)
        v_win_kv = self.W_v_win(x).view(B, T, self.n_kv, self.d_head)
        k_win = self._expand_kv_to_q(k_win_kv)
        v_win = self._expand_kv_to_q(v_win_kv)

        # 2) Build compressed block summaries with learned compressors and positional encodings (operate at KV-head granularity)
        Kcmp_kv, Vcmp_kv, block_starts, total_blocks = self._build_cmp_blocks(
            k_cmp_kv.contiguous(), v_cmp_kv.contiguous()
        )
        # Reshape to groups (G == n_kv)
        G = self.n_kv
        Hg = self.n_q // self.n_kv
        q_g = q.view(B, G, Hg, T, self.d_head)
        Kcmp_g = Kcmp_kv.permute(0, 2, 1, 3).contiguous()  # (B, G, nb, d)
        Vcmp_g = Vcmp_kv.permute(0, 2, 1, 3).contiguous()

        # 3) Compressed attention via kernel with Top-K streaming using block-causal row_max
        # Build per-row max-allowed key column index (in original token space) based on block_starts
        # For compressed attention over blocks, we allow up to block index where block_start <= t
        nb = Kcmp_g.shape[2]
        blk_starts_t = block_starts.to(device=device)
        # For each t, row_max_block = max { bi | blk_starts[bi] <= t }
        row_max_block = torch.bucketize(torch.arange(T, device=device), blk_starts_t, right=True) - 1
        row_max_block = row_max_block.clamp(min=0, max=nb - 1)
        # shape (B,G,T) for kernel
        row_max = row_max_block.view(1, 1, T).expand(B, G, T).contiguous()
        # Prepare Q/K/V for causal kernel: use group-level queries, and treat blocks as keys/values index dimension
        q_group = q_g.sum(dim=2).contiguous()  # (B,G,T,d)
        Qk = q_group.view(B, G, T, self.d_head)
        Kk = Kcmp_g  # (B,G,nb,d)
        Vk = Vcmp_g
        # Reformat for kernel call: kernel expects (B, Heads, T, d) for Q and (B, Heads, d, N) for K, (B, Heads, N, d) for V
        Q_kernel = Qk  # (B,G,T,d)
        K_kernel = Kk.contiguous()  # (B,G,nb,d)
        V_kernel = Vk  # (B,G,nb,d)
        blocks = torch.arange(nb, device=device, dtype=torch.int32)
        block_mask = blocks.view(1, 1, 1, nb) <= row_max.unsqueeze(-1)
        logits_cmp = torch.matmul(Q_kernel.float(), K_kernel.float().transpose(-2, -1)) * scale
        logits_cmp = logits_cmp.masked_fill(~block_mask, float('-inf'))
        probs_cmp = torch.softmax(logits_cmp, dim=-1)
        out_cmp_group = torch.matmul(probs_cmp, V_kernel.float()).to(Q_kernel.dtype)
        out_cmp_heads = out_cmp_group.unsqueeze(2).expand(B, G, Hg, T, self.d_head)
        out_cmp_heads = torch.nan_to_num(out_cmp_heads)

        # 4) Compute per-row TopK block indices using Triton topk kernel (grouped queries vs block summaries)
        t_idx = torch.arange(T, device=device, dtype=torch.int32)
        row_max_block = torch.bucketize(t_idx, blk_starts_t, right=True) - 1
        row_max_block = row_max_block.clamp(min=0, max=nb - 1)
        row_max = row_max_block.view(1, 1, T).expand(B, G, T).contiguous()
        topk = min(self.slc_top_n, nb)
        top_idx = topk_indices(q_group, Kcmp_g, scale, topk, row_max)
        top_idx = torch.minimum(top_idx, row_max.unsqueeze(-1))
        forced_sink = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.long).expand(B, G, T, 1)
        last_two = torch.stack([
            torch.full((B, G, T), max(0, nb - 2), device=device, dtype=torch.long),
            torch.full((B, G, T), max(0, nb - 1), device=device, dtype=torch.long)
        ], dim=-1)
        # Build per-row unique block indices and counts; call indices-based kernel
        sel_blocks_all = torch.cat([forced_sink, last_two, top_idx], dim=-1)  # (B,G,T,topk+3)
        blk_mask_bgt = torch.zeros(B, G, T, nb, dtype=torch.bool, device=device)
        blk_mask_bgt.scatter_(dim=3, index=sel_blocks_all, src=torch.ones_like(sel_blocks_all, dtype=torch.bool))
        block_count = blk_mask_bgt.sum(dim=-1).to(torch.int32)  # (B,G,T)
        Kmax = min(nb, sel_blocks_all.shape[-1])
        idx_range = torch.arange(nb, device=device, dtype=torch.int32)
        idx_exp = idx_range.view(1, 1, 1, nb).expand(B, G, T, nb)
        masked = torch.where(blk_mask_bgt, idx_exp, torch.full_like(idx_exp, nb))
        sorted_idx, _ = torch.sort(masked, dim=-1)
        block_idx = sorted_idx[..., :Kmax]
        block_idx = torch.where(block_idx == nb, torch.zeros_like(block_idx), block_idx)
        block_count = torch.minimum(block_count, torch.tensor(Kmax, device=device, dtype=torch.int32))
        # prepare inputs and call kernel
        Kslc_full = k_slc_kv.permute(0, 2, 1, 3)  # (B,G,T,d)
        Vslc_full = v_slc_kv.permute(0, 2, 1, 3)
        out_slc_group = select_attention(q_g.mean(dim=2), Kslc_full, Vslc_full, scale, block_idx, block_count, self.cmp_blk_size)  # (B,G,T,d)
        out_slc_heads = out_slc_group.unsqueeze(2).expand(B, G, Hg, T, self.d_head)
        out_slc_heads = torch.nan_to_num(out_slc_heads)

        # 6) Sliding window branch via causal kernel per head
        q_win_full = q.view(B, self.n_q, T, self.d_head)
        k_win_full = k_win.permute(0, 2, 1, 3)
        v_win_full = v_win.permute(0, 2, 1, 3)
        if T % 32 != 0:
            out_win = self._fallback_sliding_attention(q_win_full, k_win_full, v_win_full, scale, self.window_size)
        else:
            out_win = casual_attention.apply(
                q_win_full.float(),
                k_win_full.float(),
                v_win_full.float(),
                scale,
                self.window_size,
            ).to(q_win_full.dtype)
        if not torch.isfinite(out_win).all():
            out_win = self._fallback_sliding_attention(
                q_win_full,
                k_win_full,
                v_win_full,
                scale,
                self.window_size,
            )
        out_win_heads = out_win.view(B, self.n_q, T, self.d_head).view(B, G, Hg, T, self.d_head)
        out_win_heads = torch.nan_to_num(out_win_heads)

        # Combine with gates
        g_cmp_h = g_cmp.view(B, G, Hg, T)
        g_slc_h = g_slc.view(B, G, Hg, T)
        g_win_h = g_win.view(B, G, Hg, T)
        out = g_cmp_h[..., None] * out_cmp_heads + g_slc_h[..., None] * out_slc_heads + g_win_h[..., None] * out_win_heads
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, T, self.n_q * self.d_head)
        # prime caches during prefill
        if not is_decoding:
            self.prefill(x, q)
        if return_components:
            return out, (out_cmp_heads, out_slc_heads, out_win_heads)
        return out
