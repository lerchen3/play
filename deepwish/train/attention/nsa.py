import torch
import torch.nn as nn
from .casual import _attention as casual_attention
from .select import select_attention


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

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, q: torch.Tensor):
        """Build NSA caches from full prompt x (B,S,D) and q (B,n_q,S,d)."""
        self.reset_cache()
        B, T, _ = x.shape
        device = x.device
        # Project and cache per-branch K/V
        k_cmp = self.W_k_cmp(x).view(B, T, self.n_kv, self.d_head)
        v_cmp = self.W_v_cmp(x).view(B, T, self.n_kv, self.d_head)
        k_slc = self.W_k_slc(x).view(B, T, self.n_kv, self.d_head)
        v_slc = self.W_v_slc(x).view(B, T, self.n_kv, self.d_head)
        k_win = self.W_k_win(x).view(B, T, self.n_kv, self.d_head)
        v_win = self.W_v_win(x).view(B, T, self.n_kv, self.d_head)
        self._cache_cmp_k, self._cache_cmp_v = k_cmp, v_cmp
        self._cache_slc_k, self._cache_slc_v = k_slc, v_slc
        self._cache_win_k, self._cache_win_v = k_win, v_win
        # Build block summaries (KV heads)
        block_starts = list(range(0, max(1, T - self.cmp_blk_size + 1), self.cmp_stride))
        if 0 not in block_starts:
            block_starts = [0] + block_starts
        blk_k, blk_v = [], []
        for s in block_starts:
            e = min(s + self.cmp_blk_size, T)
            kblk = k_cmp[:, s:e]
            vblk = v_cmp[:, s:e]
            # pad to full block
            pad = self.cmp_blk_size - (e - s)
            if pad > 0:
                kblk = torch.cat([kblk, torch.zeros(B, pad, self.n_kv, self.d_head, device=device, dtype=kblk.dtype)], dim=1)
                vblk = torch.cat([vblk, torch.zeros_like(kblk)], dim=1)
            # add learned pos encoding
            kblk = kblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
            vblk = vblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
            # flatten and compress
            kflat = kblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
            vflat = vblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
            ksum = self.compress_k2(torch.nn.functional.gelu(self.compress_k1(kflat))).view(B, self.n_kv, self.d_head)
            vsum = self.compress_v2(torch.nn.functional.gelu(self.compress_v1(vflat))).view(B, self.n_kv, self.d_head)
            blk_k.append(ksum)
            blk_v.append(vsum)
        self._cmp_block_summary_k = torch.stack(blk_k, dim=1)  # (B, nb, n_kv, d)
        self._cmp_block_summary_v = torch.stack(blk_v, dim=1)
        self._block_starts = block_starts
        self._cached_len = T

    def _expand_kv_to_q(self, t):
        # (B, L, n_kv, d) -> (B, L, n_q, d)
        if self.n_q == self.n_kv:
            return t
        n_rep = self.n_q // self.n_kv
        B, L = t.shape[:2]
        return t.unsqueeze(2).expand(B, L, n_rep, self.n_kv, self.d_head).reshape(B, L, self.n_q, self.d_head)

    def forward(self, x, q, is_decoding: bool = False):
        B, T, D = x.shape
        device = x.device
        scale = (self.d_head) ** -0.5

        gates = torch.sigmoid(self.gate(x)).view(B, T, self.n_q, 3).permute(0, 2, 1, 3)
        g_cmp, g_slc, g_win = gates[..., 0], gates[..., 1], gates[..., 2]  # (B, n_q, T)

        if is_decoding and T == 1:
            # Single-token decode: update caches and compute outputs for last token only using torch
            t_new = self._cached_len
            # Update per-branch K/V caches with new token projections
            k_cmp_new = self.W_k_cmp(x).view(B, 1, self.n_kv, self.d_head)
            v_cmp_new = self.W_v_cmp(x).view(B, 1, self.n_kv, self.d_head)
            k_slc_new = self.W_k_slc(x).view(B, 1, self.n_kv, self.d_head)
            v_slc_new = self.W_v_slc(x).view(B, 1, self.n_kv, self.d_head)
            k_win_new = self.W_k_win(x).view(B, 1, self.n_kv, self.d_head)
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
            # Incremental update of block summaries using learned compressor (only affected/new blocks)
            block_starts_new = list(range(0, max(1, Ttot - self.cmp_blk_size + 1), self.cmp_stride))
            if 0 not in block_starts_new:
                block_starts_new = [0] + block_starts_new
            nb_new = len(block_starts_new)
            if getattr(self, "_cmp_block_summary_k", None) is None:
                # initialize from scratch with learned compressors
                blk_k, blk_v = [], []
                for s in block_starts_new:
                    e = min(s + self.cmp_blk_size, Ttot)
                    kblk = self._cache_cmp_k[:, s:e]
                    vblk = self._cache_cmp_v[:, s:e]
                    pad = self.cmp_blk_size - (e - s)
                    if pad > 0:
                        kblk = torch.cat([kblk, torch.zeros(B, pad, self.n_kv, self.d_head, device=device, dtype=kblk.dtype)], dim=1)
                        vblk = torch.cat([vblk, torch.zeros_like(kblk)], dim=1)
                    kblk = kblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
                    vblk = vblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
                    kflat = kblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
                    vflat = vblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
                    ksum = self.compress_k2(torch.nn.functional.gelu(self.compress_k1(kflat))).view(B, self.n_kv, self.d_head)
                    vsum = self.compress_v2(torch.nn.functional.gelu(self.compress_v1(vflat))).view(B, self.n_kv, self.d_head)
                    blk_k.append(ksum)
                    blk_v.append(vsum)
                self._cmp_block_summary_k = torch.stack(blk_k, dim=1)
                self._cmp_block_summary_v = torch.stack(blk_v, dim=1)
                self._block_starts = block_starts_new
            else:
                old_starts = self._block_starts
                start_to_old = {s: i for i, s in enumerate(old_starts)}
                new_k = torch.empty(B, nb_new, self.n_kv, self.d_head, device=device, dtype=self._cmp_block_summary_k.dtype)
                new_v = torch.empty_like(new_k)
                # blocks that include the new token index are affected
                affected = {s for s in block_starts_new if (Ttot - 1) >= s and (Ttot - 1) < s + self.cmp_blk_size}
                for new_idx, s in enumerate(block_starts_new):
                    if (s in start_to_old) and (s not in affected):
                        old_idx = start_to_old[s]
                        new_k[:, new_idx] = self._cmp_block_summary_k[:, old_idx]
                        new_v[:, new_idx] = self._cmp_block_summary_v[:, old_idx]
                    else:
                        e = min(s + self.cmp_blk_size, Ttot)
                        kblk = self._cache_cmp_k[:, s:e]
                        vblk = self._cache_cmp_v[:, s:e]
                        pad = self.cmp_blk_size - (e - s)
                        if pad > 0:
                            kblk = torch.cat([kblk, torch.zeros(B, pad, self.n_kv, self.d_head, device=device, dtype=kblk.dtype)], dim=1)
                            vblk = torch.cat([vblk, torch.zeros_like(kblk)], dim=1)
                        kblk = kblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
                        vblk = vblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
                        kflat = kblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
                        vflat = vblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
                        ksum = self.compress_k2(torch.nn.functional.gelu(self.compress_k1(kflat))).view(B, self.n_kv, self.d_head)
                        vsum = self.compress_v2(torch.nn.functional.gelu(self.compress_v1(vflat))).view(B, self.n_kv, self.d_head)
                        new_k[:, new_idx] = ksum
                        new_v[:, new_idx] = vsum
                self._cmp_block_summary_k = new_k
                self._cmp_block_summary_v = new_v
                self._block_starts = block_starts_new
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
            out_cmp_heads = torch.einsum('bghn,bgnd->bghd', prob_cmp, Vcmp_g)  # (B,G,Hg,d)

            # Selected branch: build indices and use indices-based kernel
            nb = len(self._block_starts)
            sel_blocks_all = torch.cat([
                torch.zeros(B, G, 1, dtype=torch.long, device=device),
                torch.full((B, G, 1), max(0, nb - 2), dtype=torch.long, device=device),
                torch.full((B, G, 1), max(0, nb - 1), dtype=torch.long, device=device),
                top_idx
            ], dim=-1)  # (B,G,3+topn)
            blk_mask_bg = torch.zeros(B, G, nb, dtype=torch.int8, device=device)
            blk_mask_bg.scatter_(dim=2, index=sel_blocks_all, src=torch.ones_like(sel_blocks_all, dtype=torch.int8))
            block_count = blk_mask_bg.sum(dim=-1).to(torch.int32)  # (B,G)
            Kmax = min(nb, sel_blocks_all.shape[-1])
            # get up to Kmax indices per (B,G) where mask is 1
            idx = torch.topk(blk_mask_bg.to(torch.int32), k=Kmax, dim=-1).indices  # (B,G,Kmax)
            block_idx = idx.unsqueeze(2)  # (B,G,1,Kmax)
            block_count = block_count.unsqueeze(2)  # (B,G,1)
            # Prepare inputs for kernel
            q_slc = q_last.mean(dim=2).unsqueeze(2)  # (B,G,1,d)
            Kslc_full = self._cache_slc_k.permute(0, 2, 1, 3)  # (B,G,Ttot,d)
            Vslc_full = self._cache_slc_v.permute(0, 2, 1, 3)
            out_slc_group = select_attention(q_slc, Kslc_full, Vslc_full, scale, block_idx, block_count, self.cmp_blk_size)  # (B,G,1,d)
            out_slc_heads = out_slc_group.unsqueeze(2).expand(B, G, Hg, 1, self.d_head)

            # Sliding window branch via kernel for last position
            win = self.window_size or Ttot
            start = max(0, Ttot - win)
            q_win = q_last.view(B, G * Hg, self.d_head)  # (B, n_q, d) collapsed by groups
            q_win = q_win.view(B, self.n_q, self.d_head).unsqueeze(2)  # (B, n_q, 1, d)
            Kwin = self._cache_win_k[:, start:Ttot].permute(0, 2, 1, 3)  # (B, G, Lw, d)
            Vwin = self._cache_win_v[:, start:Ttot].permute(0, 2, 1, 3)
            Kwin_q = self._expand_kv_to_q(Kwin.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # (B, n_q, Lw, d)
            Vwin_q = self._expand_kv_to_q(Vwin.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            out_win_last = casual_attention.apply(q_win, Kwin_q.permute(0, 1, 3, 2), Vwin_q, scale, win)  # (B, n_q, 1, d)
            out_win_heads = out_win_last.view(B, self.n_q, 1, self.d_head).view(B, G, Hg, 1, self.d_head)

            # Combine per head, apply gates of last position
            g_cmp_last = g_cmp[:, :, -1].view(B, G, Hg)
            g_slc_last = g_slc[:, :, -1].view(B, G, Hg)
            g_win_last = g_win[:, :, -1].view(B, G, Hg)
            out_heads = g_cmp_last[..., None] * out_cmp_heads + g_slc_last[..., None] * out_slc_heads + g_win_last[..., None] * out_win_heads
            out_heads = out_heads.view(B, self.n_q, self.d_head)  # merge groups
            out = out_heads.view(B, 1, self.n_q * self.d_head)
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
        block_starts = list(range(0, max(1, T - self.cmp_blk_size + 1), self.cmp_stride))
        if 0 not in block_starts:
            block_starts = [0] + block_starts
        blk_k, blk_v = [], []
        for s in block_starts:
            e = min(s + self.cmp_blk_size, T)
            kblk = k_cmp_kv[:, s:e]  # (B, L, n_kv, d)
            vblk = v_cmp_kv[:, s:e]
            pad = self.cmp_blk_size - (e - s)
            if pad > 0:
                kblk = torch.cat(
                    [kblk, torch.zeros(B, pad, self.n_kv, self.d_head, device=device, dtype=kblk.dtype)], dim=1
                )
                vblk = torch.cat([vblk, torch.zeros_like(kblk)], dim=1)
            # add learned intra-block pos enc
            kblk = kblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
            vblk = vblk + self.block_pos.view(1, self.cmp_blk_size, 1, self.d_head)
            # flatten per KV head and compress via MLPs
            kflat = kblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
            vflat = vblk.permute(0, 2, 1, 3).contiguous().view(B * self.n_kv, -1)
            ksum = self.compress_k2(torch.nn.functional.gelu(self.compress_k1(kflat))).view(B, self.n_kv, self.d_head)
            vsum = self.compress_v2(torch.nn.functional.gelu(self.compress_v1(vflat))).view(B, self.n_kv, self.d_head)
            blk_k.append(ksum)
            blk_v.append(vsum)
        # (B, nb, n_kv, d)
        Kcmp_kv = torch.stack(blk_k, dim=1)
        Vcmp_kv = torch.stack(blk_v, dim=1)
        # Reshape to groups (G == n_kv)
        G = self.n_kv
        Hg = self.n_q // self.n_kv
        q_g = q.view(B, G, Hg, T, self.d_head)
        Kcmp_g = Kcmp_kv.permute(0, 2, 1, 3)  # (B, G, nb, d)
        Vcmp_g = Vcmp_kv.permute(0, 2, 1, 3)

        # 3) Compressed attention via kernel with Top-K streaming using block-causal row_max
        # Build per-row max-allowed key column index (in original token space) based on block_starts
        # For compressed attention over blocks, we allow up to block index where block_start <= t
        nb = Kcmp_g.shape[2]
        blk_starts_t = torch.tensor(block_starts, device=device, dtype=torch.int32)
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
        K_kernel = Kk.permute(0,1,3,2).contiguous()  # (B,G,d,nb)
        V_kernel = Vk  # (B,G,nb,d)
        # Call kernel with block-causal row_max and top_k
        out_cmp_group = casual_attention.apply(
            Q_kernel, K_kernel, V_kernel, scale, 0, self.slc_top_n, row_max=row_max
        )  # (B,G,T,d)
        out_cmp_heads = out_cmp_group.unsqueeze(2).expand(B, G, Hg, T, self.d_head)

        # 4) Extract per-row TopK block indices from kernel TopIdx buffers is not exposed; recompute deterministically using same masks
        # For now, recompute indices via matmul over blocks to build the row mask for select kernel
        attn_blocks = torch.einsum('bgtd,bgnd->bgtn', q_group, Kcmp_g) * scale
        t_idx = torch.arange(T, device=device)
        blk_idx = torch.tensor(block_starts, device=device)
        causal_mask = blk_idx.view(1, 1, 1, nb) <= t_idx.view(1, 1, T, 1)
        attn_blocks = attn_blocks.masked_fill(~causal_mask, float('-inf'))
        topk = min(self.slc_top_n, nb)
        top_idx = torch.topk(attn_blocks, k=topk, dim=-1).indices  # (B,G,T,topk)
        forced_sink = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.long).expand(B, G, T, 1)
        last_two = torch.stack([
            torch.full((B, G, T), max(0, nb - 2), device=device, dtype=torch.long),
            torch.full((B, G, T), max(0, nb - 1), device=device, dtype=torch.long)
        ], dim=-1)
        # Build per-row unique block indices and counts; call indices-based kernel
        sel_blocks_all = torch.cat([forced_sink, last_two, top_idx], dim=-1)  # (B,G,T,topk+3)
        blk_mask_bgt = torch.zeros(B, G, T, nb, dtype=torch.int8, device=device)
        blk_mask_bgt.scatter_(dim=3, index=sel_blocks_all, src=torch.ones_like(sel_blocks_all, dtype=torch.int8))
        block_count = blk_mask_bgt.sum(dim=-1).to(torch.int32)  # (B,G,T)
        Kmax = min(nb, sel_blocks_all.shape[-1])
        # pick up to Kmax indices where mask==1 for each row
        idx = torch.topk(blk_mask_bgt.to(torch.int32), k=Kmax, dim=-1).indices  # (B,G,T,Kmax)
        block_idx = idx
        # prepare inputs and call kernel
        Kslc_full = k_slc_kv.permute(0, 2, 1, 3)  # (B,G,T,d)
        Vslc_full = v_slc_kv.permute(0, 2, 1, 3)
        out_slc_group = select_attention(q_g.mean(dim=2), Kslc_full, Vslc_full, scale, block_idx, block_count, self.cmp_blk_size)  # (B,G,T,d)
        out_slc_heads = out_slc_group.unsqueeze(2).expand(B, G, Hg, T, self.d_head)

        # 6) Sliding window branch via causal kernel per head
        out_win = casual_attention.apply(
            q.view(B, self.n_q, T, self.d_head),
            k_win.permute(0, 2, 1, 3),
            v_win.permute(0, 2, 1, 3),
            scale,
            self.window_size,
        )
        out_win_heads = out_win.view(B, self.n_q, T, self.d_head).view(B, G, Hg, T, self.d_head)

        # Combine with gates
        g_cmp_h = g_cmp.view(B, G, Hg, T)
        g_slc_h = g_slc.view(B, G, Hg, T)
        g_win_h = g_win.view(B, G, Hg, T)
        out = g_cmp_h[..., None] * out_cmp_heads + g_slc_h[..., None] * out_slc_heads + g_win_h[..., None] * out_win_heads
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, T, self.n_q * self.d_head)
        # prime caches during prefill
        if not is_decoding:
            self.prefill(x, q)
        return out


