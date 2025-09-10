import torch
import triton
from .select_fwd import select_attention_forward
from .select_bwd import _select_bwd


class _select_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_full, v_full, sm_scale, block_idx: torch.Tensor, block_count: torch.Tensor, cmp_blk_size: int):
        # q: (B, G, T, D), k_full/v_full: (B, G, N_ctx, D)
        # block_idx: (B, G, T, Kmax) int32 of block indices; block_count: (B, G, T) int32 number of valid blocks per row
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        B, G, T, D = q.shape
        N_ctx = k_full.shape[2]
        Kmax = block_idx.shape[-1]
        grid = (T, B * G)
        select_attention_forward[grid](
            q, k_full, v_full, sm_scale, o, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            block_idx, block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
            block_count, block_count.stride(0), block_count.stride(1), block_count.stride(2),
            B, G, T, N_ctx, cmp_blk_size, Kmax,
            HEAD_DIM=D,
            BLOCK_N=32,
        )
        ctx.save_for_backward(q, k_full, v_full, o, M, block_idx, block_count)
        ctx.sm_scale = sm_scale
        ctx.cmp_blk_size = int(cmp_blk_size)
        return o

    @staticmethod
    def backward(ctx, do):
        from .select_bwd import select_attention_backward
        q, k_full, v_full, o, M, block_idx, block_count = ctx.saved_tensors
        B, G, T, D = q.shape
        N_ctx = k_full.shape[2]
        Kmax = block_idx.shape[-1]
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k_full)
        dv = torch.zeros_like(v_full)
        grid = (T, B * G)
        select_attention_backward[grid](
            q, k_full, v_full, ctx.sm_scale,
            o, do,
            dq, dk, dv, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            block_idx, block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
            block_count, block_count.stride(0), block_count.stride(1), block_count.stride(2),
            B, G, T, N_ctx, ctx.cmp_blk_size, Kmax,
            HEAD_DIM=D,
            BLOCK_N=32,
        )
        return dq, dk, dv, None, None, None, None


def select_attention(q, k_full, v_full, sm_scale, block_idx: torch.Tensor, block_count: torch.Tensor, cmp_blk_size: int):
    return _select_attention.apply(q, k_full, v_full, sm_scale, block_idx, block_count, cmp_blk_size)


