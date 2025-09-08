import torch
import triton
from .select_fwd import select_attention_forward
from .select_bwd import _select_bwd


class _select_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, row_mask: torch.Tensor = None):
        # q: (B, G, T, D), k/v: (B, G, N_sel, D)
        # row_mask: optional (B, G, T, N_sel) with 1 for allowed positions
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(q.shape[2], 64), q.shape[0] * q.shape[1])
        if row_mask is None:
            dummy = torch.empty(1, device=q.device, dtype=torch.int32)
            select_attention_forward[grid](
                q, k, v, sm_scale, o, M,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                M.stride(0), M.stride(1), M.stride(2),
                q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                HEADS_PER_GROUP=1, HEAD_DIM=q.shape[-1], BLOCK_T=64, BLOCK_N=32,
                RowMask=dummy, stride_rmz=0, stride_rmg=0, stride_rmm=0, stride_rmn=0, USE_MASK=False,
            )
        else:
            assert row_mask.shape[:4] == (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            select_attention_forward[grid](
                q, k, v, sm_scale, o, M,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                M.stride(0), M.stride(1), M.stride(2),
                q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                HEADS_PER_GROUP=1, HEAD_DIM=q.shape[-1], BLOCK_T=64, BLOCK_N=32,
                RowMask=row_mask, stride_rmz=row_mask.stride(0), stride_rmg=row_mask.stride(1), stride_rmm=row_mask.stride(2), stride_rmn=row_mask.stride(3), USE_MASK=True,
            )
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        grid = (triton.cdiv(q.shape[2], 64), q.shape[0] * q.shape[1])
        _select_bwd[grid](
            q, k, v, ctx.sm_scale, o, do, dq, dk, dv, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            1, k.shape[2], BLOCK_T=64, BLOCK_N=32, HEAD_DIM=q.shape[-1],
        )
        return dq, dk, dv, None


def select_attention(q, k, v, sm_scale, row_mask: torch.Tensor = None):
    return _select_attention.apply(q, k, v, sm_scale, row_mask)


