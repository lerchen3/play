import torch
import triton

from train.attention.casual_fwd import attention_forward
from train.attention.casual_bwd import _attn_bwd


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, window_size: int = 0, row_max: torch.Tensor | None = None):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        has_row_max = row_max is not None
        if has_row_max:
            assert row_max.shape == q.shape[:3], "row_max must align with (B, H, T)"
            row_max_tensor = row_max.to(device=q.device, dtype=torch.int32).contiguous()
        else:
            row_max_tensor = torch.empty(1, device=q.device, dtype=torch.int32)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        attention_forward[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            row_max_tensor,
            row_max_tensor.stride(0) if has_row_max else 0,
            row_max_tensor.stride(1) if has_row_max else 0,
            row_max_tensor.stride(2) if has_row_max else 0,
            q.shape[0], q.shape[1], q.shape[2], window_size,
            BLOCK_M=64,
            BLOCK_N=32,
            HEAD_DIM=HEAD_DIM_K,
            HAS_ROW_MAX=has_row_max,
        )
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.window_size = window_size
        ctx.has_row_max = has_row_max
        ctx.row_max = row_max_tensor if has_row_max else None
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        if not do.is_contiguous():
            do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        assert N_CTX % BLOCK_N1 == 0
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        row_max_tensor = ctx.row_max if ctx.has_row_max else torch.empty(1, device=q.device, dtype=torch.int32)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, o, do, dq, dk, dv,
            M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, WINDOW_SIZE=ctx.window_size,
            HEAD_DIM=ctx.HEAD_DIM,
            HAS_ROW_MAX=ctx.has_row_max,
            RowMax=row_max_tensor,
            stride_rmz=(row_max_tensor.stride(0) if ctx.has_row_max else 0),
            stride_rmh=(row_max_tensor.stride(1) if ctx.has_row_max else 0),
            stride_rmm=(row_max_tensor.stride(2) if ctx.has_row_max else 0),
        )
        return dq, dk, dv, None, None, None


attention = _attention.apply
