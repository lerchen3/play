from train.attention.casual_fwd import attention_forward
from train.attention.casual_bwd import _attn_bwd
import torch
import triton
import triton.language as tl

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, window_size: int = 0, top_k: int = 0, row_max: torch.Tensor = None, warp_specialize=True, USE_TMA=True): # q,k,v fp16 -> o fp16
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        TopIdx = None
        TopVal = None
        if top_k and top_k > 0:
            TopVal = torch.full((q.shape[0], q.shape[1], q.shape[2], top_k), float('-inf'), device=q.device, dtype=torch.float32)
            TopIdx = torch.full((q.shape[0], q.shape[1], q.shape[2], top_k), -1, device=q.device, dtype=torch.int32)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        if row_max is None:
            dummy = torch.empty(1, device=q.device, dtype=torch.int32)
            attention_forward[grid](
                q, k, v, sm_scale, M, o, TopVal, TopIdx, top_k if top_k else 0, \
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  \
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  \
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  \
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  \
                q.shape[0], q.shape[1], q.shape[2],  \
                BLOCK_M=64,  \
                BLOCK_N=32,  \
                HEAD_DIM=HEAD_DIM_K,  \
                WINDOW_SIZE=window_size if window_size is not None else 0, \
                RowMax=dummy, stride_rmz=0, stride_rmh=0, stride_rmm=0, USE_BLOCK_CAUSAL=False)
        else:
            assert row_max.shape[:3] == (q.shape[0], q.shape[1], q.shape[2])
            attention_forward[grid](
                q, k, v, sm_scale, M, o, TopVal, TopIdx, top_k if top_k else 0, \
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  \
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  \
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  \
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  \
            q.shape[0], q.shape[1], q.shape[2],  \
            BLOCK_M=64,  \
            BLOCK_N=32,  \
            HEAD_DIM=HEAD_DIM_K,  \
            WINDOW_SIZE=window_size if window_size is not None else 0, \
            RowMax=row_max, stride_rmz=row_max.stride(0), stride_rmh=row_max.stride(1), stride_rmm=row_max.stride(2), USE_BLOCK_CAUSAL=True)
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.window_size = 0 if window_size is None else int(window_size)
        ctx.top_k = int(top_k) if top_k else 0
        ctx.top_idx = TopIdx
        return (o, TopIdx) if TopIdx is not None else o

    @staticmethod
    def backward(ctx, do): # do is fp16 as well here
        q, k, v, o, M = ctx.saved_tensors
        # ensure gradient input is contiguous
        if not do.is_contiguous():
            do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32 # dKdV Q-seqlength-block, dKdV KV-seqlength-block, dQ Q-seqlength-block, dQ KV-seqlength-block.
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 64
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, o, do, dq, dk, dv,  \
            M,  \
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  \
            N_HEAD, N_CTX,  \
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  \
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  \
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  \
            HEAD_DIM=ctx.HEAD_DIM,  \
            WINDOW_SIZE=ctx.window_size if hasattr(ctx, 'window_size') else 0)
        # If forward returned TopIdx, propagate gradient only for main output
        return dq, dk, dv, None, None, None, None
