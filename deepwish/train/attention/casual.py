from .casual_fwd import attention_forward
from .casual_bwd import _attn_bwd
import torch
import triton
import triton.language as tl

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, warp_specialize=True, USE_TMA=True): # q,k,v fp16 -> o fp16
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        attention_forward[grid](
            q, k, v, sm_scale, M, o,  \
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  \
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  \
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  \
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  \
            q.shape[0], q.shape[1], q.shape[2],  \
            BLOCK_M=64,  \
            BLOCK_N=32,  \
            HEAD_DIM=HEAD_DIM_K)
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

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
            HEAD_DIM=ctx.HEAD_DIM)
        return dq, dk, dv, None, None, None, None

attention = _attention.apply