import triton
import triton.language as tl


@triton.jit
def _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, O, DO, M, stride_tok, stride_d, H, N_CTX, \
                   BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr, \
                   start_n, start_m, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    o_ptrs  =  O + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        o  = tl.load(o_ptrs)
        delta = tl.sum(o * do, axis=1)
        ppT = pT.to(tl.float32)
        dv += tl.dot(ppT, do)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - delta[None, :])
        dsT = dsT.to(tl.float32)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        o_ptrs  += step_m * stride_tok
    return dk, dv

@triton.jit
def _attn_bwd_dq(dq, q, K, V, O, do, m, stride_tok, stride_d, H, N_CTX, \
                 BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr, \
                 start_m, start_n, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    o   = tl.load(O + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    delta = tl.sum(o * do, axis=1)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n_current = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n_current[None, :])
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = ds.to(tl.float32)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq

@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  \
              O, DO,  \
              DQ, DK, DV,  \
              M,  
              stride_z, stride_h, stride_tok, stride_d,  \
              H, N_CTX,  \
              BLOCK_M1: tl.constexpr,  \
              BLOCK_N1: tl.constexpr,  \
              BLOCK_M2: tl.constexpr,  \
              BLOCK_N2: tl.constexpr,  \
              BLK_SLICE_FACTOR: tl.constexpr,  \
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645956
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    Q += adj
    K += adj
    V += adj
    O += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = start_n
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, O, DO, M, \
                            stride_tok, stride_d, H, N_CTX, \
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM, \
                            start_n, start_m, num_steps, MASK=True)
    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, O, DO, M, \
                            stride_tok, stride_d, H, N_CTX, \
                            BLOCK_M1, BLOCK_N1, HEAD_DIM, \
                            start_n, start_m, num_steps, MASK=False)
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V, O, do, m, \
                      stride_tok, stride_d, H, N_CTX, \
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM, \
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, MASK=True)
    end_n -= num_steps * MASK_BLOCK_N2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V, O, do, m, \
                      stride_tok, stride_d, H, N_CTX, \
                      BLOCK_M2, BLOCK_N2, HEAD_DIM, \
                      start_m, end_n - num_steps * BLOCK_N2, num_steps, MASK=False)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)
