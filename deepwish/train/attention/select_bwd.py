import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=['HEAD_DIM', 'BLOCK_T', 'BLOCK_N']
)
@triton.jit
def _select_bwd(Q, K, V, sm_scale,  \
               O, DO,  \
               DQ, DK, DV,  \
               M,  \
               stride_qz, stride_qg, stride_qt, stride_qd,  \
               stride_kz, stride_kg, stride_kn, stride_kd,  \
               stride_vz, stride_vg, stride_vk, stride_vd,  \
               H_PER_GROUP, N_SEL,  \
               BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # This is a simplified backward kernel assuming reduction across selected tokens similar to FA2 bwd
    # For brevity, not fully optimized; mirrors structure from casual_bwd with selected dimension.
    LN2: tl.constexpr = 0.6931471824645956
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers per batch-group
    Q += pid_bg * stride_qg + offs_t[None, :] * stride_qt + offs_k[:, None] * stride_qd
    K += pid_bg * stride_kg + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
    V += pid_bg * stride_vg + offs_n[None, :] * stride_vk + offs_k[:, None] * stride_vd
    O += pid_bg * stride_qg + offs_t[None, :] * stride_qt + offs_k[:, None] * stride_qd
    DO += pid_bg * stride_qg + offs_t[None, :] * stride_qt + offs_k[:, None] * stride_qd

    dq = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    # Load tiles
    q = tl.load(Q)
    o = tl.load(O)
    do = tl.load(DO)
    # Stabilized softmax backward using stored m and recomputed exp2
    m = tl.load(M + pid_bg * 0 + offs_t)  # expects per-(B,G,T) row maxima; caller should supply proper M
    # iterate selected tokens
    for i in range(0, N_SEL, BLOCK_N):
        k = tl.load(K)
        v = tl.load(V)
        qk = tl.dot(q, k)
        p = tl.math.exp2(qk - m[:, None])
        p32 = p.to(tl.float32)
        dv += tl.dot(p32, do)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        ds = p * (dp - delta[:, None])
        ds = ds.to(tl.float32)
        dk += tl.dot(tl.trans(ds), q)
        dq += tl.dot(ds, k)
        K += BLOCK_N * stride_kn
        V += BLOCK_N * stride_vk

    dq *= LN2
    tl.store(DQ + pid_bg * stride_qg + offs_t[None, :] * stride_qt + offs_k[:, None] * stride_qd, dq)
    tl.store(DK + pid_bg * stride_kg + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd, dk)
    tl.store(DV + pid_bg * stride_vg + offs_n[None, :] * stride_vk + offs_k[:, None] * stride_vd, dv)


