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



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=['HEAD_DIM', 'BLOCK_N']
)
@triton.jit
def select_attention_backward(
    Q, K_full, V_full, sm_scale,
    O, DO,
    DQ, DK_full, DV_full, M,
    stride_qz, stride_qg, stride_qt, stride_qd,
    stride_kz, stride_kg, stride_kn, stride_kd,
    stride_vz, stride_vg, stride_vn, stride_vd,
    stride_oz, stride_og, stride_ot, stride_od,
    stride_dqz, stride_dqg, stride_dqt, stride_dqd,
    stride_dkz, stride_dkg, stride_dkn, stride_dkd,
    stride_dvz, stride_dvg, stride_dvn, stride_dvd,
    stride_mz, stride_mg, stride_mt,
    BlockIdx, stride_bz, stride_bg, stride_bt, stride_bk,
    BlockCount, stride_cz, stride_cg, stride_ct,
    BATCH_SIZE, NUM_GROUPS, T_CTX, SEQ_LEN, CMP_BLK_SIZE, KMAX,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645956
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)

    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS
    t = pid_t

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # row pointers
    Q_ptr = Q + b * stride_qz + g * stride_qg + t * stride_qt + offs_d * stride_qd
    O_ptr = O + b * stride_oz + g * stride_og + t * stride_ot + offs_d * stride_od
    DO_ptr = DO + b * stride_oz + g * stride_og + t * stride_ot + offs_d * stride_od
    DQ_ptr = DQ + b * stride_dqz + g * stride_dqg + t * stride_dqt + offs_d * stride_dqd
    m = tl.load(M + b * stride_mz + g * stride_mg + t * stride_mt)

    q = tl.load(Q_ptr)
    o = tl.load(O_ptr)
    do = tl.load(DO_ptr)
    dq = tl.zeros([HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    cnt = tl.load(BlockCount + b * stride_cz + g * stride_cg + t * stride_ct)
    for bi in range(0, KMAX):
        if bi >= cnt:
            break
        blk = tl.load(BlockIdx + b * stride_bz + g * stride_bg + t * stride_bt + bi * stride_bk)
        blk_start = blk * CMP_BLK_SIZE
        blk_end = tl.minimum(blk_start + CMP_BLK_SIZE, SEQ_LEN)
        start_n = blk_start
        while start_n < blk_end:
            cols = start_n + offs_n
            valid = cols < blk_end
            causal = cols <= t
            col_mask = valid & causal
            # load tiles
            k_ptr = K_full + b * stride_kz + g * stride_kg + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptr = V_full + b * stride_vz + g * stride_vg + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptr, mask=col_mask[None, :], other=0.0)
            v = tl.load(v_ptr, mask=col_mask[:, None], other=0.0)
            # logits and p
            qk = tl.dot(q, k)
            qk = tl.where(col_mask, qk, -1.0e6)
            qk = qk * qk_scale
            p = tl.math.exp2(qk - m)
            p32 = p.to(tl.float32)
            # dv contribution
            dv_tile = p32[:, None] * do[None, :]
            # dp
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            delta = tl.sum(o * do)
            ds = p32 * (dp - delta)
            # dk contribution
            dk_tile = ds[:, None] * q[None, :]
            # accumulate into dq
            dq += tl.dot(ds, k)
            # atomic add into DK_full and DV_full
            for c in range(BLOCK_N):
                if col_mask[c]:
                    col = cols[c]
                    dv_dst = DV_full + b * stride_dvz + g * stride_dvg + col * stride_dvn + offs_d * stride_dvd
                    dk_dst = DK_full + b * stride_dkz + g * stride_dkg + col * stride_dkn + offs_d * stride_dkd
                    tl.atomic_add(dv_dst, dv_tile[c, :])
                    tl.atomic_add(dk_dst, dk_tile[c, :])
            start_n += BLOCK_N

    # finalize dq scaling
    dq *= LN2
    tl.store(DQ_ptr, dq)