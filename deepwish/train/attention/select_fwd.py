import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=['HEAD_DIM', 'BLOCK_N']
)
@triton.jit
def select_attention_forward(
    Q, K_full, V_full,  # tensors: Q (B,G,T,D), K_full/V_full (B,G,SEQ_LEN,D)
    sm_scale, O, M,  # output and per-row max
    stride_qz, stride_qg, stride_qt, stride_qd,  # B, G, T, D for queries
    stride_kz, stride_kg, stride_kn, stride_kd,  # B, G, N_ctx, D for keys
    stride_vz, stride_vg, stride_vn, stride_vd,  # B, G, N_ctx, D for values
    stride_oz, stride_og, stride_ot, stride_od,  # B, G, T, D for output
    stride_mz, stride_mg, stride_mt,             # B, G, T for M
    BlockIdx, stride_bz, stride_bg, stride_bt, stride_bk,  # (B,G,T,Kmax)
    BlockCount, stride_cz, stride_cg, stride_ct,          # (B,G,T)
    BATCH_SIZE, NUM_GROUPS, T_CTX, SEQ_LEN, CMP_BLK_SIZE, KMAX,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program per (timestep, batch-group)
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)

    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS

    t = pid_t  # single row
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Pointers
    Q_ptr = Q + b * stride_qz + g * stride_qg + t * stride_qt + offs_d * stride_qd
    O_ptr = O + b * stride_oz + g * stride_og + t * stride_ot + offs_d * stride_od

    # init accumulators for numerically stable softmax
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_i = 1.0
    m_i = -float("inf")
    qk_scale = sm_scale * 1.44269504

    # Load q row
    q = tl.load(Q_ptr)

    # Load count and iterate selected block indices
    cnt = tl.load(BlockCount + b * stride_cz + g * stride_cg + t * stride_ct)
    for bi in range(0, KMAX):
        valid_bi = bi < cnt
        blk = tl.load(BlockIdx + b * stride_bz + g * stride_bg + t * stride_bt + bi * stride_bk)
        # deduplicate repeated block indices within this row like the torch reference's set()
        dup_count = 0
        for pj in range(0, bi):
            prev_blk = tl.load(BlockIdx + b * stride_bz + g * stride_bg + t * stride_bt + pj * stride_bk)
            dup_count += (prev_blk == blk)
        process_block = valid_bi & (dup_count == 0)
        # compute absolute token range for this block
        blk_start = blk * CMP_BLK_SIZE
        blk_end = tl.minimum(blk_start + CMP_BLK_SIZE, SEQ_LEN)
        # iterate over this block in BLOCK_N tiles
        start_n = blk_start
        while start_n < blk_end:
            k_ptr = K_full + b * stride_kz + g * stride_kg + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptr = V_full + b * stride_vz + g * stride_vg + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
            # valid cols within context and causal for this row
            valid = ((start_n + offs_n) < blk_end) & process_block
            causal = (start_n + offs_n) <= t
            col_mask = valid & causal
            # load tiles with mask
            k = tl.load(k_ptr, mask=col_mask[None, :], other=0.0)
            qk = tl.sum(q[:, None] * k, axis=0)
            # set masked columns to -inf before softmax so they contribute zero prob
            qk = tl.where(col_mask, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk) * qk_scale)
            qk = qk * qk_scale - m_ij
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha
            v = tl.load(v_ptr, mask=col_mask[:, None], other=0.0)
            p = p.to(tl.float32)
            v = v.to(tl.float32)
            acc = acc + tl.sum(v * p[:, None], axis=0)
            m_i = m_ij
            start_n += BLOCK_N

    m_i = m_i + tl.math.log2(l_i)
    acc = acc / l_i
    tl.store(O_ptr, acc.to(O.type.element_ty))
    # store per-row max
    m_ptr = M + b * stride_mz + g * stride_mg + t * stride_mt
    tl.store(m_ptr, m_i)


