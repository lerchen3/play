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
    MAX_TILES: tl.constexpr,
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
    l_i = tl.full([], 1.0, dtype=tl.float32)
    m_i = tl.full([], -float("inf"), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # Load q row
    q = tl.load(Q_ptr).to(tl.float32)

    # Load count and iterate selected block indices
    cnt = tl.load(BlockCount + b * stride_cz + g * stride_cg + t * stride_ct)
    max_tiles: tl.constexpr = MAX_TILES
    for bi in range(0, KMAX):
        block_active = bi < cnt
        blk = tl.load(
            BlockIdx + b * stride_bz + g * stride_bg + t * stride_bt + bi * stride_bk,
            mask=block_active,
            other=0,
        )
        duplicate = tl.full([], 0, dtype=tl.int32)
        for bj in range(0, bi):
            prev = tl.load(
                BlockIdx + b * stride_bz + g * stride_bg + t * stride_bt + bj * stride_bk,
                mask=block_active,
                other=0,
            )
            duplicate = tl.maximum(duplicate, (block_active & (blk == prev)).to(tl.int32))
        block_active = block_active & (duplicate == 0)
        blk_start = blk * CMP_BLK_SIZE
        blk_end = tl.minimum(blk_start + CMP_BLK_SIZE, SEQ_LEN)
        for tile in range(0, max_tiles):
            start_n = blk_start + tile * BLOCK_N
            tile_active = block_active & (start_n < blk_end)
            cols = start_n + offs_n
            valid = (cols < blk_end) & tile_active
            causal = (cols <= t)
            col_mask = valid & causal
            k_ptr = K_full + b * stride_kz + g * stride_kg + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptr = V_full + b * stride_vz + g * stride_vg + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptr, mask=col_mask[None, :], other=0.0).to(tl.float32)
            qk = tl.sum(k * q[:, None], axis=0)
            qk = tl.where(col_mask, qk, -1.0e6)
            max_qk = tl.max(qk)
            m_ij = tl.maximum(m_i, max_qk * qk_scale)
            qk = qk * qk_scale - m_ij
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha
            v = tl.load(v_ptr, mask=col_mask[:, None], other=0.0).to(tl.float32)
            p = p.to(tl.float32)
            acc = acc + tl.sum(v * p[:, None], axis=0)
            m_i = m_ij

    m_i = m_i + tl.math.log2(l_i)
    acc = acc / l_i
    tl.store(O_ptr, acc.to(O.type.element_ty))
    # store per-row max
    m_ptr = M + b * stride_mz + g * stride_mg + t * stride_mt
    tl.store(m_ptr, m_i)
