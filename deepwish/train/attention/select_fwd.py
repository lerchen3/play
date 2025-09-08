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
def select_attention_forward(
    Q, K, V,  # tensors
    sm_scale, O, M,  # output and per-row max
    stride_qz, stride_qg, stride_qm, stride_qk,  # B, G, T, D for queries (vertical slice per group)
    stride_kz, stride_kg, stride_kn, stride_kk,  # B, G, N_sel, D for keys
    stride_vz, stride_vg, stride_vk, stride_vn,  # B, G, N_sel, D for values
    stride_oz, stride_og, stride_om, stride_on,  # B, G, T, D for output
    stride_mz, stride_mg, stride_mm,  # B, G, T for M
    BATCH_SIZE, NUM_GROUPS, T_CTX, N_SEL,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RowMask, stride_rmz, stride_rmg, stride_rmm, stride_rmn, USE_MASK: tl.constexpr,
):
    # vertical slice: pid over tokens T, second pid over batch*groups
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)

    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS

    t_start = pid_t * BLOCK_T
    offs_t = t_start + tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q layout: (B, G, T, H_per_g, D) flattened as (B,G,T,D) per head and reduced by sum across heads inside group later
    # Here we aggregate per group by stacking heads in the D dimension (H_per_g * D)
    # For simplicity, assume Q is already reduced to per-head dimension per query head and identical selection across group.
    Q_ptrs = Q + b * stride_qz + g * stride_qg + offs_t[:, None] * stride_qm + offs_d[None, :] * stride_qk
    O_ptrs = O + b * stride_oz + g * stride_og + offs_t[:, None] * stride_om + offs_d[None, :] * stride_on

    # initialize accumulator
    acc = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_T], dtype=tl.float32) + 1.0
    m_i = tl.zeros([BLOCK_T], dtype=tl.float32) - float("inf")
    qk_scale = sm_scale * 1.44269504

    # iterate over selected positions (N_SEL) in BLOCK_N tiles
    k_ptr = K + b * stride_kz + g * stride_kg + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptr = V + b * stride_vz + g * stride_vg + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    for start_n in range(0, N_SEL, BLOCK_N):
        k = tl.load(k_ptr)
        q = tl.load(Q_ptrs)
        qk = tl.dot(q, k)
        # apply optional per-row mask over selected tokens
        if USE_MASK:
            # RowMask layout: (B,G,T,N_SEL)
            rm_ptr = RowMask + b * stride_rmz + g * stride_rmg + offs_t[:, None] * stride_rmm + (start_n + offs_n[None, :]) * stride_rmn
            mask = tl.load(rm_ptr).to(qk.dtype)
            # set disallowed to -inf by multiplying zero then add -inf
            qk = tl.where(mask > 0, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr)
        p = p.to(tl.float32)
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        k_ptr += BLOCK_N * stride_kn
        v_ptr += BLOCK_N * stride_vk

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(O.type.element_ty))
    # store per-row maxima for backward
    m_ptr = M + b * stride_mz + g * stride_mg + offs_t * stride_mm
    tl.store(m_ptr, m_i)


