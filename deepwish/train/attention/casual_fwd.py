import triton
import triton.language as tl

@triton.jit
# Block-wise attention forward with explicit diagonal flag and sliding window mask (strict lower triangular)
def attention_forward_block(acc, l_i, m_i, q,  \
                            K_block_ptr, V_block_ptr,  \
                            start_m, qk_scale,  \
                            BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  \
                            ON_DIAGONAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  \
                            SEQ_LEN: tl.constexpr, WINDOW_SIZE: tl.constexpr,  \
                            TOP_K: tl.constexpr, top_vals, top_idx, base_col: tl.constexpr, \
                            row_max, USE_BLOCK_CAUSAL: tl.constexpr):
    # determine block range: off-diagonal vs diagonal
    if not ON_DIAGONAL:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # need this to be sequential, so no race condition.
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        # build mask: either block-causal using per-row max-allowed key column, or default strict-lower + optional sliding window
        offs_n_abs = start_n + offs_n[None, :]
        if USE_BLOCK_CAUSAL:
            # allow keys whose column index <= row-specific max block index
            # row_max has shape [BLOCK_M]
            allowed = offs_n_abs <= row_max[:, None]
            mask = allowed
            qk_scaled = qk * qk_scale
            qk_masked = tl.where(mask, qk_scaled, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk_masked, 1))
            qk = qk_masked - m_ij[:, None]
        else:
            strict_causal = offs_m[:, None] > offs_n_abs
            if WINDOW_SIZE > 0:
                within_band = (offs_m[:, None] - offs_n_abs) < WINDOW_SIZE
                mask = strict_causal & within_band
            else:
                mask = strict_causal
            if ON_DIAGONAL:
                qk_scaled = qk * qk_scale
                qk_masked = tl.where(mask, qk_scaled, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk_masked, 1))
                qk = qk_masked - m_ij[:, None]
            else:
                qk_scaled = qk * qk_scale
                qk_masked = tl.where(mask, qk_scaled, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk_masked, 1))
                qk = qk_masked - m_ij[:, None]
        p = tl.math.exp2(qk)
        # exp2 is preferred over exp since GPUs natively implement exp2; exp(x) = exp2(x * 1/log(2)).
        # By folding 1/log(2) into sm_scale, we reduce each element to a single multiply and exp2.
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        p = p.to(tl.float32)
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # streaming top-k update using scaled, masked logits for consistent cross-tile ranking
        if TOP_K > 0:
            abs_cols = base_col + start_n + offs_n[None, :]
            for c in range(BLOCK_N):
                # get scaled logits prior to row-wise subtraction, apply mask
                if USE_BLOCK_CAUSAL:
                    allowed_c = abs_cols[:, c] <= row_max
                else:
                    allowed_c = mask[:, c]
                vals = (qk_scale * tl.dot(q, tl.load(K_block_ptr)[:, c])).to(qk.dtype)  # recompute single-col scaled score
                vals = tl.where(allowed_c, vals, -1.0e6)
                ids = abs_cols[:, c].to(tl.int32)
                # find position of minimum in current top_vals per row
                min_vals = top_vals[:, 0]
                min_pos = tl.zeros([BLOCK_M], dtype=tl.int32)
                for kk in range(1, TOP_K):
                    cand = top_vals[:, kk]
                    take = cand < min_vals
                    min_pos = tl.where(take, tl.full([BLOCK_M], kk, dtype=tl.int32), min_pos)
                    min_vals = tl.where(take, cand, min_vals)
                replace = vals > min_vals
                for kk in range(TOP_K):
                    slot = tl.full([BLOCK_M], kk, dtype=tl.int32)
                    mask_slot = replace & (min_pos == slot)
                    top_vals[:, kk] = tl.where(mask_slot, vals, top_vals[:, kk])
                    top_idx[:, kk] = tl.where(mask_slot, ids, top_idx[:, kk])
    return acc, l_i, m_i

@triton.jit
def attention_forward(Q, K, V, sm_scale, M, Out, TopVal, TopIdx, TOP_K: tl.constexpr,  \
                      stride_qz, stride_qh, stride_qm, stride_qk,  \
                      stride_kz, stride_kh, stride_kn, stride_kk,  \
                      stride_vz, stride_vh, stride_vk, stride_vn,  \
                      stride_oz, stride_oh, stride_om, stride_on,  \
                      BATCH_SIZE, NUM_HEADS, SEQ_LEN,  \
                      HEAD_DIM: tl.constexpr,  \
                      BLOCK_M: tl.constexpr,  \
                      BLOCK_N: tl.constexpr,  \
                      WINDOW_SIZE: tl.constexpr, \
                      RowMax, stride_rmz, stride_rmh, stride_rmm, USE_BLOCK_CAUSAL: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // NUM_HEADS
    head_id = off_hz % NUM_HEADS
    qvk_offset = batch_id.to(tl.int64) * stride_qz + head_id.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = tl.load(Q_block_ptr)
    # load per-row max-allowed key column index if in block-causal mode
    if USE_BLOCK_CAUSAL:
        rm_base = RowMax + batch_id.to(tl.int64) * stride_rmz + head_id.to(tl.int64) * stride_rmh
        row_max = tl.load(rm_base + offs_m * stride_rmm)
    else:
        row_max = tl.zeros([BLOCK_M], dtype=tl.int32)
    # initialize per-row running top-k (values and indices) if requested
    if TOP_K > 0:
        # load initial top buffers slice for current (B,H) rows
        base_ptr = (off_hz * SEQ_LEN + offs_m)[:, None] * TOP_K + tl.arange(0, TOP_K)[None, :]
        top_vals = tl.load(TopVal + base_ptr)
        top_idx = tl.load(TopIdx + base_ptr)
    # process off-diagonal blocks
    acc, l_i, m_i = attention_forward_block(acc, l_i, m_i, q,
                                            K_block_ptr, V_block_ptr,
                                            start_m, qk_scale,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,
                                            False, offs_m, offs_n, SEQ_LEN, WINDOW_SIZE,
                                            TOP_K, top_vals, top_idx, 0, row_max, USE_BLOCK_CAUSAL)
    # process diagonal block
    acc, l_i, m_i = attention_forward_block(acc, l_i, m_i, q,
                                            K_block_ptr, V_block_ptr,
                                            start_m, qk_scale,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,
                                            True, offs_m, offs_n, SEQ_LEN, WINDOW_SIZE,
                                            TOP_K, top_vals, top_idx, 0, row_max, USE_BLOCK_CAUSAL)
    if TOP_K > 0:
        # write back final top-k buffers
        base_ptr = (off_hz * SEQ_LEN + offs_m)[:, None] * TOP_K + tl.arange(0, TOP_K)[None, :]
        tl.store(TopVal + base_ptr, top_vals)
        tl.store(TopIdx + base_ptr, top_idx)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * SEQ_LEN + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
