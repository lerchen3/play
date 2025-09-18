import triton
import triton.language as tl

@triton.jit
# Block-wise attention forward with explicit diagonal flag
def attention_forward_block(acc, l_i, m_i, q,  \
                            K_block_ptr, V_block_ptr,  \
                            start_m, qk_scale,  \
                            BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  \
                            ON_DIAGONAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  \
                            SEQ_LEN: tl.constexpr):
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
        if ON_DIAGONAL:
            # strict lower-triangular mask (no self)
            mask = offs_m[:, None] > (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, 1))
            qk = qk * qk_scale - m_ij[:, None]
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
        # avoid NaN propagation when there are no valid cols: if l_ij==0 then p==0; safe
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def attention_forward(Q, K, V, sm_scale, M, Out,  \
                      stride_qz, stride_qh, stride_qm, stride_qk,  \
                      stride_kz, stride_kh, stride_kn, stride_kk,  \
                      stride_vz, stride_vh, stride_vk, stride_vn,  \
                      stride_oz, stride_oh, stride_om, stride_on,  \
                      BATCH_SIZE, NUM_HEADS, SEQ_LEN,  \
                      HEAD_DIM: tl.constexpr,  \
                      BLOCK_M: tl.constexpr,  \
                      BLOCK_N: tl.constexpr):
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
    # process off-diagonal blocks
    acc, l_i, m_i = attention_forward_block(acc, l_i, m_i, q,
                                            K_block_ptr, V_block_ptr,
                                            start_m, qk_scale,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,
                                            False, offs_m, offs_n, SEQ_LEN)
    # process diagonal block
    acc, l_i, m_i = attention_forward_block(acc, l_i, m_i, q,
                                            K_block_ptr, V_block_ptr,
                                            start_m, qk_scale,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,
                                            True, offs_m, offs_n, SEQ_LEN)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * SEQ_LEN + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
