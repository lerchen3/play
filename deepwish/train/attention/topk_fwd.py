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
def topk_forward(
    Q, K, sm_scale, TopIdx,
    stride_qz, stride_qg, stride_qt, stride_qd,
    stride_kz, stride_kg, stride_kn, stride_kd,
    stride_tz, stride_tg, stride_tt, stride_tk,
    RowMax, stride_rmz, stride_rmg, stride_rmt,
    BATCH_SIZE, NUM_GROUPS, T_CTX, N_COLS,
    TOP_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS
    t = pid_t

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + b * stride_qz + g * stride_qg + t * stride_qt + offs_d * stride_qd
    q = tl.load(q_ptr)
    q = q.to(tl.float32)

    # row-wise max allowed column (inclusive)
    rmax = tl.load(RowMax + b * stride_rmz + g * stride_rmg + t * stride_rmt)

    # initialize per-slot running maxima
    best_val = tl.full([TOP_K], -1.0e9, dtype=tl.float32)
    best_idx = tl.full([TOP_K], -1, dtype=tl.int32)
    idx_range = tl.arange(0, TOP_K)
    invalid_idx = tl.full([TOP_K], TOP_K, dtype=tl.int32)

    qk_scale = sm_scale
    qk_scale *= 1.0  # keep in native base-e; we just need ranks, scaling constant factor doesn't change order

    for tile in range(0, MAX_TILES):
        start_n = tile * BLOCK_N
        for c in range(0, BLOCK_N):
            col = start_n + c
            col_i = tl.full([], col, dtype=tl.int32)
            valid = col_i < N_COLS
            causal = col_i <= rmax
            col_valid = valid & causal
            k_ptr = K + b * stride_kz + g * stride_kg + col * stride_kn + offs_d * stride_kd
            k_col = tl.load(k_ptr, mask=col_valid, other=0.0)
            k_col = k_col.to(tl.float32)
            val = tl.sum(k_col * q) * qk_scale
            val = tl.where(col_valid, val, -1.0e9)
            min_val = tl.min(best_val, axis=0)
            min_candidates = tl.where(best_val == min_val, idx_range, invalid_idx)
            min_pos = tl.min(min_candidates, axis=0)
            replace = (val > min_val) & col_valid
            update = (idx_range == min_pos) & replace
            best_val = tl.where(update, val, best_val)
            best_idx = tl.where(update, col_i, best_idx)

    # write indices (unordered among top-k; stable order not guaranteed)
    empty = best_idx == -1
    best_idx = tl.where(empty, idx_range, best_idx)
    out_ptr = TopIdx + b * stride_tz + g * stride_tg + t * stride_tt + idx_range * stride_tk
    tl.store(out_ptr, best_idx)
