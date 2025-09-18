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
):
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS
    t = pid_t

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptr = Q + b * stride_qz + g * stride_qg + t * stride_qt + offs_d * stride_qd
    q = tl.load(q_ptr)

    # row-wise max allowed column (inclusive)
    rmax = tl.load(RowMax + b * stride_rmz + g * stride_rmg + t * stride_rmt)

    # running top-k buffers across all columns
    best_val = tl.full([TOP_K], -1.0e9, dtype=tl.float32)
    best_idx = tl.full([TOP_K], -1, dtype=tl.int32)

    qk_scale = sm_scale
    qk_scale *= 1.0  # keep in native base-e; we just need ranks, scaling constant factor doesn't change order

    start_n = 0
    while start_n < N_COLS:
        # process columns one by one to avoid vector indexing issues in Triton
        pos_k = tl.arange(0, TOP_K)
        for j in range(BLOCK_N):
            col = start_n + j
            # bounds and causal
            inb = col < N_COLS
            causal = col <= rmax
            active = inb & causal
            # load K[:, col]
            k_col_ptr = K + b * stride_kz + g * stride_kg + col * stride_kn + offs_d * stride_kd
            k_col = tl.load(k_col_ptr, mask=active, other=0.0)
            # compute q·k for this column
            v = tl.sum(q * k_col, axis=0) * qk_scale
            # if out of causal or bounds, set score to -inf
            v = tl.where(active, v, -1.0e9)

            # streaming top-k insert: replace current minimum if better
            min_val = tl.min(best_val)
            is_better = v > min_val
            is_min = best_val == min_val
            empty = best_idx == -1
            is_min_empty = is_min & empty
            neg_pos = -pos_k
            # choose lowest index among empty minima if any; otherwise lowest among all minima
            chosen_neg_empty = tl.max(tl.where(is_min_empty, neg_pos, tl.full([TOP_K], -1073741824, dtype=tl.int32)))
            chosen_neg_all = tl.max(tl.where(is_min, neg_pos, tl.full([TOP_K], -1073741824, dtype=tl.int32)))
            has_empty = tl.max(tl.where(is_min_empty, tl.full([TOP_K], 1, dtype=tl.int32), tl.full([TOP_K], 0, dtype=tl.int32)))
            chosen_neg = tl.where(has_empty == 1, chosen_neg_empty, chosen_neg_all)
            min_pos = -chosen_neg
            upd = pos_k == min_pos
            # if any empty minima exist, we must fill one of them regardless of value
            use_empty = has_empty == 1
            do_upd = (is_better | use_empty) & inb
            best_val = tl.where(upd & do_upd, v, best_val)
            best_idx = tl.where(upd & do_upd, col, best_idx)

        start_n += BLOCK_N

    # write indices (unordered among top-k; stable order not guaranteed)
    out_ptr = TopIdx + b * stride_tz + g * stride_tg + t * stride_tt + tl.arange(0, TOP_K) * stride_tk
    tl.store(out_ptr, best_idx)

