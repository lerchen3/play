import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["HEAD_DIM", "BLOCK_N"],
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
    MAX_TILES: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645956
    pid_t = tl.program_id(0)
    pid_bg = tl.program_id(1)

    b = pid_bg // NUM_GROUPS
    g = pid_bg % NUM_GROUPS
    t = pid_t

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    Q_ptr = Q + b * stride_qz + g * stride_qg + t * stride_qt + offs_d * stride_qd
    O_ptr = O + b * stride_oz + g * stride_og + t * stride_ot + offs_d * stride_od
    DO_ptr = DO + b * stride_oz + g * stride_og + t * stride_ot + offs_d * stride_od
    DQ_ptr = DQ + b * stride_dqz + g * stride_dqg + t * stride_dqt + offs_d * stride_dqd

    m = tl.load(M + b * stride_mz + g * stride_mg + t * stride_mt)

    q = tl.load(Q_ptr).to(tl.float32)
    o = tl.load(O_ptr).to(tl.float32)
    do = tl.load(DO_ptr).to(tl.float32)
    dq = tl.zeros([HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

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
            causal = cols <= t
            col_mask = valid & causal
            k_ptr = K_full + b * stride_kz + g * stride_kg + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptr = V_full + b * stride_vz + g * stride_vg + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptr, mask=col_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=col_mask[:, None], other=0.0).to(tl.float32)
            qk = tl.sum(k * q[:, None], axis=0)
            qk = tl.where(col_mask, qk, -1.0e6)
            qk = qk * qk_scale
            p = tl.math.exp2(qk - m)
            p = tl.where(col_mask, p, 0.0)
            p32 = p.to(tl.float32)
            dv_tile = p32[:, None] * do[None, :]
            dp = tl.sum(v * do[None, :], axis=1)
            delta = tl.sum(o * do)
            ds = p32 * (dp - delta)
            dk_tile = ds[:, None] * q[None, :]
            dq += tl.sum(k * ds[:, None], axis=0)

            dv_dst = DV_full + b * stride_dvz + g * stride_dvg + cols[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
            dk_dst = DK_full + b * stride_dkz + g * stride_dkg + cols[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
            tl.atomic_add(dv_dst, dv_tile, mask=col_mask[:, None])
            tl.atomic_add(dk_dst, dk_tile, mask=col_mask[:, None])

    dq *= LN2
    tl.store(DQ_ptr, dq)
