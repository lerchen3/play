# kernels/0atomicallreduce.py

# TODO : possibly if we scale up to full vocab and not bytewise, use atomics for Apple's CCE.

import math
import torch
import triton
import triton.language as tl
import time

DEVICE = "cuda"
# BLOCK_SIZE = number of rows each program reduces, and tile size for B internally
BLOCK_SIZE = 128  

# ---------------------------------------------------------------------------- #
# 1) Pure PyTorch baseline
# ---------------------------------------------------------------------------- #
def torch_allreduce(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=0)

# ---------------------------------------------------------------------------- #
# 2) Flat atomic all‐reduce in one kernel
# ---------------------------------------------------------------------------- #
@triton.jit
def _atomic_kernel(
    x_ptr,         # [A, B]
    out_ptr,       # [num_leaves, B]
    cnt_ptr,       # scalar counter
    A, B,
    stride_ax, stride_ay,
    stride_ox, stride_oy,
    BLOCK_SIZE: tl.constexpr,
):
    pid       = tl.program_id(0)                # which block of rows this program reduces
    row_start = pid * BLOCK_SIZE

    # 1) compute the partial sum for rows [row_start .. row_start+BLOCK_SIZE)
    #    in tiles of BLOCK_SIZE over the B dimension
    for col_start in range(0, B, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < B
        acc  = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(BLOCK_SIZE):
            r    = row_start + i
            vals = tl.load(x_ptr + r * stride_ax + offs * stride_ay, mask=mask, other=0.0)
            acc += vals
        # write leaf partial into out_ptr[pid, col_start:col_start+BLOCK_SIZE]
        tl.store(out_ptr + pid * stride_ox + offs * stride_oy, acc, mask=mask)

    # 2) increment the single global counter; if we're the last leaf, do a gather+sum
    old        = tl.atomic_add(cnt_ptr, 1)
    num_leaves = tl.cdiv(A, BLOCK_SIZE)
    if old == num_leaves - 1:
        # last leaf to finish: sum all leaf partials into out_ptr[0]
        for col_start in range(0, B, BLOCK_SIZE):
            offs  = col_start + tl.arange(0, BLOCK_SIZE)
            mask  = offs < B
            total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for L in range(num_leaves):
                part = tl.load(
                    out_ptr + L * stride_ox + offs * stride_oy,
                    mask=mask, other=0.0
                )
                total += part
            tl.store(out_ptr + offs * stride_oy, total, mask=mask)

def atomic_allreduce_triton(x: torch.Tensor) -> torch.Tensor:
    A, B = x.shape
    assert A % BLOCK_SIZE == 0, "A must be divisible by BLOCK_SIZE"
    num_leaves = A // BLOCK_SIZE
    # buffer for leaf partials + final result in out[0]
    out = torch.zeros((num_leaves, B), device=x.device, dtype=torch.float32)
    cnt = torch.zeros([],         device=x.device, dtype=torch.int32)
    _atomic_kernel[(num_leaves,)](
        x, out, cnt,
        A, B,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out[0]   # the fully reduced vector

# ---------------------------------------------------------------------------- #
# 3) Single‐kernel binary‐tree all‐reduce
# ---------------------------------------------------------------------------- #
@triton.jit
def _tree_kernel(
    x_ptr,         # [A, B]
    node_ptr,      # [2*num_leaves-1, B]
    cnt_ptr,       # [2*num_leaves-1] integer counters
    A, B,
    stride_ax, stride_ay,
    stride_n0, stride_nb,
    stride_cnt,
    BLOCK_SIZE: tl.constexpr,
    LOG: tl.constexpr,
):
    num_leaves = tl.cdiv(A, BLOCK_SIZE)
    # 1) leaf index in the full tree array
    pid        = tl.program_id(0)                # 0 .. num_leaves-1
    leaf_index = num_leaves - 1 + pid
    row_start  = pid * BLOCK_SIZE

    # 2) compute and store leaf partial
    for col_start in range(0, B, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < B
        acc  = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(BLOCK_SIZE):
            r    = row_start + i
            acc += tl.load(x_ptr + r * stride_ax + offs * stride_ay, mask=mask, other=0.0)
        tl.store(node_ptr + leaf_index * stride_n0 + offs * stride_nb, acc, mask=mask)

    # 3) climb up the tree. refactored to not use break.
    cur = leaf_index
    level = 0
    while (level < LOG) and (cur != 0):
        parent = (cur - 1) // 2
        old = tl.atomic_add(cnt_ptr + parent * stride_cnt, 1)
        if old == 1:
            # second child to arrive: load both children's partials & sum
            left = 2 * parent + 1
            right = left + 1
            for col_start in range(0, B, BLOCK_SIZE):
                offs = col_start + tl.arange(0, BLOCK_SIZE)
                mask = offs < B
                a = tl.load(node_ptr + left * stride_n0 + offs * stride_nb,
                            mask=mask, other=0.0)
                b = tl.load(node_ptr + right * stride_n0 + offs * stride_nb,
                            mask=mask, other=0.0)
                s = a + b
                tl.store(node_ptr + parent * stride_n0 + offs * stride_nb,
                         s, mask=mask)
            # now climb further: update current node index for next iteration
            cur = parent
        else:
            # first arrival: exit early by setting cur to 0 to end the loop
            cur = 0
        level += 1

def tree_allreduce_triton(x: torch.Tensor) -> torch.Tensor:
    A, B = x.shape
    assert A % BLOCK_SIZE == 0, "A must be divisible by BLOCK_SIZE"
    num_leaves = A // BLOCK_SIZE
    LOG        = int(math.log2(num_leaves))
    num_nodes  = 2 * num_leaves - 1
    # node_buf will hold one vector of length B per tree node
    node_buf = torch.zeros((num_nodes, B), device=x.device, dtype=torch.float32)
    cnt_buf  = torch.zeros((num_nodes,), device=x.device, dtype=torch.int32)
    _tree_kernel[(num_leaves,)](
        x, node_buf, cnt_buf,
        A, B,
        x.stride(0), x.stride(1),
        node_buf.stride(0), node_buf.stride(1),
        cnt_buf.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, LOG=LOG
    )
    # the root (node 0) holds the final sum
    return node_buf[0]

# ---------------------------------------------------------------------------- #
# Quick sanity‐check
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_shapes = [
        (128, 2**18),
        (256, 2**18),
        (512, 2**18),
        (1024, 2**18),
        (2048, 2**18),
        (4096, 2**18),
        (8192, 2**18)]
    for A, B in test_shapes:
        print(f"\nTesting A={A}, B={B}")
        x = torch.randn((A, B), device=DEVICE, dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.time()
        y_ref  = torch_allreduce(x)
        torch.cuda.synchronize()
        ref_time = time.time() - start

        torch.cuda.synchronize()
        start = time.time()
        y_flat = atomic_allreduce_triton(x)
        torch.cuda.synchronize()
        flat_time = time.time() - start

        torch.cuda.synchronize()
        start = time.time()
        y_tree = tree_allreduce_triton(x)
        torch.cuda.synchronize()
        tree_time = time.time() - start

        print(f"  ref_time:  {ref_time:.3f}s")
        print(f"  flat_time: {flat_time:.3f}s")
        print(f"  tree_time: {tree_time:.3f}s")

        for name, y in [("flat", y_flat), ("tree", y_tree)]:
            pct = (torch.isclose(y, y_ref, atol=1e-2)
                   .float().mean() * 100).item()
            print(f"  {name:5s} accuracy: {pct:5.2f}%")
            assert pct >= 99.0, f"{name} allreduce failed: {pct:.2f}%"
    print("All tests passed!")