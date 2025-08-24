import torch
import triton
import triton.language as tl
import math
import triton.testing

DEVICE = "cuda"

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        # Additional configs for smaller sizes
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_bf16(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Use fp32 accumulator for better precision while keeping inputs/outputs as bf16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load bf16 data and convert to fp32 for computation
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert back to bf16 for output
    c_output = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_store_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_store_ptrs, c_output, mask=c_mask)

def _matmul_launcher_bf16(a, b):
    # a: [M, K] and b: [K, N] -> returns [M, N]
    # Ensure inputs are bf16
    a = a.to(torch.bfloat16) if a.dtype != torch.bfloat16 else a
    b = b.to(torch.bfloat16) if b.dtype != torch.bfloat16 else b
    
    M, K = a.shape
    _, N = b.shape
    
    # Ensure contiguous
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
        
    # allocate output in BF16
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) *
                         triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bf16[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

class _MatmulBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Ensure inputs are bf16
        a = a.to(torch.bfloat16) if a.dtype != torch.bfloat16 else a
        b = b.to(torch.bfloat16) if b.dtype != torch.bfloat16 else b
        
        # Retain gradients for non-leaf input tensors if they require grad
        if a.requires_grad:
            a.retain_grad()
        if b.requires_grad:
            b.retain_grad()
            
        assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
        c = _matmul_launcher_bf16(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        with torch.no_grad():
            # Ensure grad_c is bf16
            grad_c = grad_c.to(torch.bfloat16) if grad_c.dtype != torch.bfloat16 else grad_c
            
            # Use BF16 for backward matmuls
            grad_a = _matmul_launcher_bf16(
                grad_c,
                b.transpose(-2, -1)
            )
            grad_b = _matmul_launcher_bf16(
                a.transpose(-2, -1),
                grad_c
            )
        return grad_a, grad_b

matmul_bf16 = _MatmulBF16.apply

# Benchmark against torch.matmul
configs = [
    triton.testing.Benchmark(
        x_names=["M","N","K"],
        x_vals=[128,256,512,1024,2048,4096],
        line_arg="provider",
        line_vals=["Torch","Triton"],
        line_names=["Torch","Triton"],
        styles=[("red","-"),("blue","-")],
        ylabel="Time (ms)",
        plot_name="Matmul BF16 Benchmark",
        args={"dtype": torch.bfloat16},
    )
]

@triton.testing.perf_report(configs)
def benchmark_matmul_bf16(M, N, K, provider, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    if provider == "Torch":
        result = triton.testing.do_bench(lambda: torch.matmul(a, b))
    else:
        result = triton.testing.do_bench(lambda: matmul_bf16(a, b))
    if isinstance(result, tuple):
        ms, min_ms, max_ms = result
    else:
        ms = min_ms = max_ms = result
    return ms, min_ms, max_ms

if __name__ == "__main__":
    test_configs = [
        {"M": 64, "N": 64, "K": 128},
        {"M": 128, "N": 256, "K": 512},
        {"M": 256, "N": 512, "K": 1024},
    ]
    for config in test_configs:
        M, N, K = config["M"], config["N"], config["K"]
        print(f"Testing matmul_bf16: M={M}, N={N}, K={K}")
        a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        a_ref = a.clone().detach().requires_grad_(True)
        b_ref = b.clone().detach().requires_grad_(True)
        
        c_triton = matmul_bf16(a.contiguous(), b.contiguous())
        c_torch = torch.matmul(a_ref, b_ref)
        
        correct_elements_fwd = torch.isclose(c_triton, c_torch, atol=0.001, rtol=0.001).sum().item()
        total_elements_fwd = c_torch.numel()
        percentage_correct_fwd = (correct_elements_fwd / total_elements_fwd) * 100
        print(f"  Forward pass: {percentage_correct_fwd:.2f}% of values are within 0.001 of PyTorch output.")
        assert percentage_correct_fwd >= 89.0, f"Forward pass accuracy too low: {percentage_correct_fwd:.2f}%"
        
        grad_c = torch.randn_like(c_triton)
        grad_c_clone_for_triton = grad_c.clone()
        grad_c_clone_for_torch = grad_c.clone()
        
        c_triton.backward(grad_c_clone_for_triton)
        c_torch.backward(grad_c_clone_for_torch)
        
        if a.grad is not None and a_ref.grad is not None:
            correct_elements_bwd_a = torch.isclose(a.grad, a_ref.grad, atol=0.001, rtol=0.001).sum().item()
            total_elements_bwd_a = a_ref.grad.numel()
            percentage_correct_bwd_a = (correct_elements_bwd_a / total_elements_bwd_a) * 100
            print(f"  Backward pass (grad_a): {percentage_correct_bwd_a:.2f}% of values are within 0.001 of PyTorch output.")
            assert percentage_correct_bwd_a >= 85.0, f"Backward pass accuracy for grad_a too low: {percentage_correct_bwd_a:.2f}%"
        
        if b.grad is not None and b_ref.grad is not None:
            correct_elements_bwd_b = torch.isclose(b.grad, b_ref.grad, atol=0.001, rtol=0.001).sum().item()
            total_elements_bwd_b = b_ref.grad.numel()
            percentage_correct_bwd_b = (correct_elements_bwd_b / total_elements_bwd_b) * 100
            print(f"  Backward pass (grad_b): {percentage_correct_bwd_b:.2f}% of values are within 0.001 of PyTorch output.")
            assert percentage_correct_bwd_b >= 85.0, f"Backward pass accuracy for grad_b too low: {percentage_correct_bwd_b:.2f}%"
    
    print("All matmul_bf16 tests passed!")
    benchmark_matmul_bf16.run(show_plots=True, print_data=True) 