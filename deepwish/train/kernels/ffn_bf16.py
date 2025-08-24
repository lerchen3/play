import torch
import triton
import triton.language as tl
import triton.testing
import torch.nn.functional as F

# Triton FFN forward kernel: one row at a time
@triton.jit
def _ffn_fwd_kernel(
    x_ptr, W1_ptr, b1_ptr, W2_ptr, b2_ptr, gamma_ptr, out_ptr,
    stride_xm, stride_xd,
    stride_w1_im, stride_w1_ih, stride_b1,
    stride_w2_hk, stride_w2_kd, stride_b2,
    stride_gamma, stride_outm, stride_outd,
    D: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr
):
    row = tl.program_id(0)
    # offsets for feature dimension
    offs_h = tl.arange(0, H)
    # allocate space for hidden activation in fp32
    h = tl.zeros((H,), dtype=tl.float32)
    # compute GELU(x @ W1 + b1) into h via scatter
    # MatMul + GELU in fp32, vectorized
    # load input row once
    x_row = tl.load(x_ptr + row * stride_xm + offs_h * stride_xd, mask=offs_h < D)
    # For each hidden dim, compute dot and GELU
    for j in range(H):
        w1_vec = tl.load(W1_ptr + offs_h * stride_w1_im + j * stride_w1_ih, mask=offs_h < D)
        acc = tl.sum(x_row * w1_vec, axis=0) + tl.load(b1_ptr + j * stride_b1)
        # GELU approximation
        t0 = acc + 0.044715 * acc * acc * acc
        h_val = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * t0))
        mask_h = offs_h == j
        h = tl.where(mask_h, h_val, h)
    # Compute RMSNorm denominator
    sum_sq = 0.0
    # vectorize over hidden dim for each output dim k
    for k in range(D):
        w2_vec = tl.load(W2_ptr + offs_h * stride_w2_hk + k * stride_w2_kd, mask=offs_h < H)
        acc2 = tl.sum(h * w2_vec, axis=0) + tl.load(b2_ptr + k * stride_b2)
        sum_sq += acc2 * acc2
    mean_sq = sum_sq / D
    inv_norm = tl.rsqrt(mean_sq + EPS)
    # Write normalized output in bfloat16
    for k in range(D):
        w2_vec = tl.load(W2_ptr + offs_h * stride_w2_hk + k * stride_w2_kd, mask=offs_h < H)
        acc2 = tl.sum(h * w2_vec, axis=0) + tl.load(b2_ptr + k * stride_b2)
        g_val = tl.load(gamma_ptr + k * stride_gamma)
        y = acc2 * inv_norm * g_val
        tl.store(out_ptr + row * stride_outm + k * stride_outd, y)

# Autograd wrapper for FFN
class FFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, b1, W2, b2, gamma, eps=1e-6):
        M, D = x.shape
        H = W1.shape[1]
        out = torch.empty_like(x, dtype=torch.bfloat16)
        # Launch Triton kernel
        grid = (M,)
        _ffn_fwd_kernel[grid](
            x, W1, b1, W2, b2, gamma, out,
            x.stride(0), x.stride(1),
            W1.stride(0), W1.stride(1), b1.stride(0),
            W2.stride(0), W2.stride(1), b2.stride(0),
            gamma.stride(0), out.stride(0), out.stride(1),
            D, H, eps
        )
        ctx.save_for_backward(x, W1, b1, W2, b2, gamma)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, b1, W2, b2, gamma = ctx.saved_tensors
        eps = ctx.eps
        # Recompute full-precision for gradient
        x_fp32 = x.to(torch.float32).requires_grad_(True)
        W1_fp32 = W1.to(torch.float32).requires_grad_(True)
        b1_fp32 = b1.to(torch.float32).requires_grad_(True)
        W2_fp32 = W2.to(torch.float32).requires_grad_(True)
        b2_fp32 = b2.to(torch.float32).requires_grad_(True)
        gamma_fp32 = gamma.to(torch.float32).requires_grad_(True)
        # Python FFN
        h = F.gelu(x_fp32 @ W1_fp32 + b1_fp32)
        y = h @ W2_fp32 + b2_fp32
        rms = torch.sqrt((y * y).mean(-1, keepdim=True) + eps)
        out_fp32 = y / rms * gamma_fp32
        # Autograd
        grad_out_fp32 = grad_out.to(torch.float32)
        grads = torch.autograd.grad(
            out_fp32, (x_fp32, W1_fp32, b1_fp32, W2_fp32, b2_fp32, gamma_fp32),
            grad_out_fp32,
            retain_graph=False, allow_unused=True
        )
        # Cast back to bfloat16
        grads_bf16 = [g.to(torch.bfloat16) if (g is not None) else None for g in grads]
        return (*grads_bf16, None)

# Convenience function
def ffn(x, W1, b1, W2, b2, gamma, eps=1e-6):
    return FFNFunction.apply(x, W1, b1, W2, b2, gamma, eps)

# Torch reference implementation
def ffn_torch(x, W1, b1, W2, b2, gamma, eps=1e-6):
    x_fp32 = x.to(torch.float32)
    h = F.gelu(x_fp32 @ W1.to(torch.float32) + b1.to(torch.float32))
    y = h @ W2.to(torch.float32) + b2.to(torch.float32)
    rms = torch.sqrt((y * y).mean(-1, keepdim=True) + eps)
    out = y / rms * gamma.to(torch.float32)
    return out.to(torch.bfloat16)

# Tests
def test_forward():
    M, D, H = 64, 256, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(M, D, device=device, dtype=torch.bfloat16)
    W1 = torch.randn(D, H, device=device, dtype=torch.bfloat16)
    b1 = torch.randn(H, device=device, dtype=torch.bfloat16)
    W2 = torch.randn(H, D, device=device, dtype=torch.bfloat16)
    b2 = torch.randn(D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    out_ref = ffn_torch(x, W1, b1, W2, b2, gamma)
    out_triton = ffn(x, W1, b1, W2, b2, gamma)
    assert torch.allclose(out_ref, out_triton, atol=1e-2), 'FFN forward mismatch'
    print('FFN forward test passed')

# Benchmark
configs = [
    triton.testing.Benchmark(
        x_names=['M'], x_vals=[64, 128, 256, 512], line_arg='provider',
        line_vals=['Torch', 'Triton'], line_names=['Torch', 'Triton'],
        styles=[('red','-'),('blue','-')],
        ylabel='Time (ms)', plot_name='FFN Benchmark',
        args={'D':256,'H':512}
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, provider, D, H):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(M, D, device=device, dtype=torch.bfloat16)
    W1 = torch.randn(D, H, device=device, dtype=torch.bfloat16)
    b1 = torch.randn(H, device=device, dtype=torch.bfloat16)
    W2 = torch.randn(H, D, device=device, dtype=torch.bfloat16)
    b2 = torch.randn(D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    if provider == 'Torch':
        return triton.testing.do_bench(lambda: ffn_torch(x, W1, b1, W2, b2, gamma))
    else:
        return triton.testing.do_bench(lambda: ffn(x, W1, b1, W2, b2, gamma))

if __name__ == '__main__':
    test_forward()
    benchmark.run(show_plots=True, print_data=True) 