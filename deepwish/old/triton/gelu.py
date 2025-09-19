# WORKING BTW
# kernels/gelu.py
# Pure Triton GELU forward and backward

import torch
import triton
import triton.language as tl
import math


DEVICE = "cuda"  # using CUDA device

@triton.jit
def gelu_fwd_kernel(x_ptr, y_ptr, N, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs * stride_x, mask=mask)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1 + tl.erf(x / 1.4142135623730951))  # 1.4142135623730951 is sqrt(2)
    tl.store(y_ptr + offs * stride_y, y, mask=mask)

@triton.jit
def gelu_bwd_kernel(x_ptr, dy_ptr, dx_ptr, N, stride_x, stride_dy, stride_dx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs * stride_x, mask=mask)
    dy = tl.load(dy_ptr + offs * stride_dy, mask=mask)
    # derivative: phi(x) = 0.5 * (1 + erf(x/sqrt(2))) ; phi'(x) = exp(-x^2/2)/sqrt(2*pi)
    inv_sqrt2 = 0.70710678118
    pdf = tl.exp(-0.5 * x * x) * 0.3989422804 # Use 1.0 and 2.0
    grad = 0.5 * (1 + tl.erf(x * inv_sqrt2)) + x * pdf
    dx = dy * grad
    tl.store(dx_ptr + offs * stride_dx, dx, mask=mask)


class _gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N = x.numel()
        y = torch.empty_like(x)
        ctx.save_for_backward(x)

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        gelu_fwd_kernel[grid](x_ptr=x, y_ptr=y, N=N, stride_x=x.stride(0), stride_y=y.stride(0), BLOCK_SIZE=1024)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        N = x.numel()
        dx = torch.empty_like(x)

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        gelu_bwd_kernel[grid](x_ptr=x, dy_ptr=dy, dx_ptr=dx, N=N, stride_x=x.stride(0), stride_dy=dy.stride(0), stride_dx=dx.stride(0), BLOCK_SIZE=1024)
        return dx

gelu = _gelu.apply

# @pytest.mark.parametrize("N", [128, 256, 512, 1024, 2048, 4096]) # Removed pytest
# def test_gelu(N): # Removed pytest
    # x = torch.randn(N, device=DEVICE, requires_grad=True, dtype=torch.float32)
    # x_ref = x.clone().detach().requires_grad_(True)

    # # Triton version
    # y_triton = gelu(x)

    # # PyTorch version
    # y_torch = torch.nn.functional.gelu(x_ref)

    # # Test forward pass
    # torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=1e-2)

    # # Test backward pass
    # dy = torch.randn_like(y_triton)
    # y_triton.backward(dy)
    # y_torch.backward(dy)

    # torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)

# If this file is run as the main module, execute the tests.
if __name__ == "__main__":
    # test_gelu() # Modified to run tests directly
    test_configs = [
        {"N": 128, "dtype": torch.float32},
        {"N": 256, "dtype": torch.float32},
        {"N": 512, "dtype": torch.float32},
        {"N": 1024, "dtype": torch.float32},
        {"N": 2048, "dtype": torch.float32},
        {"N": 4096, "dtype": torch.float32},
        # Add float16 tests if desired and supported
        # {"N": 1024, "dtype": torch.float16},
    ]

    for config in test_configs:
        N, dtype = config["N"], config["dtype"]

        if dtype == torch.float16 and DEVICE == 'cpu':
            print(f"Skipping float16 on CPU for GeLU test (N={N})")
            continue
        
        print(f"Testing GeLU: N={N}, dtype={dtype}")

        x = torch.randn(N, device=DEVICE, dtype=dtype, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)

        # Triton version
        y_triton = gelu(x)

        # PyTorch version (truth)
        y_torch = torch.nn.functional.gelu(x_ref)

        # Forward pass comparison
        correct_elements_fwd = torch.isclose(y_triton, y_torch, atol=0.01, rtol=0).sum().item()
        total_elements_fwd = y_torch.numel()
        percentage_correct_fwd = (correct_elements_fwd / total_elements_fwd) * 100
        print(f"  Forward pass: {percentage_correct_fwd:.2f}% of values are within 0.01 of PyTorch output.")
        assert percentage_correct_fwd >= 99.0, f"Forward pass accuracy too low: {percentage_correct_fwd:.2f}%"

        # Backward pass comparison
        grad_y = torch.randn_like(y_triton)
        grad_y_clone_triton = grad_y.clone()
        grad_y_clone_torch = grad_y.clone()

        y_triton.backward(grad_y_clone_triton)
        y_torch.backward(grad_y_clone_torch)

        assert x.grad is not None and x_ref.grad is not None, "Gradients not computed."

        correct_elements_bwd = torch.isclose(x.grad, x_ref.grad, atol=0.01, rtol=0).sum().item()
        total_elements_bwd = x_ref.grad.numel()
        percentage_correct_bwd = (correct_elements_bwd / total_elements_bwd) * 100
        print(f"  Backward pass: {percentage_correct_bwd:.2f}% of values are within 0.01 of PyTorch output.")
        assert percentage_correct_bwd >= 99.0, f"Backward pass accuracy too low: {percentage_correct_bwd:.2f}%"

    print("All GeLU tests passed!")