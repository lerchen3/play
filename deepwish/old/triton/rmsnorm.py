import torch
import triton
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr, N, D,
    stride_xn, stride_xd,
    stride_wn,
    stride_yn, stride_yd,
    stride_rstdn,
    BLOCK_SIZE_D: tl.constexpr,
    EPS: tl.constexpr
):
    row = tl.program_id(0)
    
    _sum_sq = tl.zeros((), dtype=tl.float32)
    for col_block_start in range(0, D, BLOCK_SIZE_D):
        offs_d = col_block_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        x = tl.load(x_ptr + row * stride_xn + offs_d * stride_xd, mask=mask_d, other=0.0)
        _sum_sq += tl.sum(x * x, axis=0)
    
    mean_sq = _sum_sq / D
    rstd_val = tl.rsqrt(mean_sq + EPS)
    tl.store(rstd_ptr + row * stride_rstdn, rstd_val)

    for col_block_start in range(0, D, BLOCK_SIZE_D):
        offs_d = col_block_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        x = tl.load(x_ptr + row * stride_xn + offs_d * stride_xd, mask=mask_d, other=0.0)
        w = tl.load(w_ptr + offs_d * stride_wn, mask=mask_d, other=0.0)
        y = x * rstd_val * w
        tl.store(y_ptr + row * stride_yn + offs_d * stride_yd, y, mask=mask_d)


@triton.jit
def rmsnorm_bwd_kernel(
    x_ptr, w_ptr, dy_ptr, rstd_ptr,
    dx_ptr, dw_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_wn,
    stride_dyn, stride_dyd,
    stride_rstdn,
    stride_dxn, stride_dxd,
    stride_dwn,
    BLOCK_SIZE_D: tl.constexpr,
    EPS: tl.constexpr
):
    row = tl.program_id(0)

    rstd_val = tl.load(rstd_ptr + row * stride_rstdn)

    dot_val = tl.zeros((), dtype=tl.float32)
    sum_dy_x_rstd = tl.zeros((), dtype=tl.float32)

    for col_block_start in range(0, D, BLOCK_SIZE_D):
        offs_d = col_block_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D

        x = tl.load(x_ptr + row * stride_xn + offs_d * stride_xd, mask=mask_d, other=0.0)
        w = tl.load(w_ptr + offs_d * stride_wn, mask=mask_d, other=0.0)
        dy = tl.load(dy_ptr + row * stride_dyn + offs_d * stride_dyd, mask=mask_d, other=0.0)

        dw_row_block = dy * x * rstd_val
        tl.store(dw_ptr + row * D + offs_d, dw_row_block, mask=mask_d)

        dot_val += tl.sum(w * x * dy, axis=0)
        sum_dy_x_rstd += tl.sum(dy * x * rstd_val, axis=0)

    for col_block_start in range(0, D, BLOCK_SIZE_D):
        offs_d = col_block_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D

        x = tl.load(x_ptr + row * stride_xn + offs_d * stride_xd, mask=mask_d, other=0.0)
        w = tl.load(w_ptr + offs_d * stride_wn, mask=mask_d, other=0.0)
        dy = tl.load(dy_ptr + row * stride_dyn + offs_d * stride_dyd, mask=mask_d, other=0.0)

        dx_block = (w * dy * rstd_val) - ( (rstd_val * rstd_val * rstd_val / D) * w * x * dot_val )
        tl.store(dx_ptr + row * stride_dxn + offs_d * stride_dxd, dx_block, mask=mask_d)

class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        N, D = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(N, device=x.device, dtype=torch.float32)

        grid = (N,)
        rmsnorm_fwd_kernel[grid](
            x, weight, y, rstd,
            N, D,
            x.stride(0), x.stride(1), 
            weight.stride(0),
            y.stride(0), y.stride(1),
            rstd.stride(0),
            BLOCK_SIZE_D=1024,
            EPS=eps
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        N, D = x.shape

        dx = torch.empty_like(x)
        dw_row_contribs = torch.empty_like(x)

        grid = (N,)
        rmsnorm_bwd_kernel[grid](
            x, weight, dy, rstd, 
            dx, dw_row_contribs, 
            N, D, 
            x.stride(0), x.stride(1),
            weight.stride(0),
            dy.stride(0), dy.stride(1),
            rstd.stride(0),
            dx.stride(0), dx.stride(1),
            D,
            BLOCK_SIZE_D=1024,
            EPS=eps
        )
        
        dw = dw_row_contribs.sum(dim=0)
        return dx, dw, None

rmsnorm_triton = _RMSNorm.apply

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rstd_ref = torch.rsqrt(variance + eps)
    y_ref = x * rstd_ref * weight
    return y_ref

if __name__ == "__main__":
    test_configs = [
        {"N": 2, "D": 128, "dtype": torch.float32},
        {"N": 4, "D": 256, "dtype": torch.float32},
        {"N": 1, "D": 512, "dtype": torch.float32},
        {"N": 2, "D": 128, "dtype": torch.float16},
        {"N": 4, "D": 256, "dtype": torch.float16},
        {"N": 1, "D": 512, "dtype": torch.float16},
    ]
    eps_val = 1e-5

    for config in test_configs:
        N, D, dtype = config["N"], config["D"], config["dtype"]

        if dtype == torch.float16 and DEVICE == 'cpu':
            print(f"Skipping float16 on CPU for RMSNorm test (N={N}, D={D})")
            continue

        print(f"Testing RMSNorm: N={N}, D={D}, dtype={dtype}, eps={eps_val}")

        x = torch.randn(N, D, device=DEVICE, dtype=dtype, requires_grad=True)
        weight = torch.randn(D, device=DEVICE, dtype=dtype, requires_grad=True)

        x_ref = x.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)

        y_triton = rmsnorm_triton(x, weight, eps_val)
        y_torch = rmsnorm_torch(x_ref, weight_ref, eps_val)

        correct_elements_fwd = torch.isclose(y_triton, y_torch, atol=0.01, rtol=0).sum().item()
        total_elements_fwd = y_torch.numel()
        percentage_correct_fwd = (correct_elements_fwd / total_elements_fwd) * 100
        print(f"  Forward pass: {percentage_correct_fwd:.2f}% of values are within 0.01 of PyTorch output.")
        assert percentage_correct_fwd >= 99.0, f"Forward pass accuracy too low: {percentage_correct_fwd:.2f}%"

        grad_y = torch.randn_like(y_triton)
        grad_y_clone_triton = grad_y.clone()
        grad_y_clone_torch = grad_y.clone()

        y_triton.backward(grad_y_clone_triton)
        y_torch.backward(grad_y_clone_torch)
        
        assert x.grad is not None and x_ref.grad is not None, "Gradients for x not computed."
        assert weight.grad is not None and weight_ref.grad is not None, "Gradients for weight not computed."

        correct_elements_bwd_x = torch.isclose(x.grad, x_ref.grad, atol=0.01, rtol=0).sum().item()
        total_elements_bwd_x = x_ref.grad.numel()
        percentage_correct_bwd_x = (correct_elements_bwd_x / total_elements_bwd_x) * 100
        print(f"  Backward pass (grad_x): {percentage_correct_bwd_x:.2f}% of values are within 0.01 of PyTorch output.")
        assert percentage_correct_bwd_x >= 99.0, f"Backward pass accuracy for grad_x too low: {percentage_correct_bwd_x:.2f}%"

        correct_elements_bwd_w = torch.isclose(weight.grad, weight_ref.grad, atol=0.01, rtol=0).sum().item()
        total_elements_bwd_w = weight_ref.grad.numel()
        percentage_correct_bwd_w = (correct_elements_bwd_w / total_elements_bwd_w) * 100
        print(f"  Backward pass (grad_weight): {percentage_correct_bwd_w:.2f}% of values are within 0.01 of PyTorch output.")
        assert percentage_correct_bwd_w >= 99.0, f"Backward pass accuracy for grad_weight too low: {percentage_correct_bwd_w:.2f}%"

    print("All RMSNorm tests passed!")