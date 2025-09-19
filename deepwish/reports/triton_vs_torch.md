# Triton vs. PyTorch Kernel Benchmark Sweeps

Timings collected with CUDA events (averaged per iteration).

PyTorch baselines use unfused reference implementations; Triton kernels are forward-optimized.
Backward timings rely on PyTorch autograd when Triton backward kernels are unavailable.

## triton_cce_loss vs. PyTorch (vocab sweep)

| vocab | Torch FWD (ms) | Triton FWD (ms) | FWD Speedup | Torch BWD (ms) | Triton BWD (ms) | BWD Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.39 | 3.05 | 0.13x | 0.70 | 94.85 | 0.01x |
| 4096 | 0.53 | 4.41 | 0.12x | 1.05 | 187.11 | 0.01x |
| 8192 | 0.93 | 8.29 | 0.11x | 1.70 | 371.11 | 0.00x |
| 16384 | 1.65 | 15.99 | 0.10x | 3.14 | 739.35 | 0.00x |

Configs:
- batch=4, seq=1024, dim=1024, vocab=2048, dtype=torch.float32, block_size=128, warmup=1, iters=5
- batch=4, seq=1024, dim=1024, vocab=4096, dtype=torch.float32, block_size=128, warmup=1, iters=5
- batch=4, seq=1024, dim=1024, vocab=8192, dtype=torch.float32, block_size=128, warmup=1, iters=5
- batch=4, seq=1024, dim=1024, vocab=16384, dtype=torch.float32, block_size=128, warmup=1, iters=5

## Environment
* PyTorch 2.7.0
* CUDA device: NVIDIA A100-SXM4-40GB