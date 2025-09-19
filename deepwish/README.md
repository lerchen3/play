# deepwish

deepwish is a GPU-first research stack for rebuilding and extending the Qwen3 / Qwen3-Next / DeepSeek-V3 model architetures. Every regression, notebook, and plot in this repo reflects that baseline so the deltas are attributable to architecture changes, not tokenizers.

**Platform note:** serious work happens on Linux with recent CUDA + Triton. The tiny configs and non-kernel unit tests run on CPU, but all custom kernels, multi-GPU training, and offload paths expect NVIDIA hardware.

## Code Organization
- `configs/` - Versioned YAML recipes; `deepseekv3_tiny.yaml` is the quick sanity check config.
- `examples/` - Sample chat CSV used by the dataset loader.
- `inference/` - Autoregressive runners, speculative decoding (`specdec.py`), and architecture-specific entry points like `qwen3next_inference.py`.
- `model/` - Shared blocks plus architecture packages for DeepSeek-V3 (MLA MoE + MTP), Qwen3 (GQA), and Qwen3-Next (GatedDelta + gated attention).
- `train/` - Distributed training loop, offload flows, and Triton attention kernels under `train/attention/` and `train/kernels/`.
- `scripts/` - CLI helpers (`train_tiny.sh`, `infer_tiny.sh`, `profile_kernels.py`, etc.) used in the docs and tests.
- `plots/` & `reports/` - Collected benchmarks, ablations, and run summaries; the latest Triton perf charts and attention quality notes live here.
- `tests/` - Pytest suite spanning dataset IO, attention kernels, inference stepping, and architecture-level forward passes.
- `wait-it-works.ipynb` - End-to-end GPU notebook that wires Triton kernels into inference and logs kernel timings.

## Attention & Triton R&D
- **NSA attention** (compressed, select, sliding-window) sits in `train/attention/nsa.py` with a design doc in `train/attention/nsa.md`. The Triton kernels cover fused prefill + decode and ship with forward/backward unit tests.
- **Qwen3** adopts grouped-query attention (GQA) with cache-aware key/value handling in `model/qwen3/kv.py` and is fully exercised by `tests/test_qwen3_model.py`, running NSA attention.
- **Qwen3-Next** layers gated attention and the "GatedDelta" update path. See `model/qwen3next/gated_delta.py` plus inference coverage in `tests/test_qwen3next_components.py`.
- **DeepSeek-V3** integrates MLA, expert routing, and multi-token prediction. Components live under `model/deepseekv3/` and feed into `train/offload_step_dsv3.py`.
- All kernel variants have Triton reimplementations (`train/kernels/cce_triton.py`, `train/kernels/fa2.py`) with BF16/FP32 toggles and matching CPU fallbacks for correctness checks.

## Training Runs
- The main entry point is `train/train.py` (torch.distributed, gradient accumulation, activation offload, EMA). `train/train_offload.py` mirrors the flow for CPU-mapped weight shards.
- `scripts/train_tiny.sh` launches a 1-GPU sanity run; larger configs swap in via `--config` or environment overrides.
- Full-scale training runs are in progress on the Qwen3 and Qwen3-Next branches; results and write-ups drop into `reports/` as they land.
- Upcoming work targets fresh architectures that are not derived from Qwen3, Qwen3-Next, or DeepSeek-V3. Design sketches and kernel stubs are incubating under `experiments/` and will graduate once end-to-end tests exist.

## Inference Path
- `inference/inference.py` provides the common sampling loop (prefill + decode + cache eviction) with hooks for NSA and GQA.
- `inference/qwen3next_inference.py` layers the gated attention + GatedDelta updates needed for the Next stack.
- `inference/specdec.py` implements speculative decoding with configurable draft depth; unit tests cover fallback-to-main transitions.
- `scripts/infer_tiny.sh` is the quickest way to exercise a checkpoint: set `MODEL_PATH`, `TOKENIZER_PATH=Qwen/Qwen3-0.6B`, and a prompt.

## Evaluation, Tests, and Reports
- Run `pytest` from the repo root to hit: dataset formatting (`tests/test_dataset.py`), CPU/GPU offload logic (`tests/test_cpu_offload.py`), NSA kernels (`tests/test_nsa_attention.py`) tested against torch, and architecture forwards (`tests/test_model_forward.py`, `tests/test_qwen3next_components.py`) tested against torch.
- Triton kernels ship with focused tests under `model/qwen3next/kernels`, `train/attention/`, and `train/kernels` that are callable directly for kernel profiling.
- `plots/` captures throughput vs. latency charts, loss curves, and kernel benchmarks; the figures referenced in `reports/` are regenerated via the scripts in `scripts/`.
- `reports/` links the quantitative story together; expect kernel perf breakdowns, and inference latency sweeps tied back to the Triton experiments.

## Getting Started
1. Create a virtual environment and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   export PYTHONPATH="$(pwd):${PYTHONPATH}"
   ```
2. Select a CUDA-matched PyTorch build (`pip install torch==2.8.0+cu121` as of now) so Triton 3.x stays happy.
3. Optional: point `TOKENIZER_PATH` to a local cache of `Qwen/Qwen3-0.6B` if you are working offline.

## Troubleshooting
- **Triton compile hiccups** - check CUDA driver vs. PyTorch build (`nvidia-smi`, `python -m torch.utils.collect_env`). Re-run the kernel tests to warm the JIT cache.
- **Distributed launch issues** - confirm `torchrun --standalone --nnodes=1 --nproc_per_node=<gpus>` aligns with visible GPUs and that NCCL has peer-to-peer permissions.
- **Non-finite losses** - dial back LR, disable BF16 (`--precision fp32`), and rerun the tiny config before scaling out.

Released under MIT (see `LICENSE`).