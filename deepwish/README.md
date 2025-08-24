Building SOTA language model fluency.

- Architecture taken from Deepseek-v3, outlined in https://arxiv.org/pdf/2412.19437, in `train/`

- Hand-crafted gpu kernels writen in OpenAI's triton library in `triton/`

- Smart distributed CPU offloading code written in `train/offload_step.py`

- Vanilla inference and MTP-based speculative decoding with compressed KV cache in `inference/`

Plots in `plots/` taken from first 11.5 hours of training.

This project started as an attempt to answer the question "What if we trained an LLM from scratch solely from Deepseek-R1 reasoning traces?" Currently awaiting two orders of magnitude more GPU resources to properly answer this question (with a larger model), though with our 300M FP32 param model we do currently answer 1+1 correctly :D

Pending: mixed precision / fp8 kernels and training.