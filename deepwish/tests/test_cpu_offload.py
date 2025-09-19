"""Ensure forward/backward passes succeed on CPU for all major models."""

from __future__ import annotations

import argparse

import torch

from model.deepseekv3 import DeepSeekV3Model
from model.qwen3 import Qwen3Model
from model.qwen3next.training import Qwen3NextTrainModel


def _make_deepseek_args(device: torch.device) -> argparse.Namespace:
    args = argparse.Namespace()
    args.vocab_size = 32
    args.pad_token_id = 0
    args.d_model = 48
    args.seq_len = 16
    args.n_heads = 4
    args.dc_kv = 12
    args.dc_q = 12
    args.num_layers = 2
    args.rmsnorm_eps = 1e-6
    args.n_shared_experts = 1
    args.n_routed_experts = 2
    args.k_routed_experts = 1
    args.bias_update_speed = 0.001
    args.moe_balance_factor = 0.01
    args.d_ff_expert_mult = 2
    args.d_ff_expert = args.d_model * args.d_ff_expert_mult
    args.mtp_depth = 0
    args.device = device
    args.d_head = 12
    args.num_kv_heads = args.n_heads
    args.use_nsa = False
    args.window_size = None
    return args


def test_deepseek_cpu_forward_backward():
    torch.manual_seed(0)
    device = torch.device("cpu")
    args = _make_deepseek_args(device)
    model = DeepSeekV3Model(args)

    input_ids = torch.randint(0, args.vocab_size, (2, args.seq_len), device=device)
    input_ids[:, -1] = args.pad_token_id
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = args.pad_token_id

    losses = model(input_ids, target_main=targets, is_training=True)
    assert losses.shape == (2,)
    loss = losses.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    grads = [g for g in grads if g is not None]
    assert grads, "Expected gradients for DeepSeek parameters on CPU"
    assert all(torch.isfinite(g).all() for g in grads)

    model.zero_grad(set_to_none=True)
    model.prefill_with_cache(input_ids)
    step_logits = model.step_with_cache(input_ids[:, -1:], return_all_logits=False)
    assert step_logits.shape == (2, args.vocab_size)
    assert step_logits.device.type == "cpu"


def _make_qwen3_args(seq_len: int = 12) -> argparse.Namespace:
    args = argparse.Namespace()
    args.vocab_size = 40
    args.pad_token_id = 0
    args.d_model = 48
    args.num_layers = 2
    args.n_q_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 96
    args.seq_len = seq_len
    args.rmsnorm_eps = 1e-6
    args.use_nsa = False
    args.window_size = None
    args.nsa_cmp_blk_size = 16
    args.nsa_cmp_stride = 8
    args.nsa_slc_top_n = 4
    return args


def test_qwen3_cpu_forward_backward():
    torch.manual_seed(1)
    args = _make_qwen3_args()
    model = Qwen3Model(args)

    input_ids = torch.randint(0, args.vocab_size, (2, args.seq_len))
    targets = torch.roll(input_ids.clone(), shifts=-1, dims=1)
    targets[:, -1] = args.pad_token_id

    loss = model(input_ids, target_main=targets, is_training=True)
    assert loss.ndim == 0
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    grads = [g for g in grads if g is not None]
    assert grads, "Expected gradients for Qwen3 parameters on CPU"
    assert all(torch.isfinite(g).all() for g in grads)

    model.zero_grad(set_to_none=True)
    model.reset_kv_cache()
    model.prefill_with_cache(input_ids)
    step_logits = model.step_with_cache(input_ids[:, -1:], return_all_logits=False)
    assert step_logits.shape == (2, args.vocab_size)
    assert step_logits.device.type == "cpu"


def _make_qwen3next_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.vocab_size = 48
    args.pad_token_id = 0
    args.d_model = 64
    args.d_ff = 128
    args.num_layers = 2
    args.n_q_heads = 4
    args.n_kv_heads = 2
    args.linear_num_key_heads = 2
    args.linear_num_value_heads = 4
    args.linear_key_head_dim = 16
    args.linear_value_head_dim = 16
    args.linear_conv_kernel_dim = 2
    args.mtp_depth = 0
    args.num_experts = 0
    args.num_experts_per_tok = 2
    args.decoder_sparse_step = 1
    return args


def test_qwen3next_cpu_forward_backward():
    torch.manual_seed(2)
    args = _make_qwen3next_args()
    model = Qwen3NextTrainModel(args)

    seq_len = 10
    input_ids = torch.randint(0, args.vocab_size, (2, seq_len))
    targets = torch.roll(input_ids.clone(), shifts=-1, dims=1)
    targets[:, -1] = args.pad_token_id

    losses = model(input_ids, target_main=targets, is_training=True)
    assert losses.shape == (2,)
    total_loss = losses.sum()
    total_loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    grads = [g for g in grads if g is not None]
    assert grads, "Expected gradients for Qwen3-Next parameters on CPU"
    assert all(torch.isfinite(g).all() for g in grads)
