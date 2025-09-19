import argparse
import argparse

import torch

from model.deepseekv3 import DeepSeekV3Model


def make_args():
    args = argparse.Namespace()
    args.vocab_size = 32
    args.pad_token_id = 0
    args.d_model = 64
    args.seq_len = 32
    args.n_heads = 4
    args.dc_kv = 16
    args.dc_q = 16
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
    args.device = torch.device('cuda')
    args.d_head = 16
    args.num_kv_heads = args.n_heads
    args.use_nsa = False
    args.window_size = None
    return args


def test_deepseek_forward_and_cache():
    args = make_args()
    model = DeepSeekV3Model(args).cuda()

    input_ids = torch.randint(0, args.vocab_size, (2, args.seq_len), device='cuda')
    input_ids[:, -1] = args.pad_token_id

    target = torch.roll(input_ids, shifts=-1, dims=1)
    target[:, -1] = args.pad_token_id

    losses = model(input_ids, target_main=target, is_training=True)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()

    # Prefill + single step should run on CPU
    model.prefill_with_cache(input_ids)
    step_ids = torch.randint(0, args.vocab_size, (2, 1), device='cuda')
    logits = model.step_with_cache(step_ids)
    assert logits.shape == (2, args.vocab_size)
    assert torch.isfinite(logits).all()
