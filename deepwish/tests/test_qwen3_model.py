import argparse

import pytest
import torch

from model.qwen3 import Qwen3Model


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Qwen3 tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def make_args(use_nsa: bool = False, seq_len: int = 16):
    args = argparse.Namespace()
    args.vocab_size = 32
    args.pad_token_id = 0
    args.d_model = 128
    args.num_layers = 1
    args.n_q_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 256
    args.seq_len = seq_len
    args.rmsnorm_eps = 1e-6
    args.use_nsa = use_nsa
    args.window_size = None
    args.nsa_cmp_blk_size = max(16, seq_len // 2)
    args.nsa_cmp_stride = max(8, seq_len // 4)
    args.nsa_slc_top_n = max(4, seq_len // 4)
    return args


def test_qwen3_forward_and_decode_no_nsa():
    torch.manual_seed(0)
    args = make_args(use_nsa=False)
    model = Qwen3Model(args).to(DEVICE)

    input_ids = torch.randint(0, args.vocab_size, (2, args.seq_len), device=DEVICE)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = args.pad_token_id

    loss = model(input_ids, target_main=targets, is_training=True)
    assert loss.ndim == 0 and torch.isfinite(loss)

    model.reset_kv_cache()
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, is_training=False)
    assert logits.shape == (2, args.seq_len, args.vocab_size)
    assert torch.isfinite(logits).all()

    model.reset_kv_cache()
    model.prefill_with_cache(input_ids)
    step_logits = model.step_with_cache(input_ids[:, -1:], return_all_logits=False)
    assert step_logits.shape == (2, args.vocab_size)
    assert torch.isfinite(step_logits).all()


def test_qwen3_forward_and_decode_with_nsa():
    torch.manual_seed(1)
    args = make_args(use_nsa=True)
    model_train = Qwen3Model(args).to(DEVICE)

    input_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device=DEVICE)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = args.pad_token_id

    loss = model_train(input_ids, target_main=targets, is_training=True)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # fresh copy for inference/cache checks
    torch.manual_seed(1)
    model = Qwen3Model(args).to(DEVICE)
    model.eval()
    infer_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device=DEVICE)
    with torch.no_grad():
        logits = model(infer_ids, is_training=False)
    assert logits.shape == (1, args.seq_len, args.vocab_size)
    assert torch.isfinite(logits).all()

    model.reset_kv_cache()
    cache_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device=DEVICE)
    model.prefill_with_cache(cache_ids)
    first_layer = model.layers[0]
    assert hasattr(first_layer.attn, "_nsa")
    assert first_layer.attn._nsa._cached_len == args.seq_len

    step_token = torch.randint(0, args.vocab_size, (1, 1), device=DEVICE)
    step_logits = model.step_with_cache(step_token)
    assert step_logits.shape == (1, args.vocab_size)
    assert torch.isfinite(step_logits).all()
    assert first_layer.attn._nsa._cached_len == args.seq_len + 1
