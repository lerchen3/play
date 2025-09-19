"""Tests exercising DeepSeek-V3 kernels and transformer building blocks."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from model.deepseekv3 import DeepSeekV3Model, triton_cce_loss
from .base_attention import attention
from .gqa import GroupedQueryAttention
from .mla import MLA
from .moe import DeepSeekMoE
from .mtp import MTP
from .transformer import TransformerBlock


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for DeepSeek component tests", allow_module_level=True)

DEVICE = torch.device("cuda")


class DummyArgs:
    def __init__(self):
        self.vocab_size = 32
        self.pad_token_id = 0
        self.d_model = 128
        self.seq_len = 16
        self.n_heads = 4
        self.dc_kv = 32
        self.dc_q = 32
        self.num_layers = 2
        self.rmsnorm_eps = 1e-6
        self.n_shared_experts = 1
        self.n_routed_experts = 2
        self.k_routed_experts = 1
        self.bias_update_speed = 0.001
        self.moe_balance_factor = 0.01
        self.d_ff_expert_mult = 2
        self.d_ff_expert = self.d_model * self.d_ff_expert_mult
        self.mtp_depth = 1
        self.device = DEVICE
        self.d_head = self.d_model // self.n_heads
        self.num_kv_heads = self.n_heads
        self.use_nsa = False
        self.window_size = None


def test_triton_cce_loss_matches_cross_entropy():
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 4, device=DEVICE, requires_grad=True)
    weight = torch.randn(5, 4, device=DEVICE, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [3, 4, 1]], device=DEVICE)
    loss = triton_cce_loss(hidden, weight, targets, ignore_index=0)
    logits = hidden.view(-1, 4) @ weight.t()
    ref_loss = F.cross_entropy(logits, targets.view(-1), ignore_index=0)
    assert torch.allclose(loss, ref_loss, atol=1e-6)


def test_attention_fallback_matches_manual():
    torch.manual_seed(1)
    q = (0.1 * torch.randn(2, 3, 4, 32, device=DEVICE, dtype=torch.float32)).requires_grad_(True)
    k = (0.1 * torch.randn(2, 3, 4, 32, device=DEVICE, dtype=torch.float32)).requires_grad_(True)
    v = (0.1 * torch.randn(2, 3, 4, 32, device=DEVICE, dtype=torch.float32)).requires_grad_(True)
    scale = (32) ** -0.5
    out = attention(q, k, v, scale)
    assert torch.isfinite(out).all()


def test_grouped_query_attention_cache_roundtrip():
    torch.manual_seed(2)
    module = GroupedQueryAttention(d_model=128, n_q_heads=4, n_kv_heads=2, seq_len=8).to(DEVICE)
    hidden = torch.randn(1, 4, 128, device=DEVICE)
    out = module(hidden)
    assert out.shape == hidden.shape
    cached = module(hidden, return_cache=True)
    assert isinstance(cached, tuple) and len(cached) == 3


def test_mla_handles_cache():
    torch.manual_seed(3)
    module = MLA(128, 4, dc_kv=32, dc_q=32, seq_len=8, device=DEVICE).to(DEVICE)
    hidden = torch.randn(1, 3, 128, device=DEVICE)
    output, c_kv, kR = module(hidden, return_latent=True)
    assert output.shape == hidden.shape
    assert c_kv.shape[1] == hidden.size(1)
    step = torch.randn(1, 1, 128, device=DEVICE)
    out_step = module(step, cached_c_kv=c_kv, cached_kR=kR, cached_len=hidden.size(1))
    assert out_step.shape == step.shape


def test_deepseek_moe_gating_distribution():
    torch.manual_seed(4)
    moe = DeepSeekMoE(d_model=128, ns=1, nr=3, kr=2, bias_sp=0.01, bal=0.01, d_ff=256, device=DEVICE).to(DEVICE)
    hidden = torch.randn(2, 3, 128, device=DEVICE)
    output = moe(hidden, training=True)
    assert output.shape == hidden.shape
    assert hasattr(moe, "last_expert_usage")


def test_transformer_block_variants():
    args = DummyArgs()
    block = TransformerBlock(args).to(DEVICE)
    hidden = torch.randn(1, 4, args.d_model, device=DEVICE)
    out = block(hidden)
    assert out.shape == hidden.shape
    if importlib.util.find_spec("triton.language") is None:
        pytest.skip("NSA attention requires Triton kernels")
    args_nsa = DummyArgs()
    args_nsa.use_nsa = True
    args_nsa.num_kv_heads = 2
    block_nsa = TransformerBlock(args_nsa).to(DEVICE)
    out_nsa = block_nsa(hidden)
    assert out_nsa.shape == hidden.shape


def test_mtp_cache_workflow():
    args = DummyArgs()
    mtp = MTP(depth=1, args=args).to(DEVICE)
    hidden = torch.randn(1, 3, args.d_model, device=DEVICE)
    stacked = mtp(hidden)
    assert stacked.shape == (1, 3, 1, args.d_model)
    mtp.prefill_with_cache(hidden)
    step = torch.randn(1, 1, args.d_model, device=DEVICE)
    out_step = mtp.step_with_cache_hidden(step)
    assert out_step.shape == (1, 1, args.d_model)


def test_deepseek_model_end_to_end():
    args = DummyArgs()
    model = DeepSeekV3Model(args).to(DEVICE)
    batch = torch.randint(0, args.vocab_size, (1, args.seq_len), device=DEVICE)
    targets = torch.randint(0, args.vocab_size, (1, args.seq_len), device=DEVICE)
    losses = model(batch, target_main=targets, is_training=True)
    assert losses.shape == (2,)
    model.prefill_with_cache(batch)
    next_logits = model.step_with_cache(batch[:, :1])
    assert next_logits.shape == (1, args.vocab_size)
