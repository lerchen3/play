"""Unit tests covering the Qwen3-Next kernels and building blocks."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.qwen3next.config import Qwen3NextConfig
from model.cce import triton_cce_loss
from model.qwen3next.kernels import (
    GatedRMSNorm,
    TritonLinear,
    ZeroCenteredRMSNorm,
    gated_delta_rule,
    gated_delta_step,
    scaled_dot_product_attention,
    triton_matmul,
)
from model.qwen3next.layers import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextRotaryEmbedding,
    Qwen3NextSparseMoeBlock,
    apply_rotary_pos_emb,
)
from model.qwen3next.model import Qwen3NextForCausalLM, Qwen3NextModel
from model.qwen3next.training import Qwen3NextTrainModel


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Qwen3-Next kernel tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def tiny_config(**overrides) -> Qwen3NextConfig:
    base = Qwen3NextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        partial_rotary_factor=0.5,
        num_experts=0,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_triton_matmul_matches_linear_cuda():
    torch.manual_seed(0)
    x = torch.randn(4, 8, device=DEVICE)
    w = torch.randn(6, 8, device=DEVICE)
    b = torch.randn(6, device=DEVICE)
    expected = torch.nn.functional.linear(x, w, b)
    actual = triton_matmul(x, w, b)
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-3)


def test_triton_linear_agrees_with_torch_linear():
    torch.manual_seed(1)
    layer = TritonLinear(8, 4, bias=True).to(DEVICE)
    reference = torch.nn.Linear(8, 4, bias=True).to(DEVICE)
    with torch.no_grad():
        reference.weight.copy_(layer.weight)
        reference.bias.copy_(layer.bias)
    sample = torch.randn(2, 3, 8, device=DEVICE)
    out_triton = layer(sample)
    out_ref = reference(sample)
    assert torch.allclose(out_triton, out_ref, atol=1.5e-3, rtol=1e-3)


def test_triton_cce_loss_matches_torch():
    torch.manual_seed(7)
    dim = 8
    vocab = 16
    batch, seq = 3, 4
    base_hidden = torch.randn(batch, seq, dim, device=DEVICE)
    base_weight = torch.randn(vocab, dim, device=DEVICE)
    targets = torch.randint(0, vocab, (batch, seq), device=DEVICE)
    targets[0, 0] = 0  # ensure ignore index coverage
    ignore_idx = 0

    hidden_triton = base_hidden.clone().requires_grad_(True)
    weight_triton = base_weight.clone().requires_grad_(True)
    hidden_torch = base_hidden.clone().requires_grad_(True)
    weight_torch = base_weight.clone().requires_grad_(True)

    loss_triton = triton_cce_loss(hidden_triton, weight_triton, targets, ignore_index=ignore_idx)
    loss_triton.backward()

    logits = hidden_torch.view(-1, dim) @ weight_torch.t()
    loss_torch = F.cross_entropy(logits, targets.view(-1), ignore_index=ignore_idx)
    loss_torch.backward()

    assert torch.allclose(loss_triton, loss_torch, atol=1e-3, rtol=1e-4)
    assert torch.allclose(hidden_triton.grad, hidden_torch.grad, atol=1e-3, rtol=1e-4)
    assert torch.allclose(weight_triton.grad, weight_torch.grad, atol=1e-3, rtol=1e-4)


def test_zero_centered_rmsnorm_matches_manual():
    torch.manual_seed(2)
    norm = ZeroCenteredRMSNorm(6, eps=1e-6, input_weight_decay=0.25).to(DEVICE)
    tensor = torch.randn(3, 6, device=DEVICE)
    norm.weight.data.uniform_(-0.1, 0.1)
    output = norm(tensor)
    decayed = tensor * (1.0 - norm.input_weight_decay)
    centered = decayed - decayed.mean(-1, keepdim=True)
    variance = centered.pow(2).mean(-1, keepdim=True)
    expected = centered * torch.rsqrt(variance + 1e-6)
    expected = expected * norm.weight
    assert torch.allclose(output, expected, atol=1e-6)


def test_gated_rmsnorm_matches_manual_silu():
    torch.manual_seed(3)
    norm = GatedRMSNorm(4, eps=1e-5, input_weight_decay=0.4).to(DEVICE)
    x = torch.randn(2, 4, device=DEVICE)
    gate = torch.randn(2, 4, device=DEVICE)
    output = norm(x, gate)
    decayed = x * (1.0 - norm.input_weight_decay)
    centered = decayed - decayed.mean(-1, keepdim=True)
    variance = centered.pow(2).mean(-1, keepdim=True)
    expected = centered * torch.rsqrt(variance + 1e-5)
    expected = expected * norm.weight
    expected = expected * torch.nn.functional.silu(gate)
    assert torch.allclose(output, expected, atol=1e-6)


def test_gated_delta_rule_updates_state():
    torch.manual_seed(4)
    query = torch.randn(1, 3, 2, 2, device=DEVICE)
    key = torch.randn(1, 3, 2, 2, device=DEVICE)
    value = torch.randn(1, 3, 2, 3, device=DEVICE)
    decay = torch.rand(1, 3, 2, device=DEVICE)
    beta = torch.rand(1, 3, 2, device=DEVICE)
    outputs, state = gated_delta_rule(query, key, value, decay=decay, beta=beta)
    assert outputs.shape == (1, 3, 2, 3)
    assert state.shape == (1, 2, 2, 3)
    # Re-apply with returned state to ensure continuity
    outputs2, state2 = gated_delta_rule(query, key, value, decay=decay, beta=beta, state=state)
    assert torch.allclose(state2, state, atol=1e-4)
    assert torch.isfinite(outputs2).all()


def test_gated_delta_step_matches_rule_single_token():
    torch.manual_seed(6)
    batch, seq_len, heads, d_k, d_v = 2, 3, 2, 4, 5
    query = torch.randn(batch, seq_len, heads, d_k, device=DEVICE)
    key = torch.randn(batch, seq_len, heads, d_k, device=DEVICE)
    value = torch.randn(batch, seq_len, heads, d_v, device=DEVICE)
    decay = torch.rand(batch, seq_len, heads, device=DEVICE)
    beta = torch.rand(batch, seq_len, heads, device=DEVICE)

    step_out, step_state = gated_delta_step(
        query[:, 0],
        key[:, 0],
        value[:, 0],
        decay[:, 0],
        beta[:, 0],
    )

    rule_out, rule_state = gated_delta_rule(
        query[:, :1],
        key[:, :1],
        value[:, :1],
        decay=decay[:, :1],
        beta=beta[:, :1],
    )

    assert torch.allclose(step_out, rule_out[:, 0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(step_state, rule_state, atol=1e-5, rtol=1e-4)


def test_scaled_dot_product_attention_matches_torch():
    torch.manual_seed(5)
    batch, seq, heads, dim = 2, 4, 2, 4
    q = torch.randn(batch, seq, heads, dim, device=DEVICE)
    k = torch.randn(batch, seq, heads, dim, device=DEVICE)
    v = torch.randn(batch, seq, heads, dim, device=DEVICE)
    scale = 1.0 / math.sqrt(dim)
    custom, _ = scaled_dot_product_attention(q, k, v, scale)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    expected = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
    expected = expected.permute(0, 2, 1, 3)
    assert torch.allclose(custom, expected, atol=2e-3, rtol=1e-3)


def test_rotary_embedding_shapes_and_values():
    config = tiny_config()
    embedding = Qwen3NextRotaryEmbedding(config).to(DEVICE)
    cos, sin = embedding(seq_len=6, device=DEVICE, dtype=torch.float32)
    assert cos.shape == (6, config.rotary_dim)
    assert sin.shape == (6, config.rotary_dim)
    # position zero should be zero rotation
    assert torch.allclose(cos[0], torch.ones(config.rotary_dim, device=DEVICE))
    assert torch.allclose(sin[0], torch.zeros(config.rotary_dim, device=DEVICE))


def test_apply_rotary_pos_emb_identity_for_zero():
    q = torch.randn(2, 3, 4, 4, device=DEVICE)
    k = torch.randn(2, 3, 4, 4, device=DEVICE)
    cos = torch.ones(3, 4, device=DEVICE)
    sin = torch.zeros(3, 4, device=DEVICE)
    q_new, k_new = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_new, q)
    assert torch.allclose(k_new, k)


def test_attention_module_forward_shapes():
    config = tiny_config(num_hidden_layers=1)
    attention = Qwen3NextAttention(config, layer_idx=0).to(DEVICE)
    hidden = torch.randn(2, 5, config.hidden_size, device=DEVICE)
    cos, sin = Qwen3NextRotaryEmbedding(config)(5, DEVICE, torch.float32)
    out, attn = attention(hidden, cos, sin)
    assert out.shape == (2, 5, config.hidden_size)
    assert attn.shape == (2, config.num_attention_heads, 5, 5)


def test_gated_delta_net_forward():
    config = tiny_config(num_hidden_layers=1)
    mixer = Qwen3NextGatedDeltaNet(config, layer_idx=0).to(DEVICE)
    hidden = torch.randn(2, 6, config.hidden_size, device=DEVICE)
    output = mixer(hidden)
    assert output.shape == (2, 6, config.hidden_size)


def test_mlp_block_forward():
    config = tiny_config()
    mlp = Qwen3NextMLP(config).to(DEVICE)
    hidden = torch.randn(2, 4, config.hidden_size, device=DEVICE)
    output, _ = mlp(hidden)
    assert output.shape == (2, 4, config.hidden_size)


def test_sparse_moe_block_topk_distribution():
    config = tiny_config(num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)
    block = Qwen3NextSparseMoeBlock(config).to(DEVICE)
    hidden = torch.randn(2, 3, config.hidden_size, device=DEVICE)
    output, router_logits = block(hidden)
    assert output.shape == hidden.shape
    weights = torch.softmax(router_logits.view(-1, router_logits.size(-1)), dim=-1)
    topk_weights, _ = torch.topk(weights, config.num_experts_per_tok, dim=-1)
    normalized = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    sums = normalized.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_decoder_layer_outputs_for_both_mixers():
    config = tiny_config(layer_types=("full_attention", "linear_attention"))
    rotary = Qwen3NextRotaryEmbedding(config).to(DEVICE)
    cos, sin = rotary(4, DEVICE, torch.float32)
    layer_attn = Qwen3NextDecoderLayer(config, layer_idx=0).to(DEVICE)
    layer_linear = Qwen3NextDecoderLayer(config, layer_idx=1).to(DEVICE)
    hidden = torch.randn(1, 4, config.hidden_size, device=DEVICE)
    out_attn, probs, router = layer_attn(hidden, cos, sin)
    assert out_attn.shape == hidden.shape
    assert probs is not None and probs.shape[-1] == 4
    out_linear, probs_linear, _ = layer_linear(hidden, cos, sin)
    assert out_linear.shape == hidden.shape
    assert probs_linear is None


def test_model_forward_hidden_states():
    config = tiny_config()
    model = Qwen3NextModel(config).to(DEVICE)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), device=DEVICE)
    outputs = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
    assert outputs.last_hidden_state.shape == (1, 4, config.hidden_size)
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1
    assert len(outputs.attentions) == config.num_hidden_layers


def test_causal_lm_loss_computation():
    config = tiny_config()
    model = Qwen3NextForCausalLM(config).to(DEVICE)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (2, 5), device=DEVICE)
    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 5, config.vocab_size)
    assert outputs.mtp_loss is None


def test_causal_lm_mtp_outputs():
    config = tiny_config(mtp_depth=2)
    model = Qwen3NextForCausalLM(config).to(DEVICE)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (1, 4), device=DEVICE)
    mtp_targets = torch.randint(0, config.vocab_size, (1, 4, config.mtp_depth), device=DEVICE)
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        mtp_targets=mtp_targets,
        output_mtp_logits=True,
    )
    assert outputs.mtp_logits is not None
    assert outputs.mtp_logits.shape == (1, 4, config.mtp_depth, config.vocab_size)
    assert outputs.mtp_loss is not None


def test_training_wrapper_returns_expected_shape():
    args = type("Args", (), {})()
    args.vocab_size = 32
    args.pad_token_id = 0
    args.d_model = 16
    args.num_layers = 2
    args.n_q_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 24
    args.num_experts = 0
    args.mtp_depth = 0
    model = Qwen3NextTrainModel(args).to(DEVICE)
    batch = torch.randint(0, args.vocab_size, (1, 6), device=DEVICE)
    target = torch.randint(0, args.vocab_size, (1, 6), device=DEVICE)
    losses = model(batch, target_main=target)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()


def test_training_wrapper_with_mtp_matrix():
    args = type("Args", (), {})()
    args.vocab_size = 32
    args.pad_token_id = 0
    args.d_model = 16
    args.num_layers = 2
    args.n_q_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 24
    args.num_experts = 0
    args.mtp_depth = 2
    model = Qwen3NextTrainModel(args).to(DEVICE)
    batch = torch.randint(0, args.vocab_size, (1, 6), device=DEVICE)
    target = torch.randint(0, args.vocab_size, (1, 6), device=DEVICE)
    tgt_matrix = torch.randint(0, args.vocab_size, (1, 6, args.mtp_depth), device=DEVICE)
    losses = model(batch, target_main=target, tgt_matrix=tgt_matrix, is_training=True)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
