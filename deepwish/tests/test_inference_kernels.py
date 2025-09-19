import pytest
import torch

from model.decode_triton import single_query_attention
from model.qwen3next.kernels.deltanet import gated_delta_rule, gated_delta_step


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inference kernel tests")
def test_single_query_attention_matches_torch():
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, D = 2, 4, 256, 64
    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    out = single_query_attention(q, k, v, scale)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, vf)

    assert torch.allclose(out.to(torch.float32), ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inference kernel tests")
def test_gated_delta_step_matches_rule():
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T, H, D_K, D_V = 2, 1, 4, 16, 32

    query = torch.randn(B, T, H, D_K, device=device, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn(B, T, H, D_V, device=device, dtype=torch.float16)
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float16))
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float16))

    full_out, final_state = gated_delta_rule(query, key, value, decay=decay, beta=beta)
    step_out, step_state = gated_delta_step(query[:, 0], key[:, 0], value[:, 0], decay[:, 0], beta[:, 0])

    assert torch.allclose(full_out[:, 0].to(torch.float32), step_out.to(torch.float32), atol=1e-3, rtol=1e-3)
    assert torch.allclose(final_state.to(torch.float32), step_state.to(torch.float32), atol=1e-3, rtol=1e-3)
