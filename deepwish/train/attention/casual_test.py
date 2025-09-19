import torch
import pytest

from .casual import _attention as causal_attention


def torch_causal(q, k, v, sm_scale):
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(p, v)


@pytest.mark.parametrize('B,H,T,D', [(2, 4, 64, 64), (1, 8, 128, 32)])
def test_causal_matches_torch(B, H, T, D):
    torch.manual_seed(0)
    device = 'cuda'
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm = 0.5
    ref = torch_causal(q, k, v, sm)
    out = causal_attention.apply(q, k, v, sm)
    assert torch.allclose(ref, out, atol=1e-2, rtol=0)

