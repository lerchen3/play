import torch
import pytest

from .casual import _attention as casual_attention
from .select import select_attention
from .nsa import NativeSparseAttention


def torch_causal(q, k, v, sm_scale, window_size=0):
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    ar = torch.arange(T, device=q.device)
    mask = ar[None, :] < ar[:, None]
    if window_size and window_size > 0:
        mask &= (ar[:, None] - ar[None, :]) < window_size
    scores = scores.masked_fill(~mask, float('-inf'))
    p = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(p, v)


@pytest.mark.parametrize('B,H,T,D,win', [(2, 4, 64, 32, 0), (1, 8, 128, 64, 32)])
def test_causal_matches_torch(B, H, T, D, win):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm = 0.5
    ref = torch_causal(q, k, v, sm, win)
    out = casual_attention.apply(q, k, v, sm, win)
    assert torch.allclose(ref, out, atol=1e-2, rtol=0)


def test_select_matches_dense_small():
    B, G, T, D = 1, 2, 16, 32
    Ns = 8
    torch.manual_seed(0)
    q = torch.randn(B, G, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    # Build selected indices and dense K/V for comparison
    full_k = torch.randn(B, G, T, D, device='cuda', dtype=torch.float32)
    full_v = torch.randn(B, G, T, D, device='cuda', dtype=torch.float32)
    sel_idx = torch.randperm(T, device='cuda')[:Ns]
    k = full_k[:, :, sel_idx]
    v = full_v[:, :, sel_idx]
    sm = 0.5
    out_sel = select_attention(q, k, v, sm)
    # Dense reference over selected positions only
    scores = torch.einsum('bgtd,bgnd->bgtn', q, k) * sm
    p = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    ref = torch.einsum('bgtn,bgnd->bgtd', p, v)
    assert torch.allclose(ref, out_sel, atol=1e-2, rtol=0)


def test_nsa_shapes_forward_decode():
    B, T, d_model = 1, 32, 128
    n_q, n_kv, d = 8, 2, 16
    x = torch.randn(B, T, d_model, device='cuda', dtype=torch.float32)
    q = torch.randn(B, n_q, T, d, device='cuda', dtype=torch.float32)
    nsa = NativeSparseAttention(d_model, n_q, n_kv, d, seq_len=T).cuda()
    out = nsa(x, q, is_decoding=False)
    assert out.shape == (B, T, n_q * d)
    # decode single token
    x1 = torch.randn(B, 1, d_model, device='cuda', dtype=torch.float32)
    q1 = torch.randn(B, n_q, 1, d, device='cuda', dtype=torch.float32)
    out1 = nsa(x1, q1, is_decoding=True)
    assert out1.shape == (B, 1, n_q * d)


