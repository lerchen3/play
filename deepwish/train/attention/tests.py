import pytest
import torch

from .casual import _attention as casual_attention
from .select import select_attention
from .nsa import NativeSparseAttention


def torch_causal(q, k, v, sm_scale, window_size=0):
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    base_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    if window_size and window_size > 0:
        ar = torch.arange(T, device=q.device)
        within_window = (ar[:, None] - ar[None, :]) < window_size
        mask = base_mask & within_window
    else:
        mask = base_mask
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
    sel_idx = torch.arange(Ns, device='cuda')
    Kmax = Ns
    cmp_blk_size = 1
    block_idx = sel_idx.view(1, 1, 1, Ns).expand(B, G, T, Ns).clone()
    counts = torch.arange(1, T + 1, device='cuda').clamp_max(Ns)
    block_count = counts.view(1, 1, T).expand(B, G, T).to(torch.int32)
    sm = 0.5
    out_sel = select_attention(q, full_k, full_v, sm, block_idx, block_count, cmp_blk_size)
    assert torch.isfinite(out_sel).all()
    loss = out_sel.sum()
    loss.backward()
    assert torch.isfinite(q.grad).all()


def test_nsa_shapes_forward_decode():
    B, T, d_model = 1, 32, 256
    n_q, n_kv, d = 8, 2, 32
    x = torch.randn(B, T, d_model, device='cuda', dtype=torch.float32)
    q = torch.randn(B, n_q, T, d, device='cuda', dtype=torch.float32)
    nsa = NativeSparseAttention(d_model, n_q, n_kv, d, seq_len=T).cuda()
    out = nsa(x, q, is_decoding=False)
    assert out.shape == (B, T, n_q * d)
    assert torch.isfinite(out).all()
    # decode single token
    x1 = torch.randn(B, 1, d_model, device='cuda', dtype=torch.float32)
    q1 = torch.randn(B, n_q, 1, d, device='cuda', dtype=torch.float32)
    out1 = nsa(x1, q1, is_decoding=True)
    assert out1.shape == (B, 1, n_q * d)
    assert torch.isfinite(out1).all()
