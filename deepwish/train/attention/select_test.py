import torch
import pytest

from .select import select_attention


@pytest.mark.parametrize('B,G,T,D,cmp_blk_size,top_k', [(1, 2, 32, 64, 4, 3), (2, 3, 65, 32, 8, 5)])
def test_select_indices_vs_torch(B, G, T, D, cmp_blk_size, top_k):
    torch.manual_seed(0)
    device = 'cuda'
    q = torch.randn(B, G, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, G, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, G, T, D, device=device, dtype=torch.float32)
    sm = 0.2
    # build synthetic block indices per row using random selection
    nb = (T + cmp_blk_size - 1) // cmp_blk_size
    Kmax = min(nb, top_k + 3)
    block_idx = torch.randint(low=0, high=nb, size=(B, G, T, Kmax), device=device, dtype=torch.int32)
    block_count = torch.randint(low=1, high=Kmax + 1, size=(B, G, T), device=device, dtype=torch.int32)

    out = select_attention(q, k, v, sm, block_idx, block_count, cmp_blk_size)
    # brute reference: for each row, gather tokens from blocks <= t
    ref = torch.zeros_like(q)
    for b in range(B):
        for g in range(G):
            for t in range(T):
                kset = set()
                cnt = int(block_count[b, g, t].item())
                for i in range(cnt):
                    bi = int(block_idx[b, g, t, i].item())
                    s = bi * cmp_blk_size
                    e = min(T, s + cmp_blk_size)
                    for jj in range(s, e):
                        if jj <= t:
                            kset.add(jj)
                if len(kset) == 0:
                    continue
                ids = torch.tensor(sorted(list(kset)), device=device)
                qv = q[b, g, t]
                Ksel = k[b, g, ids]
                Vsel = v[b, g, ids]
                scores = (Ksel @ qv) * sm
                probs = torch.softmax(scores, dim=0).to(Vsel.dtype)
                ref[b, g, t] = probs @ Vsel
    assert torch.allclose(ref, out, atol=2e-2, rtol=0)


