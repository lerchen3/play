import torch
import pytest

from .topk import topk_indices


def torch_topk(q_group, k_full, sm_scale, top_k, row_max=None):
    B, G, T, D = q_group.shape
    N = k_full.shape[2]
    scores = torch.einsum('bgtd,bgnd->bgtn', q_group, k_full) * sm_scale
    if row_max is not None:
        idx = torch.arange(N, device=scores.device).view(1, 1, 1, N)
        cap = row_max.view(B, G, T, 1)
        scores = scores.masked_fill(idx > cap, float('-inf'))
    return torch.topk(scores, k=top_k, dim=-1).indices.to(torch.int32)


@pytest.mark.parametrize('B,G,T,D,N,top_k', [(2, 3, 17, 64, 127, 8), (1, 2, 64, 32, 256, 16)])
def test_topk_vs_torch(B, G, T, D, N, top_k):
    torch.manual_seed(0)
    device = 'cuda'
    q_group = torch.randn(B, G, T, D, device=device, dtype=torch.float32)
    k_full = torch.randn(B, G, N, D, device=device, dtype=torch.float32)
    sm = 0.1
    # causal caps per row
    rmax = torch.randint(low=0, high=N, size=(B, G, T), device=device, dtype=torch.int32)
    tri_idx = topk_indices(q_group, k_full, sm, top_k, rmax)
    torch_idx = torch_topk(q_group, k_full, sm, top_k, rmax)
    # Allow set equality ignoring order
    tri_sorted, _ = torch.sort(tri_idx, dim=-1)
    torch_sorted, _ = torch.sort(torch_idx, dim=-1)
    assert torch.equal(tri_sorted, torch_sorted)


