import pytest
import torch

from train.attention.nsa import NativeSparseAttention


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for NSA tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def made_up_args():
    return dict(
        d_model=128,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=32,
        seq_len=32,
        cmp_blk_size=16,
        cmp_stride=8,
        slc_top_n=4,
        window_size=16,
    )


def test_nsa_branch_outputs_are_distinct():
    torch.manual_seed(0)
    cfg = made_up_args()
    nsa = NativeSparseAttention(**cfg).to(DEVICE)

    B, T = 2, 16
    x = torch.randn(B, T, cfg["d_model"], device=DEVICE)
    q = torch.randn(B, cfg["n_q_heads"], T, cfg["d_head"], device=DEVICE)

    out, (cmp, slc, win) = nsa(x, q, return_components=True)

    assert out.shape == (B, T, cfg["n_q_heads"] * cfg["d_head"])
    for branch in (cmp, slc, win):
        assert torch.isfinite(branch).all()

    # Each branch should contribute differently – pairwise L2 norms must be non-zero
    eps = 1e-4
    assert torch.linalg.norm(cmp - slc) > eps
    assert torch.linalg.norm(cmp - win) > eps
    assert torch.linalg.norm(slc - win) > eps


def test_nsa_decode_path_updates_caches_and_is_finite():
    torch.manual_seed(1)
    cfg = made_up_args()
    nsa = NativeSparseAttention(**cfg).to(DEVICE)

    B, T = 1, 12
    x = torch.randn(B, T, cfg["d_model"], device=DEVICE)
    q = torch.randn(B, cfg["n_q_heads"], T, cfg["d_head"], device=DEVICE)

    nsa.prefill(x, q)
    assert nsa._cached_len == T

    step_x = torch.randn(B, 1, cfg["d_model"], device=DEVICE)
    step_q = torch.randn(B, cfg["n_q_heads"], 1, cfg["d_head"], device=DEVICE)

    out, components = nsa(step_x, step_q, is_decoding=True, return_components=True)

    assert out.shape == (B, 1, cfg["n_q_heads"] * cfg["d_head"])
    for branch in components:
        assert torch.isfinite(branch).all()
    assert torch.isfinite(out).all()
    assert nsa._cached_len == T + 1
