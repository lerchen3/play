try:
    from .cce_triton import TritonCCE as _TritonCCE
    from .cce_triton import triton_cut_cross_entropy as _triton_cut_cross_entropy
except Exception as exc:  # pragma: no cover - Triton should always be present
    raise RuntimeError("Triton CCE kernels are required but failed to import") from exc
else:
    TritonCCE = _TritonCCE
    triton_cut_cross_entropy = _triton_cut_cross_entropy


__all__ = ["TritonCCE", "triton_cut_cross_entropy"]
