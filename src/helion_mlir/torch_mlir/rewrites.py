from __future__ import annotations

from ..torch_mlir_helper import (
    _rewrite_batch_matmul_f16_accumulation as rewrite_batch_matmul_f16_accumulation,
    _rewrite_linalg_generic_scalars as rewrite_linalg_generic_scalars,
    _rewrite_amax_keepdim_from_expand_shape as rewrite_amax_keepdim_from_expand_shape,
)

__all__ = [
    "rewrite_batch_matmul_f16_accumulation",
    "rewrite_linalg_generic_scalars",
    "rewrite_amax_keepdim_from_expand_shape",
]
