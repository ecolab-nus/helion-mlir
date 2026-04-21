"""Custom broadcast op registered with Helion's decorator API.

broadcast(src, dim, out_shape) returns a tensor with ``out_shape``.
This op is a minimal spike for MLIR lowering verification only.
Triton codegen, syntax checks, and verifier logic are intentionally omitted.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.fx import has_side_effect

from helion import exc
from helion.language import _decorators


@has_side_effect
@_decorators.api(allow_host_tensor=False, tiles_as_sizes=True)
def broadcast(
    src: torch.Tensor,
    dim: int,
    out_shape: Sequence[int | torch.SymInt],
) -> torch.Tensor:
    """Broadcast ``src`` to ``out_shape``.

    Args:
        src: Input tensor.
        dim: Broadcast axis with respect to output tensor (forwarded to MLIR).
        out_shape: Output shape specification.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(broadcast)
def _(src: torch.Tensor, dim: int, out_shape: Sequence[int | torch.SymInt]) -> torch.Tensor:
    # Minimal fake implementation: shape is entirely dictated by `out_shape`.
    _ = dim
    return src.new_empty(tuple(out_shape))
