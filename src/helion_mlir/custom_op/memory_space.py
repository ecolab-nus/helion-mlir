"""Tracing-only tensor memory-space annotation."""

from __future__ import annotations

import torch
from torch.fx import has_side_effect

from helion import exc
from helion.language import _decorators


@has_side_effect
@_decorators.api(allow_host_tensor=False)
def set_memory_space(tensor: torch.Tensor, mem_space: int) -> torch.Tensor:
    """Annotate a kernel tensor with an integer MLIR tensor encoding."""
    raise exc.NotInsideKernel


@_decorators.register_fake(set_memory_space)
def _(tensor: torch.Tensor, mem_space: int) -> torch.Tensor:
    _ = mem_space
    return tensor.new_empty(tuple(tensor.shape))
