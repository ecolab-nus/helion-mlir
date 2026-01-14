"""MLIR lowerings for PyTorch ATen operations.

All ATen operations are now handled generically by ir_visitor.py using torch-mlir
to generate torch dialect MLIR. This module is kept for possible future custom
lowerings but currently has no registrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx
from torch.ops import aten

from ..op_registry import register_lowering
from .base import MLIRLowering
from ..torch_mlir_helper import import_aten_node_to_mlir, inline_torch_mlir_output

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


# All ATen operations now use torch-mlir's torch dialect output.
# Custom lowerings can be added here if needed for specific optimizations.
