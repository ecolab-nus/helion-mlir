"""MLIR lowerings for PyTorch ATen operations.

Most ATen operations are now handled generically by ir_visitor.py using torch-mlir.
This module contains only specialized lowerings that need custom handling (e.g. for optimization).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.fx
from torch.ops import aten

from ..op_registry import register_lowering
from .base import MLIRLowering
from ..torch_mlir_helper import import_aten_node_to_mlir, inline_torch_mlir_output

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


@register_lowering(aten.addmm.default)
class AddmmLowering(MLIRLowering):
    """Lowering for aten.addmm (matrix multiply with add) -> linalg.matmul.
    
    aten.addmm(bias, mat1, mat2) computes: bias + mat1 @ mat2
    linalg.matmul accumulates into output: out = out + mat1 @ mat2
    
    So we use bias as the output tensor, and linalg.matmul will accumulate into it.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        builder = ctx.builder
        
        # Get operands: addmm(bias, mat1, mat2)
        operands = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                ssa = ctx.fx_value_map.get(arg.name, f"%{arg.name}")
                operands.append(ssa)
        
        if len(operands) < 3:
             # Use generic torch-mlir handling
             return self._emit_generic(ctx, node, operands)
        
        bias_ssa, mat1_ssa, mat2_ssa = operands[0], operands[1], operands[2]
        tensor_type = ctx.tensor_type
        
        # Emit linalg.matmul: accumulates into output tensor
        result = builder.fresh("matmul")
        builder.emit(
            f'{result} = linalg.matmul '
            f'ins({mat1_ssa}, {mat2_ssa} : {tensor_type}, {tensor_type}) '
            f'outs({bias_ssa} : {tensor_type}) -> {tensor_type}'
        )
        
        ctx.fx_value_map[node.name] = result
        return result

    def _emit_generic(self, ctx: "LoweringContext", node: "torch.fx.Node", operands: list[str]) -> str:
        # Use torch-mlir to generate MLIR for this operation
        mlir_text = import_aten_node_to_mlir(node)
        if mlir_text is None:
             raise RuntimeError(f"Failed to lower ATen op: {node.name} ({node.target})")
        
        return inline_torch_mlir_output(mlir_text, operands, ctx.builder)
