"""Utilities for Helion-to-MLIR lowering.

This package provides infrastructure for converting Helion kernels
(via their Device IR and FX graphs) to MLIR text representation.

Main entry points:
- generate_mlir: Generate MLIR from a bound Helion kernel
- validate_with_helion_opt: Validate emitted MLIR with helion-opt or mlir-opt

Architecture:
- IRVisitor: Walks FX graphs instruction-by-instruction
- MLIRBuilder: Text emission and SSA naming
- LoweringContext: State management during lowering
"""

# Main entry points
from .helion_mlir import generate_mlir, validate_with_helion_opt

# IR visitor for custom extensions
from .ir_visitor import IRVisitor

# Core infrastructure (for extending with new lowerings)
from .mlir_builder import (
    MLIRBuilder,
    is_concrete_size,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
    format_shape_attr,
    format_string_attr,
    format_attr_dict,
)

from .lowering_context import (
    LoweringContext,
    LoopInfo,
    KernelArgInfo,
    first_debug_name,
    resolve_extent,
    collect_reduction_block_ids,
)

__all__ = [
    # Main entry points
    "generate_mlir",
    "validate_with_helion_opt",
    # IR visitor
    "IRVisitor",
    # Builder and utilities
    "MLIRBuilder",
    "is_concrete_size",
    "torch_dtype_to_mlir_element_type",
    "format_tensor_type",
    "format_shape_attr",
    "format_string_attr",
    "format_attr_dict",
    # Lowering context
    "LoweringContext",
    "LoopInfo",
    "KernelArgInfo",
    "first_debug_name",
    "resolve_extent",
    "collect_reduction_block_ids",
]
