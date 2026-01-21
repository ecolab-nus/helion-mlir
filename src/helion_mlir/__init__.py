"""Utilities for Helion-to-MLIR lowering.

This package provides infrastructure for converting Helion kernels
(via their Device IR and FX graphs) to MLIR text representation.

Main entry points:
- generate_mlir: Generate MLIR from a bound Helion kernel
- validate_with_mlir_opt: Validate emitted MLIR with mlir-opt

Architecture:
- IRVisitor: Walks FX graphs instruction-by-instruction
- MLIROutputHelper: Text emission and SSA naming
- LoweringContext: State management during lowering
"""

# Main entry points
from .helion_mlir import generate_mlir

# IR visitor for custom extensions
from .ir_visitor import IRVisitor

# Core infrastructure (for extending with new lowerings)
from .mlir_utils import (
    MLIROutputHelper,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
    format_shape_attr,
    format_string_attr,
    format_attr_dict,
)

from .lowering_context import (
    LoweringContext,
    collect_reduction_block_ids,
)

# Torch-MLIR integration (via FxImporter)
from .torch_mlir_helper import (
    TorchMLIRNodeImporter,
    import_aten_node_to_mlir,
)

# Debug utilities
from .debug_utils import (
    print_debug_info,
    print_device_ir,
    print_nodes_with_symbols,
    print_compile_env,
    validate_with_mlir_opt,
    run_dce_cleanup,
)

__all__ = [
    # Main entry points
    "generate_mlir",
    "validate_with_mlir_opt",
    # IR visitor
    "IRVisitor",
    # Builder and utilities
    "MLIROutputHelper",
    "torch_dtype_to_mlir_element_type",
    "format_tensor_type",
    "format_shape_attr",
    "format_string_attr",
    "format_attr_dict",
    # Lowering context
    "LoweringContext",
    "collect_reduction_block_ids",
    # Debug utilities
    "print_debug_info",
    "print_device_ir",
    "print_nodes_with_symbols",
    "print_compile_env",
    "validate_with_mlir_opt",
    "run_dce_cleanup",
]
