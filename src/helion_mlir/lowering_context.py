"""Lowering context for FX-to-MLIR conversion.

This module defines the LoweringContext dataclass that holds all state
needed during the lowering process, including the MLIR builder, kernel
metadata, and mappings between FX nodes and MLIR SSA values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .mlir_builder import (
    MLIRBuilder,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)

if TYPE_CHECKING:
    from torch import Tensor
    import torch





class LoweringContext:
    """Context passed to lowering functions during FX-to-MLIR conversion.
    
    This class holds all the state needed during the lowering process,
    including references to the builder, kernel metadata, and mappings
    between FX values and MLIR SSA values.
    
    Most kernel metadata is accessed via property methods that derive
    values directly from bound_kernel, avoiding redundant storage.
    """
    
    def __init__(self, bound_kernel: Any):
        """Initialize a LoweringContext from a bound Helion kernel.
        
        Args:
            bound_kernel: The bound Helion kernel
        """
        import torch
        
        self.bound_kernel = bound_kernel
        self.builder = MLIRBuilder()
        # Always output linalg-on-tensors for now as per user request
        self.aten_output_type = "linalg-on-tensors"
        
        # Derive kernel name from the kernel function
        self.kernel_name = bound_kernel.kernel.fn.__name__
        
        # Get parameter names from the kernel signature
        param_names = list(bound_kernel.kernel.signature.parameters.keys())
        fake_args = bound_kernel.fake_args
        
        # Build arg_mlir_types: maps arg_name -> MLIR type string (for tensor args only)
        # BoundKernel.fake_args types:
        #   - FakeTensor: regular tensor argument -> becomes MLIR function parameter
        #   - int/float: ConstExpr compile-time constant -> not in function signature
        self.arg_mlir_types: dict[str, str] = {}
        
        for name, fake_arg in zip(param_names, fake_args):
            if isinstance(fake_arg, torch.Tensor):
                shape = [
                    int(s) if not isinstance(s, torch.SymInt) else None
                    for s in fake_arg.shape
                ]
                element_type = torch_dtype_to_mlir_element_type(fake_arg.dtype)
                self.arg_mlir_types[name] = format_tensor_type(shape, element_type)
        
        if not self.arg_mlir_types:
            raise ValueError("No tensor arguments found")
        

        
        # Build loop extents from block sizes
        # Key: block_id, Value: concrete extent (from shape_env)
        self.loop_extents: dict[int, int] = {}
        shape_env = bound_kernel.env.shape_env
        
        for info in bound_kernel.env.block_sizes:
            size = info.size
            if isinstance(size, int):
                self.loop_extents[info.block_id] = size
            elif isinstance(size, torch.SymInt):
                # Look up concrete value in shape_env
                sym = size._sympy_()
                if sym in shape_env.var_to_val:
                    self.loop_extents[info.block_id] = int(shape_env.var_to_val[sym])
                else:
                    # Fall back to size_hint
                    self.loop_extents[info.block_id] = bound_kernel.env.size_hint(size)
            else:
                self.loop_extents[info.block_id] = 1  # fallback
        
        # Mutable state that gets populated during lowering
        self.symbols: dict[str, str] = {}
        self.host_tensors: dict[str, str] = {}
        
        # SSA Value Tracking (populated during graph walking by IRVisitor)
        self.node_values: dict[str, str] = {}      # FX node name → MLIR SSA value
        self.node_types: dict[str, str] = {}       # FX node name → MLIR type string
        self.initial_acc_ssa: dict[str, str] = {}  # Accumulator node → initial SSA (for phi)
        
        # Graph Registry (populated before walking starts)
        self.graphs: dict[int, Any] = {}           # Graph ID → ForLoopGraphInfo
        
        # Pre-computed MLIR values (computed before graph walking)
        self.block_size_ssa: dict[int, str] = {}   # Block ID → block size SSA value
        self.reduction_trip_counts: dict[int, str] = {}  # Block ID → trip count SSA
        
        # Multi-value loop tracking
        self.loop_result_values: dict[str, Any] = {}  # Loop name → result info
    
    # -------------------------------------------------------------------------
    # Property methods - derive values from bound_kernel
    # -------------------------------------------------------------------------
    
    @property
    def device_ir(self) -> Any:
        """Get DeviceIR from bound_kernel."""
        return self.bound_kernel.host_function.device_ir
    
    @property
    def block_sizes(self) -> dict[int, Any]:
        """Get block sizes as dict from bound_kernel.env."""
        return {info.block_id: info for info in self.bound_kernel.env.block_sizes}
    
    @property
    def parallel_block_ids(self) -> list[int]:
        """Get grid block IDs for parallel loops."""
        grid_block_groups = self.device_ir.grid_block_ids
        if not grid_block_groups:
            return []
        return list(grid_block_groups[0])
    
    @property
    def env(self) -> Any:
        """Get CompileEnvironment from bound_kernel."""
        return self.bound_kernel.env

    
    def get_module_attributes(self) -> dict[str, tuple[object, str]]:
        """Get block sizes as module attributes.
        
        Returns a dict mapping attribute name to (value, type) for module attributes.
        Uses -1 for undefined (symbolic) block sizes.
        
        Naming convention:
        - loom.block_size_0, loom.block_size_1, ... = tile sizes for each block ID
        """
        attrs = {}
        # Emit block sizes using block_id-based naming
        for info in self.env.block_sizes:
            attr_name = f"loom.block_size_{info.block_id}"
            
            # Use size directly if concrete int
            if isinstance(info.size, int):
                attrs[attr_name] = (info.size, "index")
            else:
                # Use -1 for undefined/symbolic sizes
                attrs[attr_name] = (-1, "index")
        
        return attrs
    
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# def first_debug_name(names: set[str], *, fallback: str) -> str:
#     """Get the first debug name from a set, sanitizing for MLIR."""
#     for name in names:
#         if name:
#             return name.replace(".", "_").replace("-", "_")
#     return fallback

def resolve_extent(name: str, lhs: "Tensor", rhs: "Tensor") -> int:
    """Resolve the extent for a named dimension."""
    if name in {"tile_m", "m"}:
        return int(lhs.size(0))
    if name in {"tile_b", "b"}:
        return int(lhs.size(0))
    if name in {"tile_n", "n"}:
        return int(rhs.size(1))
    if name in {"tile_k", "k"}:
        return int(lhs.size(1))
    raise ValueError(f"Cannot resolve extent for loop named '{name}'.")


def collect_reduction_block_ids(device_ir: "DeviceIR") -> list[int]:
    """Collect all block IDs that are used in reduction loops."""
    block_ids: list[int] = []
    for graph_info in device_ir.graphs:
        candidate = getattr(graph_info, "block_ids", None)
        if candidate:
            for block_id in candidate:
                if block_id not in block_ids:
                    block_ids.append(block_id)
    return block_ids
